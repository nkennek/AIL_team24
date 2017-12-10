# sys.path.append(os.path.dirname(__file__))

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from instance_normalization import InstanceNormalization


def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.data.shape)
    else:
        return h


class ResBlock(chainer.Chain):
    def __init__(self, ch, norm='instance', activation=F.relu):
        super(ResBlock, self).__init__()
        self.activation = activation
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            self.c1 = L.Convolution2D(ch, ch, 3, 1, 1, initialW=w)
            if norm == 'batch':
                self.norm0 = L.BatchNormalization(ch)
                self.norm1 = L.BatchNormalization(ch)
            elif norm == 'instance':
                self.norm0 = InstanceNormalization(ch)
                self.norm1 = InstanceNormalization(ch)

    def __call__(self, x):
        h = self.c0(x)
        h = self.norm0(h)
        h = self.activation(h)
        h = self.c1(h)
        h = self.norm1(h)
        return h + x


class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, ksize=3, pad=1, norm='instance', sample='down',
                 activation=F.relu, dropout=False, noise=False):
        super(CBR, self).__init__()
        self.activation = activation
        self.dropout = dropout
        self.sample = sample
        self.noise = noise
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            if sample == 'down':
                self.c = L.Convolution2D(ch0, ch1, ksize, 2, pad, initialW=w)
            elif sample == 'none-9':
                self.c = L.Convolution2D(ch0, ch1, 9, 1, 4, initialW=w)
            elif sample == 'none-7':
                self.c = L.Convolution2D(ch0, ch1, 7, 1, 3, initialW=w)
            elif sample == 'none-5':
                self.c = L.Convolution2D(ch0, ch1, 5, 1, 2, initialW=w)
            else:
                self.c = L.Convolution2D(ch0, ch1, ksize, 1, pad, initialW=w)
            if norm == 'batch':
                if self.noise:
                    self.norm = L.BatchNormalization(ch1, use_gamma=False)
                else:
                    self.norm = L.BatchNormalization(ch1)
            elif norm == 'instance':
                if self.noise:
                    self.norm = InstanceNormalization(ch1, use_gamma=False)
                else:
                    self.norm = InstanceNormalization(ch1)
            else:
                raise ValueError('invalid norm parameter for CBR')

    def __call__(self, x):
        if self.sample == "down" or self.sample == "none" or self.sample == 'none-9' or self.sample == 'none-7' or self.sample == 'none-5':
            h = self.c(x)
        elif self.sample == "up":
            h = F.unpooling_2d(x, 2, 2, 0, cover_all=False)
            h = self.c(h)
        else:
            print("unknown sample method %s" % self.sample)
        h = self.norm(h)
        if self.noise:
            h = add_noise(h)
        if self.dropout:
            h = F.dropout(h)
        if self.activation is not None:
            h = self.activation(h)
        return h


class Generator(chainer.Chain):
    def __init__(self, norm='instance', n_resblock=9):
        super(Generator, self).__init__()
        self.n_resblock = n_resblock
        with self.init_scope():
            # nn.ReflectionPad2d in original
            self.c1 = CBR(3, 32, norm=norm, sample='none-7')
            self.c2 = CBR(32, 64, norm=norm, sample='down')
            self.c3 = CBR(64, 128, norm=norm, sample='down')
            for i in range(n_resblock):
                setattr(self, 'c' + str(i + 4), ResBlock(128, norm=norm))
            # nn.ConvTranspose2d in original
            setattr(self, 'c' + str(n_resblock + 4),
                    CBR(128, 64, norm=norm, sample='up'))
            setattr(self, 'c' + str(n_resblock + 5),
                    CBR(64, 32, norm=norm, sample='up'))
            setattr(self, 'c' + str(n_resblock + 6),
                    CBR(32, 3, norm=norm, sample='none-7', activation=F.tanh))

    def __call__(self, x):
        h = self.c1(x)
        for i in range(2, self.n_resblock + 7):
            h = getattr(self, 'c' + str(i))(h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, norm='instance', in_ch=3, n_down_layers=3):
        super(Discriminator, self).__init__()
        base = 64
        ksize = 4
        pad = 2
        self.n_down_layers = n_down_layers

        with self.init_scope():
            self.c0 = CBR(in_ch, 64, ksize=ksize, pad=pad, norm=norm,
                          sample='down', activation=F.leaky_relu,
                          dropout=False, noise=False)

            for i in range(1, n_down_layers):
                setattr(self, 'c' + str(i),
                        CBR(base, base * 2, ksize=ksize, pad=pad, norm=norm,
                            sample='down',
                            activation=F.leaky_relu, dropout=False,
                            noise=False))
                base *= 2

            setattr(self, 'c' + str(n_down_layers),
                    CBR(base, base * 2, ksize=ksize, pad=pad, norm=norm,
                        sample='none',
                        activation=F.leaky_relu, dropout=False,
                        noise=False))
            base *= 2

            setattr(self, 'c' + str(n_down_layers + 1),
                    CBR(base, 1, ksize=ksize, pad=pad, norm=norm, sample='none',
                        activation=None, dropout=False, noise=False))

    def __call__(self, x_0):
        h = self.c0(x_0)
        for i in range(1, self.n_down_layers + 2):
            h = getattr(self, 'c' + str(i))(h)
        return h
