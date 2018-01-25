#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L


class Encoder(Chain):
    def __init__(self, n_latent, wscale=0.02):
        super(Encoder, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.c0 = L.Convolution2D(3, 64, 4, 2, 1, initialW=w)
            self.c1 = L.Convolution2D(64, 128, 4, 2, 1, initialW=w)
            self.c2 = L.Convolution2D(128, 256, 4, 2, 1, initialW=w)
            self.c3 = L.Convolution2D(256, 512, 4, 2, 1, initialW=w)
            self.l0 = L.Linear(8 * 8 * 512, 512, initialW=w)
            self.l1_mu = L.Linear(512, n_latent, initialW=w)
            self.l1_var = L.Linear(512, n_latent, initialW=w)
            self.l1_dis = L.Linear(512, 1, initialW=w)
            self.bn0 = L.BatchNormalization(64)
            self.bn1 = L.BatchNormalization(128)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(512)
            self.bn4 = L.BatchNormalization(512)

    def __call__(self, x):
        h = F.tanh(self.bn0(self.c0(x)))
        h = F.tanh(self.bn1(self.c1(h)))
        h = F.tanh(self.bn2(self.c2(h)))
        h = F.tanh(self.bn3(self.c3(h)))
        h = F.tanh(self.bn4(self.l0(h)))
        mu = self.l1_mu(h)
        ln_var = self.l1_var(h)
        p = self.l1_dis(h)
        return mu, ln_var, p
    
    def dis(self, x):
        h = F.tanh(self.bn0(self.c0(x)))
        h = F.tanh(self.bn1(self.c1(h)))
        h = F.tanh(self.bn2(self.c2(h)))
        h = F.tanh(self.bn3(self.c3(h)))
        h = F.tanh(self.bn4(self.l0(h)))
        p = self.l1_dis(h)
        return p


class Decoder(Chain):
    def __init__(self, n_latent, wscale=0.02):
        super(Decoder, self).__init__()
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(n_latent, 512, initialW=w)
            self.l1 = L.Linear(512, 8 * 8 * 512, initialW=w)
            self.dc0 = L.Deconvolution2D(512, 256, 4, 2, 1, initialW=w)
            self.dc1 = L.Deconvolution2D(256, 128, 4, 2, 1, initialW=w)
            self.dc2 = L.Deconvolution2D(128, 64, 4, 2, 1, initialW=w)
            self.dc3 = L.Deconvolution2D(64, 3, 4, 2, 1, initialW=w)
            self.bn0 = L.BatchNormalization(512)
            self.bn1 = L.BatchNormalization(8 * 8 * 512)
            self.bn2 = L.BatchNormalization(256)
            self.bn3 = L.BatchNormalization(128)
            self.bn4 = L.BatchNormalization(64)

    def __call__(self, z, sigmoid=True):
        h = F.tanh(self.bn0(self.l0(z)))
        h = F.reshape(F.tanh(self.bn1(self.l1(h))), (len(h), 512, 8, 8))
        h = F.tanh(self.bn2(self.dc0(h)))
        h = F.tanh(self.bn3(self.dc1(h)))
        h = F.tanh(self.bn4(self.dc2(h)))
        h = self.dc3(h)
        if sigmoid:
            return F.sigmoid(h)
        else:
            return h