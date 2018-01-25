#!/usr/bin/env python
# -*- coding:utf-8 -*-

import chainer
from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.encoder, self.decoder = kwargs.pop("models")
        super(Updater, self).__init__(*args, **kwargs)

    def loss_encoder(self, encoder, map_original, map_fake, mu, ln_var, p_original, p_fake, batchsize, C=1.0, D=1.0):
        loss = F.bernoulli_nll(map_original, map_fake) / map_original.size
        loss += C * F.loss.vae.gaussian_kl_divergence(mu, ln_var) / batchsize
        L1 = F.sum(F.softplus(-p_original)) / batchsize
        L2 = F.sum(F.softplus(p_fake)) / batchsize
        loss += D *  (L1 + L2) / 2
        chainer.report({'loss': loss}, encoder)
        return loss

    def loss_decoder(self, decoder, map_original, map_fake, p_fake, batchsize, C=1.0):
        loss = F.bernoulli_nll(map_original, map_fake) / map_original.size
        loss += C * F.sum(F.softplus(-p_fake)) / batchsize
        chainer.report({'loss': loss}, decoder)
        return loss

    def update_core(self):
        opt_en = self.get_optimizer("encoder")
        opt_de = self.get_optimizer("decoder")
        
        batch = self.get_iterator("main").next()
        x_original = Variable(self.converter(batch, self.device))
        xp = chainer.cuda.get_array_module(x_original.data)

        encoder, decoder = self.encoder, self.decoder
        batchsize = len(batch)

        mu, ln_var, p_original  = encoder(x_original)
        z = F.gaussian(mu, ln_var)
        x_fake = decoder(z)
        p_fake = encoder.dis(x_fake)

        opt_en.update(self.loss_encoder, encoder, x_original, x_fake, mu, ln_var, p_original, p_fake, batchsize, C=0.001,D=1.0)
        opt_de.update(self.loss_decoder, decoder, x_original, x_fake, p_fake, batchsize, C=0.01)


class PreUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.encoder, self.decoder = kwargs.pop("models")
        super(PreUpdater, self).__init__(*args, **kwargs)

    def loss_encoder(self, encoder, x_original, x_fake, mu, ln_var, p_original, p_fake, batchsize, C=1.0, D=1.0):
        loss = F.mean_squared_error(x_original, x_fake)#F.bernoulli_nll(x_original, x_fake) / x_original.size
        loss += C * F.loss.vae.gaussian_kl_divergence(mu, ln_var) / batchsize
        L1 = F.sum(F.softplus(-p_original)) / batchsize
        L2 = F.sum(F.softplus(p_fake)) / batchsize
        loss += D * (L1 + L2) / 2
        chainer.report({'loss': loss}, encoder)
        return loss

    def loss_decoder(self, decoder, x_original, x_fake, batchsize):
        loss =  F.mean_squared_error(x_original, x_fake)#F.bernoulli_nll(x_original, x_fake) / x_original.size
        chainer.report({'loss': loss}, decoder)
        return loss

    def update_core(self):
        opt_en = self.get_optimizer("encoder")
        opt_de = self.get_optimizer("decoder")
        
        batch = self.get_iterator("main").next()
        x_original = Variable(self.converter(batch, self.device))
        #print(x_original.data.mean(), x_original.data.max(), x_original.data.min())
        xp = chainer.cuda.get_array_module(x_original.data)

        encoder, decoder = self.encoder, self.decoder
        batchsize = len(batch)

        mu, ln_var, p_original = encoder(x_original)
        z = F.gaussian(mu, ln_var)
        x_fake = decoder(z)
        #print(x_fake.data.mean(), x_fake.data.max(), x_fake.data.min())
        p_fake = encoder.dis(x_fake)

        opt_en.update(self.loss_encoder, encoder, x_original, x_fake, mu, ln_var, p_original, p_fake, batchsize, C=0.0005)
        opt_de.update(self.loss_decoder, decoder, x_original, x_fake, batchsize)
        