#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda

from net import *
from updater import PreUpdater
from visualize import out_generated_image

import glob
import numpy as np


def make_image(f, len_mea):
    music_list = np.load(f) * 255
    music_list = music_list.transpose(0, 2, 1, 3).tolist()
    music_image_list = []
    for i in range(5, len(music_list) - len_mea + 1):
        music_image = music_list[i]
        for j in range(1, len_mea):
            music_image += music_list[i + j]
        music_image = np.array(music_image)
        if music_image.sum() > 11:
            music_image_list.append(music_image)
    return music_image_list


def load_data(gpu=-1):
    data = np.load("train_ladys_mini/datas.npy")
    return data[:8000], data[8000:]


def main():
    parser = argparse.ArgumentParser(description='VBA GAN sample')
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=3,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.  Default is cifar-10.')
    parser.add_argument('--out', '-o', default='pre_result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--n_latent', '-n', type=int, default=100,
                        help='Number of hidden units (z)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed of z at visualization stage')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# n_latent: {}'.format(args.n_latent))
    print('# epoch: {}'.format(args.epoch))
    print('')

    encoder = Encoder(n_latent=args.n_latent)
    decoder = Decoder(n_latent=args.n_latent)
    dis = Discriminator()

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        encoder.to_gpu()
        decoder.to_gpu()
        dis.to_gpu()

    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), "hook_dec")
        return optimizer
    opt_encoder = make_optimizer(encoder)
    opt_decoder = make_optimizer(decoder)
    opt_dis = make_optimizer(dis)

    train, test =  load_data(gpu=args.gpu)
    test = chainer.cuda.to_gpu(test)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)

    # Set up a trainer
    updater = PreUpdater(
        models=(encoder, decoder, dis),
        iterator=train_iter,
        optimizer={
            "encoder": opt_encoder, "decoder": opt_decoder, 'dis': opt_dis},
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        encoder, 'encoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        decoder, 'decoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'encoder/loss', 'decoder/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_generated_image(
            encoder, decoder, test[:9],
            3, 3, args.seed, args.out),
        trigger=display_interval)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)


    trainer.run()


if __name__ == '__main__':
    main()
