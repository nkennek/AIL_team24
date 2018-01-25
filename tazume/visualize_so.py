#!/usr/bin/env python

import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import chainer
import chainer.cuda
from chainer import Variable


def save_images(x, rows, cols, dst, authenticit, iteration):
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3))[:, :, ::-1]
    preview_dir = '{}/preview_{}'.format(dst, authenticit)
    preview_path = preview_dir + \
                   '/image{:0>8}.png'.format(iteration)
    if not os.path.exists(preview_dir):
        os.makedirs(preview_dir)
    plt.imsave(preview_path, np.squeeze(x))


def out_generated_image(encoder, decoder, test, rows, cols, seed, dst):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_images = rows * cols
        x_test = test[:n_images]
        xp = chainer.cuda.get_device_from_array(test)
        with chainer.using_config('train', False):
            z, _, _ = encoder(x_test)
            x = decoder(z)

        x_test = np.asarray(np.clip(chainer.cuda.to_cpu(x_test) * 255, 0.0, 255.0), dtype=np.uint8)
        x = chainer.cuda.to_cpu(x.data)
        np.random.seed()
        x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)

        save_images(x_test, rows, cols, dst, "original", trainer.updater.iteration)
        save_images(x, rows, cols, dst, "fake", trainer.updater.iteration)
    return make_image
