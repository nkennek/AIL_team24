#!/usr/bin/env python
# -*- coding:utf-8 -*

import numpy as np

import chainer
from chainer import cuda
from chainer import function_node
from chainer.utils import type_check


def initalize_naive_kernel(width, xp=np):
    kernel = xp.zeros((width, width), dtype=xp.float32)
    center = int(width/2)
    for i in range(width):
        for j in range(width):
            if xp.abs(i-center)<width/6 and xp.abs(j-center)<width/6:
                kernel[i][j] = 1.
                continue

            bigger_idx = max(abs(i-center), abs(j-center))
            val = (center-bigger_idx)/(2*center/3)
            kernel[i][j] = val

    return kernel


def calc_rec_loss(x0, x1, kernel=None):
    """Mean absolute error function.
    This function computes mean absolute error between two variables. The mean
    is taken over the minibatch.
    """
    if kernel is None:
        return chainer.functions.mean_absolute_error(x0, x1)

    return MeanAbsoluteErrorCustomized().apply((x0, x1, kernel))[0]


class MeanAbsoluteErrorCustomized(function_node.FunctionNode):

    """Mean absolute error function.
    https://github.com/chainer/chainer/blob/v3.2.0/chainer/functions/loss/mean_absolute_error.py
    """

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        type_check.expect(
            in_types[0].dtype == np.float32,
            in_types[1].dtype == np.float32,
            in_types[2].dtype == np.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1, kernel = inputs
        self.diff = kernel*(x0 - x1)
        diff = self.diff.ravel()
        return np.array(abs(diff).sum() / diff.size, dtype=diff.dtype),

    def forward_gpu(self, inputs):
        x0, x1, kernel = inputs
        self.diff = kernel*(x0 - x1)
        diff = self.diff.ravel()
        return abs(diff).sum() / diff.dtype.type(diff.size),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        coeff = gy * gy.data.dtype.type(1. / self.diff.size)
        coeff = chainer.functions.broadcast_to(coeff, self.diff.shape)
        gx0 = coeff * cuda.get_array_module(gy.data).sign(self.diff)
        return gx0, -gx0
