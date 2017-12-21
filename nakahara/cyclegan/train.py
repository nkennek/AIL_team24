import argparse
import os
from datetime import datetime as dt

import matplotlib

matplotlib.use('Agg')

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import net
from dataset import Dataset
from updater import Updater
from visualization import visualize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='datasets')
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--vis_folder', '-e', default='visualization',
                        help='Directory to output the visualization result')

    parser.add_argument('--learning_rate_g', type=float, default=0.0002,
                        help='Learning rate for generator')
    parser.add_argument('--learning_rate_d', type=float, default=0.0002,
                        help='Learning rate for discriminator')

    parser.add_argument('--gen_class', default='Generator',
                        help='Default generator class')
    parser.add_argument('--dis_class', default='Discriminator',
                        help='Default discriminator class')
    parser.add_argument('--load_gen_f_model', default='',
                        help='load generator model')
    parser.add_argument('--load_gen_g_model', default='',
                        help='load generator model')
    parser.add_argument('--load_dis_x_model', default='',
                        help='load discriminator model')
    parser.add_argument('--load_dis_y_model', default='',
                        help='load discriminator model')
    parser.add_argument('--norm', default='instance',
                        choices=['instance', 'batch'])

    parser.add_argument('--lambda_A', type=float, default=10.0,
                        help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='weight for cycle loss (B -> A -> B)')

    # Note that this is different from original implementation
    parser.add_argument('--lambda_identity', type=float, default=0.0,
                        help='lambda for l1 loss to stop unnecessary changes')

    parser.add_argument('--flip', type=int, default=1,
                        help='flip images for data augmentation')
    parser.add_argument('--resize_to', type=int, default=286,
                        help='resize the image to')
    parser.add_argument('--crop_to', type=int, default=256,
                        help='crop the resized image to')
    parser.add_argument('--load_dataset', default=None,
                        help='load dataset')

    parser.add_argument('--lrdecay_start', type=int, default=100,
                        help='anneal the learning rate (by epoch)')
    parser.add_argument('--lrdecay_period', type=int,
                        default=100, help='period to anneal the learning')

    args = parser.parse_args()
    print(args)

    root = args.root

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    gen_g = getattr(net, args.gen_class)(args.norm)
    dis_x = getattr(net, args.dis_class)(args.norm)
    gen_f = getattr(net, args.gen_class)(args.norm)
    dis_y = getattr(net, args.dis_class)(args.norm)

    if args.load_gen_g_model != '':
        serializers.load_npz(args.load_gen_g_model, gen_g)
        print('Generator G(X->Y) model loaded')

    if args.load_gen_f_model != '':
        serializers.load_npz(args.load_gen_f_model, gen_f)
        print('Generator F(Y->X) model loaded')

    if args.load_dis_x_model != '':
        serializers.load_npz(args.load_dis_x_model, dis_x)
        print('Discriminator X model loaded')

    if args.load_dis_y_model != '':
        serializers.load_npz(args.load_dis_y_model, dis_y)
        print('Discriminator Y model loaded')

    if not os.path.exists(args.vis_folder):
        os.makedirs(args.vis_folder)

    # select GPU
    if args.gpu >= 0:
        gen_g.to_gpu()
        gen_f.to_gpu()
        dis_x.to_gpu()
        dis_y.to_gpu()
        print('use gpu {}'.format(args.gpu))

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        return optimizer

    opt_g = make_optimizer(gen_g, alpha=args.learning_rate_g)
    opt_f = make_optimizer(gen_f, alpha=args.learning_rate_g)
    opt_x = make_optimizer(dis_x, alpha=args.learning_rate_d)
    opt_y = make_optimizer(dis_y, alpha=args.learning_rate_d)

    train_dir = root if args.load_dataset is None else os.path.join(
        args.load_dataset)
    train_A_dataset = Dataset(
        path=os.path.join(train_dir, 'trainA'), flip=args.flip,
        resize_to=args.resize_to, crop_to=args.crop_to)
    train_B_dataset = Dataset(
        path=os.path.join(train_dir, 'trainB'), flip=args.flip,
        resize_to=args.resize_to, crop_to=args.crop_to)

    if args.batch_size > 1:
        train_A_iter = chainer.iterators.MultiprocessIterator(
            train_A_dataset, args.batch_size, n_processes=3)
        train_B_iter = chainer.iterators.MultiprocessIterator(
            train_B_dataset, args.batch_size, n_processes=3)
    else:
        train_A_iter = chainer.iterators.SerialIterator(
            train_A_dataset, args.batch_size)
        train_B_iter = chainer.iterators.SerialIterator(
            train_B_dataset, args.batch_size)

    # Set up a trainer
    updater = Updater(
        models=(gen_g, gen_f, dis_x, dis_y),
        iterator={
            'main': train_A_iter,
            'train_B': train_B_iter,
        },
        optimizer={
            'gen_g': opt_g,
            'gen_f': opt_f,
            'dis_x': opt_x,
            'dis_y': opt_y
        },
        device=args.gpu,
        params={
            'lambda_A': args.lambda_A,
            'lambda_B': args.lambda_B,
            'lambda_identity': args.lambda_identity,
            'batch_size': args.batch_size,
            'image_size': args.crop_to,
            'lrdecay_start': args.lrdecay_start,
            'lrdecay_period': args.lrdecay_period,
            'dataset': train_A_dataset
        })

    log_interval = (100, 'iteration')
    model_save_interval = (10000, 'iteration')
    out = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))
    trainer = training.Trainer(updater, (
        args.lrdecay_start + args.lrdecay_period, 'epoch'), out=out)
    trainer.extend(extensions.snapshot_object(
        gen_g, 'gen_g{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        gen_f, 'gen_f{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_x, 'dis_x{.updater.iteration}.npz'), trigger=model_save_interval)
    trainer.extend(extensions.snapshot_object(
        dis_y, 'dis_y{.updater.iteration}.npz'), trigger=model_save_interval)

    log_keys = ['epoch', 'iteration', 'gen_g/loss_cycle', 'gen_f/loss_cycle',
                'gen_g/loss_id', 'gen_f/loss_id', 'gen_g/loss_gen',
                'gen_f/loss_gen', 'dis_x/loss', 'dis_y/loss']
    trainer.extend(
        extensions.LogReport(keys=log_keys, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=20))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(
                ['gen_g/loss_cycle', 'gen_f/loss_cycle', 'gen_g/loss_id',
                 'gen_f/loss_id', 'gen_g/loss_gen', 'gen_f/loss_gen',
                 'dis_x/loss', 'dis_y/loss'], 'iteration',
                trigger=(100, 'iteration'), file_name='loss.png'))

    trainer.extend(
        visualize(gen_g, gen_f, args.vis_folder),
        trigger=(1, 'epoch')
    )

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
