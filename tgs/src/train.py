#!/usr/bin/env python

"""
Training for TGS Kaggle competition
"""

import argparse
import os
import sys
import warnings

import keras
import keras.preprocessing.image
import tensorflow as tf

from utils.load_data import load_tgs_data
from utils.losses import focal
from models.vgg import vggunet



def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def freeze_model(model):
    """ Set all layers in a model to non-trainable.
    The weights for these layers will not be updated during training.
    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model

def create_models(backbone, num_classes, weights, multi_gpu=0, freeze_backbone=False):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone           : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """
    modifier = freeze_model if freeze_backbone else None

    # Keras recommends initialising a multi-gpu model on the CPU to ease weight sharing, and to prevent OOM errors.
    # optionally wrap in a parallel model
    model = backbone(num_classes, weights)

    # # make prediction model
    # prediction_model = retinanet_bbox(model=model)

    # compile model
    model.compile(
        loss= focal(),
        optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
    )

    return model


def create_generator(X, y):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    # common_args = {
    #     'batch_size'       : args.batch_size,
    #     'image_min_side'   : args.image_min_side,
    #     'image_max_side'   : args.image_max_side,
    #     'preprocess_image' : preprocess_image,
    # }

    # create random transform generator for augmenting training data
    transform_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        fill_mode='reflect',
        horizontal_flip=False,
        vertical_flip=False,
        rescale=(128,128)
    )

    transform_generator = transform_generator.flow(X, y)

    return transform_generator


def check_args(parsed_args):
    """ Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.multi_gpu > 1 and parsed_args.batch_size < parsed_args.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             parsed_args.multi_gpu))

    if parsed_args.multi_gpu > 1 and parsed_args.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(parsed_args.multi_gpu,
                                                                                                parsed_args.snapshot))

    if parsed_args.multi_gpu > 1 and not parsed_args.multi_gpu_force:
        raise ValueError("Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if 'resnet' not in parsed_args.backbone:
        warnings.warn('Using experimental backbone {}. Only resnet50 has been properly tested.'.format(parsed_args.backbone))

    return parsed_args


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    # subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    # subparsers.required = True

    # coco_parser = subparsers.add_parser('tgs')
    # coco_parser.add_argument('coco_path', help='Path to dataset directory (ie. /tmp/COCO).')

    # pascal_parser = subparsers.add_parser('pascal')
    # pascal_parser.add_argument('pascal_path', help='Path to dataset directory (ie. /tmp/VOCdevkit).')

    # kitti_parser = subparsers.add_parser('kitti')
    # kitti_parser.add_argument('kitti_path', help='Path to dataset directory (ie. /tmp/kitti).')


    group = parser
    group.add_argument('--snapshot',          help='Resume training from a snapshot.')
    group.add_argument('--imagenet-weights',  help='Initialize the model with pretrained imagenet weights. This is the default behaviour.', action='store_const', const=True, default=True)
    group.add_argument('--weights',           help='Initialize the model with weights from a file.', default='/home/ryan/cs/datasets/tgs/weights/vgg16_weights_th_dim_ordering_th_kernels.h5')
    group.add_argument('--no-weights',        help='Don\'t initialize the model with any weights.', dest='imagenet_weights', action='store_const', const=False)

    parser.add_argument('--backbone',        help='Backbone model used by retinanet.', default='resnet50', type=str)
    parser.add_argument('--batch-size',      help='Size of the batches.', default=1, type=int)
    parser.add_argument('--gpu',             help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--multi-gpu',       help='Number of GPUs to use for parallel processing.', type=int, default=0)
    parser.add_argument('--multi-gpu-force', help='Extra flag needed to enable (experimental) multi-gpu support.', action='store_true')
    parser.add_argument('--epochs',          help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps',           help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--snapshot-path',   help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--no-snapshots',    help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation',   help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--freeze-backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)

    return check_args(parser.parse_args(args))

def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    X, y = load_tgs_data('/home/ryan/cs/datasets/tgs/train', train=True)
    X_test, y_test = load_tgs_data('/home/ryan/cs/datasets/tgs/test', train=False)

    trn_generator = create_generator(X[:100], y[:100])
    val_generator = create_generator(X[100:], y[100:])

    weights = args.weights
    # default to imagenet if nothing else is specified
    if weights is None and args.imagenet_weights:
        weights = backbone.download_imagenet()

    print('Creating model...')
    model = create_models(
        backbone=vggunet,
        num_classes=1,
        weights=weights,
        multi_gpu=args.multi_gpu,
        freeze_backbone=args.freeze_backbone
    )

    # print model summary
    # print(model.summary())



    # start training
    model.fit_generator(
        generator=trn_generator,
        validation_data=val_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
    )


if __name__ == '__main__':
    main()

