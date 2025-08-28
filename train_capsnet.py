# -*- coding: utf-8 -*-

"""
Training a capsule network - Updated for TensorFlow 2.x / Keras 3.x (2025)

Created on Thu Jul 5 19:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

Original author: Xifeng Guo
Original source: https://github.com/XifengGuo/CapsNet-Keras
"""

# imports
from __future__ import division
from __future__ import print_function

import glob
import json
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot

# Import your custom Capsule Network layers
from libs.CapsuleNetwork import CapsuleLayer
from libs.CapsuleNetwork import Length
from libs.CapsuleNetwork import Mask
from libs.CapsuleNetwork import PrimaryCaps
from libs.PipelineUtils import shutdown

# configurations
# -----------------------------------------------------------------------------

PROCESS_SEED = None
ARCHITECTURE = 'capsnet'
DECODER_NAME = 'decoder'
INPUT_DSHAPE = (64, 64, 3)
NUM_TCLASSES = 10

DATASET_ID = 'synthetic_digits'
DATA_TRAIN = 'data/{}/imgs_train/'.format(DATASET_ID)
DATA_VALID = 'data/{}/imgs_valid/'.format(DATASET_ID)
LABEL_MAPS = 'data/{}/labelmap.json'.format(DATASET_ID)

SAMP_TRAIN = 2000
SAMP_VALID = 400
SAVE_AUGMT = False
BATCH_SIZE = 100
NUM_EPOCHS = 10  # Reduced for faster training on CPU

LEARN_RATE = 0.001
DECAY_RATE = 0.9
RECON_COEF = 0.0005
N_ROUTINGS = 1

F_OPTIMIZE = optimizers.Adam(learning_rate=LEARN_RATE)
OUTPUT_DIR = 'output/{}/{}/'.format(DATASET_ID, ARCHITECTURE)

AUTH_TOKEN = None
TELCHAT_ID = None
F_SHUTDOWN = False

# -----------------------------------------------------------------------------

# setup seed for random number generators for reproducibility
np.random.seed(PROCESS_SEED)
tf.random.set_seed(PROCESS_SEED)

# setup image data format
K.set_image_data_format('channels_last')

# setup paths for augmented data
if SAVE_AUGMT:
    aug_dir_train = os.path.join(OUTPUT_DIR, 'augmented_data/imgs_train')
    aug_dir_valid = os.path.join(OUTPUT_DIR, 'augmented_data/imgs_valid')
else:
    aug_dir_train = None
    aug_dir_valid = None

# setup paths for model architectures
mdl_dir = os.path.join(OUTPUT_DIR, 'models')
mdl_train_file = os.path.join(mdl_dir, '{}_train.json'.format(ARCHITECTURE))
mdl_evaln_file = os.path.join(mdl_dir, '{}.json'.format(ARCHITECTURE))

# setup paths for callbacks
log_dir = os.path.join(OUTPUT_DIR, 'logs')
cpt_dir = os.path.join(OUTPUT_DIR, 'checkpoints')
tbd_dir = os.path.join(OUTPUT_DIR, 'tensorboard')

log_file = os.path.join(log_dir, 'training.csv')
cpt_best = os.path.join(cpt_dir, '{}_best.weights.h5'.format(ARCHITECTURE))
cpt_last = os.path.join(cpt_dir, '{}_last.weights.h5'.format(ARCHITECTURE))

# validate paths
def validate_paths():
    flag = True
    data_dirs = [DATA_TRAIN, DATA_VALID]
    
    for directory in data_dirs:
        if not os.path.isdir(directory):
            print('[INFO] Data directory not found at {}'.format(directory))
            flag = False
    
    output_dirs = [OUTPUT_DIR, aug_dir_train, aug_dir_valid, mdl_dir, log_dir, cpt_dir, tbd_dir]
    output_dirs = [directory for directory in output_dirs if directory is not None]
    
    for directory in output_dirs:
        if not os.path.isdir(directory):
            os.makedirs(directory)
        elif len(glob.glob(os.path.join(directory, '*.*'))) > 0:
            print('[INFO] Output directory {} must be empty'.format(directory))
            flag = False
    
    return flag

# load data
def load_data():
    # image data generator configuration for training data augmentation
    data_gen_train = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1.0/255.0,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
    )

    # image data generator configuration for validation data augmentation
    data_gen_valid = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0.0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1.0/255.0,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0
    )

    # training image data generator
    data_flow_train = data_gen_train.flow_from_directory(
        directory=DATA_TRAIN,
        target_size=INPUT_DSHAPE[:-1],
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=PROCESS_SEED,
        save_to_dir=aug_dir_train,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest'
    )

    # validation image data generator
    data_flow_valid = data_gen_valid.flow_from_directory(
        directory=DATA_VALID,
        target_size=INPUT_DSHAPE[:-1],
        color_mode='rgb',
        classes=None,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=PROCESS_SEED,
        save_to_dir=aug_dir_valid,
        save_prefix='',
        save_format='png',
        follow_links=False,
        subset=None,
        interpolation='nearest'
    )

    return (data_flow_train, data_flow_valid)

# Create data generator for Capsule Networks
def create_capsnet_generator(data_generator):
    """
    Generator that yields data in the format required by Capsule Networks:
    Input: (x_batch, y_batch) - for dual input model
    Output: (y_batch, x_batch) - for dual output model (classification + reconstruction)
    """
    while True:
        x_batch, y_batch = next(data_generator)
        yield (x_batch, y_batch), (y_batch, x_batch)

# calculate margin loss - Updated for TensorFlow 2.x
def margin_loss(y_true, y_pred):
    """
    Margin loss for Capsule Networks
    """
    loss = y_true * tf.square(tf.maximum(0.0, 0.9 - y_pred)) + \
           0.5 * (1 - y_true) * tf.square(tf.maximum(0.0, y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

# build model
def build_model():
    # input layer
    x = layers.Input(shape=INPUT_DSHAPE)

    # ~~~~~~~~ Capsule network ~~~~~~~~
    # layer 1: convolution layer
    conv1 = layers.Conv2D(
        filters=256, kernel_size=9, strides=2,
        padding='valid', activation='relu',
        name='Conv_1'
    )(x)

    # layer 2: convolution layer
    conv2 = layers.Conv2D(
        filters=128, kernel_size=9, strides=1,
        padding='valid', activation='relu',
        name='Conv_2'
    )(conv1)

    # layer 3: PrimaryCaps layer (convolution layer with `squash` activation)
    primarycaps = PrimaryCaps(
        conv2, dim_capsule=8, n_channels=32,
        kernel_size=9, strides=2, padding='valid'
    )

    # layer 4: DigitCaps layer (dynamic routing)
    digitcaps = CapsuleLayer(
        num_capsule=NUM_TCLASSES,
        dim_capsule=16,
        routings=N_ROUTINGS,
        name='DigitCaps'
    )(primarycaps)

    # layer 5: Output layer
    outputcaps = Length(name=ARCHITECTURE)(digitcaps)

    # ~~~~~~~~ Decoder network ~~~~~~~~
    y = layers.Input(shape=(NUM_TCLASSES,))
    masked_train = Mask()([digitcaps, y])
    masked_evaln = Mask()(digitcaps)

    decoder = models.Sequential(name=DECODER_NAME)
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*NUM_TCLASSES))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(INPUT_DSHAPE), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=INPUT_DSHAPE, name='Reconstruction'))

    # ~~~~~~~~ Models for training and evaluation ~~~~~~~~
    model_train = models.Model([x, y], [outputcaps, decoder(masked_train)])
    model_evaln = models.Model(x, [outputcaps, decoder(masked_evaln)])

    # Compile the training model
    model_train.compile(
    optimizer=F_OPTIMIZE,
    loss=[margin_loss, 'mse'],
    metrics=['accuracy', 'mse'],  # ✅ Valid metric names
    loss_weights=[1.0, RECON_COEF]
)



    return [model_train, model_evaln]

# create callbacks
def callbacks():
    cb_log = CSVLogger(filename=log_file, append=True)
    cb_stp = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1)
    
    # Fixed ModelCheckpoint callbacks with mode='max' for accuracy
    cb_cpt_best = ModelCheckpoint(
        filepath=cpt_best,
        monitor='val_accuracy',  # Use standard metric name
        mode='max',              # ✅ Maximize accuracy
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    cb_cpt_last = ModelCheckpoint(
        filepath=cpt_last,
        monitor='val_accuracy',  # Use standard metric name  
        mode='max',              # ✅ Maximize accuracy
        save_best_only=False,
        save_weights_only=True,
        verbose=0
    )
    
    return [cb_log, cb_stp, cb_cpt_best, cb_cpt_last]  # Removed TensorBoard for speed


# plot learning curve
def plot(train_history):
    # plot training and validation loss
    pyplot.figure(figsize=(12, 4))
    
    pyplot.subplot(1, 2, 1)
    pyplot.plot(train_history.history['loss'], label='loss')
    pyplot.plot(train_history.history['val_loss'], label='val_loss')
    
    # Handle different loss naming conventions
    capsnet_loss_key = None
    decoder_loss_key = None
    
    for key in train_history.history.keys():
        if 'capsnet_loss' in key and 'val' not in key:
            capsnet_loss_key = key
        elif 'decoder_loss' in key and 'val' not in key:
            decoder_loss_key = key
    
    if capsnet_loss_key:
        pyplot.plot(train_history.history[capsnet_loss_key], label=capsnet_loss_key)
        pyplot.plot(train_history.history['val_' + capsnet_loss_key], label='val_' + capsnet_loss_key)
    
    if decoder_loss_key:
        pyplot.plot(train_history.history[decoder_loss_key], label=decoder_loss_key)
        pyplot.plot(train_history.history['val_' + decoder_loss_key], label='val_' + decoder_loss_key)
    
    pyplot.title('Training and Validation Loss')
    pyplot.xlabel('epoch')
    pyplot.ylabel('loss')
    pyplot.legend()

    # plot training and validation accuracy
    pyplot.subplot(1, 2, 2)
    
    # Handle different accuracy naming conventions
    acc_key = None
    for key in train_history.history.keys():
        if 'accuracy' in key and 'val' not in key:
            acc_key = key
            break
    
    if acc_key:
        pyplot.plot(train_history.history[acc_key], label=acc_key)
        pyplot.plot(train_history.history['val_' + acc_key], label='val_' + acc_key)
    
    pyplot.title('Training and Validation Accuracy')
    pyplot.xlabel('epoch')
    pyplot.ylabel('accuracy')
    pyplot.legend()
    
    pyplot.tight_layout()
    pyplot.savefig(os.path.join(log_dir, 'training_plots.png'), dpi=150, bbox_inches='tight')
    pyplot.show(block=False)

# train model
def train():
    # validate paths
    if not validate_paths():
        return

    # load data
    (data_flow_train, data_flow_valid) = load_data()

    # save labelmap
    with open(LABEL_MAPS, 'w') as file:
        json.dump(data_flow_train.class_indices, file)
    print('[INFO] Created labelmap')

    # build model
    print('[INFO] Building model... ', end='')
    model_train, model_evaln = build_model()
    print('done')

    model_train.summary()
    model_evaln.summary()

    # serialize models to json
    model_train_json = model_train.to_json()
    model_evaln_json = model_evaln.to_json()

    with open(mdl_train_file, 'w') as file:
        file.write(model_train_json)
    with open(mdl_evaln_file, 'w') as file:
        file.write(model_evaln_json)

    # create callbacks
    cb_list = callbacks()

    # Create generators for Capsule Network training
    train_gen = create_capsnet_generator(data_flow_train)
    valid_gen = create_capsnet_generator(data_flow_valid)

    # train model
    print('[INFO] Starting training...')
    train_history = model_train.fit(
        train_gen,
        steps_per_epoch=int(SAMP_TRAIN/BATCH_SIZE),
        epochs=NUM_EPOCHS,
        verbose=1,
        callbacks=cb_list,
        validation_data=valid_gen,
        validation_steps=int(SAMP_VALID/BATCH_SIZE)
    )

    # plot learning curve
    plot(train_history)
    
    print('[INFO] Training completed successfully!')
    return

# main
if __name__ == '__main__':
    train()
    if F_SHUTDOWN:
        shutdown()
