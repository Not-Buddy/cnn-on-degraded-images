# -*- coding: utf-8 -*-

"""
Building blocks for a capsule network - Updated for TensorFlow 2.x / Keras 3.x (2025)

Created on Tue May 29 11:00:00 2018
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/cnn-on-degraded-images

Original author: Xifeng Guo
Original source: https://github.com/XifengGuo/CapsNet-Keras
"""

# imports
from __future__ import division

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.keras import layers

# Length class - Updated for TensorFlow 2.x
class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by choosing the label with the largest output.
    """
    
    def call(self, inputs, **kwargs):
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
    
    def get_config(self):
        config = super(Length, self).get_config()
        return config

# Mask class - Updated for TensorFlow 2.x
class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    """
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = tf.one_hot(indices=tf.argmax(x, axis=1), depth=x.shape[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # Masked inputs, shape = [None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * tf.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config

# CapsuleLayer class - COMPLETELY FIXED for TensorFlow 2.x
class CapsuleLayer(layers.Layer):
    """
    The capsule layer with FIXED routing algorithm that resolves einsum dimension mismatches.
    This implementation uses proper tensor operations and avoids shape conflicts.
    """
    
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform', **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix - CORRECTED SHAPE ORDER
        self.W = self.add_weight(
            shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
            initializer=self.kernel_initializer,
            name='W'
        )
        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape = [batch_size, input_num_capsule, input_dim_capsule]
        batch_size = tf.shape(inputs)[0]
        
        # Expand inputs for broadcasting
        # inputs_expand: [batch_size, input_num_capsule, 1, input_dim_capsule, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 4)
        
        # Expand W for broadcasting  
        # W_expand: [1, input_num_capsule, num_capsule, input_dim_capsule, dim_capsule]
        W_expand = tf.expand_dims(self.W, 0)
        
        # Compute prediction vectors (votes)
        # u_hat: [batch_size, input_num_capsule, num_capsule, dim_capsule]
        u_hat = tf.reduce_sum(inputs_expand * W_expand, axis=3)
        
        # Transpose to [batch_size, num_capsule, input_num_capsule, dim_capsule]
        u_hat = tf.transpose(u_hat, [0, 2, 1, 3])

        # Initialize routing logits b: [batch_size, num_capsule, input_num_capsule]
        b = tf.zeros([batch_size, self.num_capsule, self.input_num_capsule])

        # Dynamic routing algorithm
        for i in range(self.routings):
            # Compute coupling coefficients c: [batch_size, num_capsule, input_num_capsule]
            c = tf.nn.softmax(b, axis=1)
            
            # Expand c for broadcasting: [batch_size, num_capsule, input_num_capsule, 1]
            c_expand = tf.expand_dims(c, axis=-1)
            
            # Compute weighted sum: [batch_size, num_capsule, dim_capsule]
            s = tf.reduce_sum(c_expand * u_hat, axis=2)
            
            # Apply squash activation
            v = squash(s)

            if i < self.routings - 1:
                # Update routing logits
                # v_expand: [batch_size, num_capsule, 1, dim_capsule]
                v_expand = tf.expand_dims(v, axis=2)
                
                # Compute agreement: [batch_size, num_capsule, input_num_capsule]
                agreement = tf.reduce_sum(u_hat * v_expand, axis=-1)
                b += agreement

        return v

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])

    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'routings': self.routings
        }
        base_config = super(CapsuleLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# squash function - Updated for TensorFlow 2.x
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

# PrimaryCaps function - Updated for TensorFlow 2.x
def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                          activation='relu', name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, output_shape=lambda input_shape: input_shape, name='primarycap_squash')(outputs)
