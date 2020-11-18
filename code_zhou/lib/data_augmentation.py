#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 12:16:00 2019

@author: xulin
"""

import tensorflow as tf
import numpy as np

def decode_img(path_tensor, img_size, channels = 3):
    tf.random.set_random_seed(2333)
    img = tf.read_file(path_tensor)
    img = tf.image.decode_image(img, channels)
    img = tf.image.random_crop(img, img_size)
    return img


## Data augmentation function from Voxnet, which randomly translates
## and/or horizontally flips a chunk of data.
def jitter_chunk(src, max_ij, max_k):
    dst = src.copy()
    if np.random.binomial(1, .2):
        dst[:, ::-1, :, :, :] = dst
    if np.random.binomial(1, .2):
        dst[:, :, ::-1, :, :] = dst
    shift_ijk = [np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_ij, max_ij),
                 np.random.random_integers(-max_k, max_k)]
    for axis, shift in enumerate(shift_ijk):
        if shift != 0:
            # beware wraparound
            dst = np.roll(dst, shift, axis+1)
    return dst