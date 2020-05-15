"""
@author:Joker
@license: Apache Licence 
@file: utils.py
@time: 2020/05/09
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""
import datetime
import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


# Current Dir
CURRENT_DIR = os.getcwd()

# Dataset Name
DATASET_IMAGES_NAME = "dataset/basketballImgs"
DATASET_GROUNDTRUTH_DIR_NAME = "dataset/basketball"

# AlexNet weights
ALEXNET_WEIGHTS_PATH = "./weights/bvlc_alexnet.npy"

# Tensorboard Dir
TENSORBOARD_DIR = os.path.join(CURRENT_DIR, "model_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# get GT Config, 超参数
RPos = 16  # Manhattan positive
RNeg = 0 # Manhattan negtive.



def constructGtScoreMaps(response_size, stride):
    """Construct a batch of groundtruth score maps
    Args:
      response_size: A list or tuple with two elements [ho, wo]
      stride: Embedding stride e.g., 8
    Return:
      A float tensor of shape [batch_size] + response_size
      表示当响应图中某位置𝑢和响应图中目标位置𝑐的距离乘以比例因子𝑘后小于𝑅则为正样本：曼哈顿距离
    """

    def _logistic_label(X, Y, rPos, rNeg):
        # dist_to_center = tf.sqrt(tf.square(X) + tf.square(Y))  # L2 metric
        dist_to_center = tf.abs(X) + tf.abs(Y)  # Block metric
        yu = tf.where(dist_to_center <= rPos, tf.ones_like(X),
                     tf.where(dist_to_center < rNeg, 0.5 * tf.ones_like(X), tf.zeros_like(X)))
        return yu

    def get_center(x):
        return (x - 1.) / 2.

    ho = response_size[0]
    wo = response_size[1]
    y = tf.cast(tf.range(0, ho), dtype=tf.float32) - get_center(ho) # [-8, .... 8] like gt "k"
    x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [Y, X] = tf.meshgrid(y, x)

    # gt "R"
    rPos = RPos / stride
    rNeg = RNeg / stride
    gt = _logistic_label(X, Y, rPos, rNeg)
    # Duplicate a batch of maps
    # gt = tf.reshape(gt, [1] + response_size)
    return gt

