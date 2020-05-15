"""
@author:Joker
@license: Apache Licence 
@file: utils_alex.py
@time: 2020/05/15
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""

from utils import tf


class AlexLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(AlexLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.conv1Weights = self.add_weight(name="Wconv1", shape=(11, 11, input_shape[-1], 96),
                                            initializer=tf.initializers.glorot_normal(), trainable=True)
        self.conv1Bais = self.add_weight(name="Bconv1", shape=(96,),
                                         initializer=tf.initializers.zeros(), trainable=True)
        self.conv2Weights = self.add_weight(name="Wconv2", shape=(5, 5, 96, 256),
                                            initializer=tf.initializers.glorot_normal(), trainable=True)
        self.conv2Bais = self.add_weight(name="Bconv2", shape=(256,),
                                         initializer=tf.initializers.zeros(), trainable=True)
        self.conv3Weights = self.add_weight(name="Wconv3", shape=(3, 3, 256, 192),
                                            initializer=tf.initializers.glorot_normal(), trainable=True)
        self.conv3Bais = self.add_weight(name="Bconv3", shape=(192,),
                                         initializer=tf.initializers.zeros(), trainable=True)
        self.conv4Weights = self.add_weight(name="Wconv4", shape=(3, 3, 192, 192),
                                            initializer=tf.initializers.glorot_normal(), trainable=True)
        self.conv4Bais = self.add_weight(name="Bconv3", shape=(192,),
                                         initializer=tf.initializers.zeros(), trainable=True)
        self.conv5Weights = self.add_weight(name="Wconv5", shape=(3, 3, 192, self.units),
                                            initializer=tf.initializers.glorot_normal(), trainable=True)
        self.conv5Bais = self.add_weight(name="Bconv3", shape=(self.units,),
                                         initializer=tf.initializers.zeros(), trainable=True)

        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.bn3 = tf.keras.layers.BatchNormalization(name="bn3")
        self.bn4 = tf.keras.layers.BatchNormalization(name="bn4")

    def Conv(self, inputs, filters, strides, padding, name):
        conv = tf.nn.conv2d(input=inputs, filters=filters, strides=strides, padding=padding, name=name)
        return conv

    def Pool(self, input, ksize, strides, padding, name):
        pool = tf.nn.max_pool(input=input, ksize=ksize, strides=strides, padding=padding, name=name)
        return pool

    def call(self, inputs, **kwargs):
        # layer1
        conv1 = tf.add(self.Conv(inputs, self.conv1Weights, (2, 2), "VALID", "conv1"), self.conv1Bais)
        pool1 = self.Pool(conv1, (3, 3), (2, 2), "VALID", "pool1")
        bn1 = self.bn1(pool1)
        relu1 = tf.nn.relu(bn1)
        # layer2
        conv2 = tf.add(self.Conv(relu1, self.conv2Weights, (1, 1), "VALID", "conv2"), self.conv2Bais)
        pool2 = self.Pool(conv2, (3, 3), (2, 2), "VALID", "pool2")
        bn2 = self.bn2(pool2)
        relu2 = tf.nn.relu(bn2)
        # layer3
        conv3 = tf.add(self.Conv(relu2, self.conv3Weights, (1, 1), "VALID", "conv3"), self.conv3Bais)
        bn3 = self.bn3(conv3)
        relu3 = tf.nn.relu(bn3)
        # layer 4
        conv4 = tf.add(self.Conv(relu3, self.conv4Weights, (1, 1), "VALID", "conv4"), self.conv4Bais)
        bn4 = self.bn4(conv4)
        relu4 = tf.nn.relu(bn4)
        # layer 5
        conv5 = tf.add(self.Conv(relu4, self.conv5Weights, (1, 1), "VALID", "conv5"), self.conv5Bais)
        return conv5
