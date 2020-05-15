"""
@author:Joker
@license: Apache Licence 
@file: utils_response.py
@time: 2020/05/15
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""
from utils import tf


class ResponseLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ResponseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.responseBais = self.add_weight(name="responsebais", shape=(1,),
                                            initializer=tf.initializers.zeros(), trainable=True)

    def _translation_match(self, z, x):
        """
        一对一卷积
        """
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        output = tf.keras.backend.conv2d(x, z, strides=[1, 1, 1, 1], padding='valid')
        return output

    def call(self, inputs, **kwargs):
        z = inputs[0]
        x = inputs[1]
        response = tf.map_fn(lambda conv: self._translation_match(conv[0], conv[1]), (z, x), dtype=tf.float32)
        response = tf.squeeze(response, axis=[1, 4])
        response = tf.add(response, self.responseBais)
        return response
