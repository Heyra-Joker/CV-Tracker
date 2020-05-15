"""
@author:Joker
@license: Apache Licence 
@file: test2.py
@time: 2020/05/15
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""

import tensorflow as tf

model_loaded = tf.keras.models.load_model('./save_models/siamese_model')

print(model_loaded.summary())