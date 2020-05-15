"""
@author:Joker
@license: Apache Licence 
@file: siamese_fc.py
@time: 2020/05/15
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""

from utils import *
from utils_alex import AlexLayer
from get_dataset import GetDataset
from utils_response import ResponseLayer


class SiameseFCModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(SiameseFCModel, self).__init__(**kwargs)
        self.alexLayer = AlexLayer(units=192)
        self.responseLayer = ResponseLayer()

    def call(self, inputs, training=None, mask=None):
        zImages = inputs[0]
        xImages = inputs[1]
        zConv = self.alexLayer(zImages)
        xConv = self.alexLayer(xImages)
        response = self.responseLayer((zConv, xConv))
        return response


class SiameseFC:
    def __init__(self, epoch, batchSize):
        self.epoch = epoch
        self.batchSize = batchSize
        self.model = SiameseFCModel()

    def train(self, lr):
        getDataset = GetDataset(batchSize=self.batchSize, epoch=self.epoch)
        recorder = getDataset.record()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        self.model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "auc"])
        tensorboard_callback = tf.keras.callbacks.TensorBoard(TENSORBOARD_DIR, histogram_freq=1,
                                                              update_freq=100,
                                                              profile_batch=0)
        self.model.fit(x=recorder, callbacks=[tensorboard_callback], epochs=self.epoch)
        tf.saved_model.save(self.model, MODEL_SAVE_DIR)
        print('export saved model.')

if __name__ == '__main__':
    siameseFC = SiameseFC(epoch=2, batchSize=8)
    siameseFC.train(lr=0.001)
