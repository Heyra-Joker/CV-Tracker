"""
@author:Joker
@license: Apache Licence 
@file: get_dataset.py
@time: 2020/05/09
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""

from utils import *


class GetDataset:
    def __init__(self, batchSize, epoch):
        self.groundTruthPath = os.path.join(DATASET_GROUNDTRUTH_DIR_NAME, "groundtruth_new.txt")
        self.batchSize = batchSize
        self.epoch = epoch

    def dataSet(self):
        def parseImage(path):
            path = tf.strings.join((DATASET_IMAGES_NAME, "/", path))
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            # image = tf.image.per_image_standardization(image)
            return image

        def parse(row):
            rowStrings = tf.strings.split(row, ":")
            zCorrdinate = tf.strings.to_number(tf.strings.split(rowStrings[0], ","))
            xCorrdinate = tf.strings.to_number(tf.strings.split(rowStrings[3], ","))
            zImageName = parseImage(rowStrings[1])
            xImageName = parseImage(rowStrings[2])
            return zImageName, xImageName, zCorrdinate, xCorrdinate

        dataset = tf.data.TextLineDataset([self.groundTruthPath], buffer_size=1000, num_parallel_reads=1000)
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(1000).repeat(self.epoch).batch(self.batchSize)
        return dataset

    def record(self):
        recorder = self.dataSet()
        for zImageName, xImageName, zCorrdinate, xCorrdinate in recorder.take(1):
            imagez = Image.fromarray(zImageName.numpy()[0].astype("uint8"))
            imagex = Image.fromarray(xImageName.numpy()[0].astype("uint8"))
            imagez.show()
            imagex.show()


if __name__ == '__main__':
    getDataset = GetDataset(batchSize=1, epoch=1)
    getDataset.record()
