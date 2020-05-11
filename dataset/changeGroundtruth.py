"""
@author:Joker
@license: Apache Licence
@file: changeGroundtruth.py
@time: 2020/05/09
@blog: https://github.com/woaij100
@description: --

🤡
code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
🤡
"""

import os
import numpy as np
from cv2 import cv2

DATASET_IMAGES_NAME = "basketballImgs"


class ChangeGround:
    """
    Change 8 coordinate to 4 corrdinate.
    """

    def __init__(self, txtDirName, isShow=False, imageDir=None):
        """
        :param txtDirName: groundtruth.txt dir name
        :param isShow: show image in new corrdinate
        :param imageDir: if show image, need set image dir.
        """
        self.isShow = isShow
        self.imageDir = imageDir
        self.groundtruthPath = os.path.join(os.getcwd(), txtDirName, "groundtruth.txt")
        groundtruthPathNew = os.path.join(os.getcwd(), txtDirName, "groundtruth_new.txt")
        self.groundtruthPathNewOpen = open(groundtruthPathNew, "a")
        self.images = os.listdir(imageDir)
        self.images.sort()

    def readTxT(self):
        with open(self.groundtruthPath) as ground:
            lines = ground.readlines()
            return lines

    def getCXY(self, region, center):
        """
        :param region: 8 corrdinate array.
        :param center: is return center with cx, cy
        :return: 4 corrdinate.
        """
        cx = np.mean(region[::2])
        cy = np.mean(region[1::2])
        x1 = np.min(region[::2])
        x2 = np.max(region[::2])
        y1 = np.min(region[1::2])
        y2 = np.max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        if center:
            return cx.astype(np.int64), cy.astype(np.int64), w.astype(np.int64), h.astype(np.int64)
        else:
            return (cx - w / 2).astype("float32"), (cy - h / 2).astype("float32"), w.astype("float32"), h.astype(
                "float32")

    def imagePad(self, image, pad_width, constant_values, constant_position):
        """
        padding Image
        """
        if constant_position == "left":
            constant_values = np.ones((image.shape[0], pad_width, 3)) * constant_values
            image = np.concatenate((constant_values, image), axis=1)
        elif constant_position == "right":
            constant_values = np.ones((image.shape[0], pad_width, 3)) * constant_values
            image = np.concatenate((image, constant_values), axis=1)
        elif constant_position == "up":
            constant_values = np.ones((pad_width, image.shape[1], 3)) * constant_values
            image = np.concatenate((constant_values, image), axis=0)
        else:
            constant_values = np.ones((pad_width, image.shape[1], 3)) * constant_values
            image = np.concatenate((image, constant_values), axis=0)

        return image.astype("uint8")

    def CropZAndX(self, path, corrdinate, halfOffset, mode="x"):
        """
        获取Z与X
        :param path: image path
        :param corrdinate: bbox corrdinate
        :param halfOffset: crop half offset, like 127 => Z, then halfOffset=127/2
        :param mode: get "Z" or "X"
        :return:
        """
        x, y, w, h = corrdinate
        centerX = x + w / 2
        centerY = y + h / 2
        if mode == "x":
            # 随机在一定范围内游走
            augment = np.random.randint(-10, 10)
            centerX += augment
            centerY += augment

        image = cv2.imread(path)
        height, width, _ = image.shape
        meanB = np.floor(np.mean(image[:, :, 0])).astype(np.int)
        meanG = np.floor(np.mean(image[:, :, 1])).astype(np.int)
        meanR = np.floor(np.mean(image[:, :, 2])).astype(np.int)

        # crop
        cropLeft = np.ceil(centerX - halfOffset).astype(np.int)
        cropRight = np.ceil(centerX + halfOffset).astype(np.int)
        cropUp = np.ceil(centerY - halfOffset).astype(np.int)
        cropDown = np.ceil(centerY + halfOffset).astype(np.int)

        # 处理越界问题,使用RGB均值pad
        # pad left
        if cropLeft < 0:  # 宽越界左边
            image = self.imagePad(image, -cropLeft, np.array([meanB, meanG, meanR]), "left")  # B,G,R=>height,width
            cropLeft = cropLeft * -1
        elif cropRight > width:  # 宽越界右边
            image = self.imagePad(image, cropRight - width, np.array([meanB, meanG, meanR]), "right")
        elif cropUp < 0:  # 高越界上边
            image = self.imagePad(image, -cropUp, np.array([meanB, meanG, meanR]), "up")
            cropUp = cropUp * -1
        elif cropDown > height:  # 高越界下边
            image = self.imagePad(image, cropDown - height, np.array([meanB, meanG, meanR]), "down")

        cropImage = image[cropUp:cropDown, cropLeft:cropRight]
        cropImagePath = path.split(".")[0] + "-crop" + "-" + mode + "." + path.split(".")[1]
        cv2.imwrite(cropImagePath, cropImage)
        return cropImagePath.split("/")[-1]

    def createZX(self, ZXData):
        m = ZXData.shape[0]
        for i in range(m):
            ZIndex = np.random.randint(0, m)
            XIndex = np.random.randint(np.maximum(0, ZIndex - 5), np.minimum(m, ZIndex + 5))  # fps +- 5
            ZimagePath = os.path.join(DATASET_IMAGES_NAME, self.images[ZIndex])
            XimagePath = os.path.join(DATASET_IMAGES_NAME, self.images[XIndex])
            Zcorrdinate = ZXData[ZIndex]
            Xcorrdinate = ZXData[XIndex]
            Zimage = self.CropZAndX(ZimagePath, Zcorrdinate, 63.5, mode="z")
            Ximage = self.CropZAndX(XimagePath, Xcorrdinate, 127.5, mode="x")
            # line : z corrdinate:z image name:x image name: x corrdinate
            self.groundtruthPathNewOpen.write(
                "".join((",".join(Zcorrdinate.astype(np.str)), ":", Zimage, ":", Ximage, ":",
                         ",".join(Xcorrdinate.astype(np.str)))) + "\n"
            )

    def change(self):
        ZXData = []
        lines = self.readTxT()
        for index, line in enumerate(lines):
            bbox = line.strip("\n").split(",")
            x, y, w, h = self.getCXY(np.array(bbox).astype(np.float), center=False)
            centerX, centerY = int((x + w / 2)), int((y + h / 2))
            if self.isShow:
                image = cv2.imread(os.path.join(self.imageDir, self.images[index]))
                cv2.rectangle(image, (x, y), (x + w, y + h), (80, 255, 60), 1)
                cv2.circle(image, (centerX, centerY), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imshow("image", image)
                cv2.waitKey(10)
            ZXData.append([x, y, w, h])
        ZXData = np.array(ZXData)
        self.createZX(ZXData)
        self.groundtruthPathNewOpen.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    change_ground = ChangeGround("basketball", isShow=False, imageDir="basketballImgs")
    change_ground.change()
