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
import os
from PIL import Image


list_ = os.listdir("./basketballImgs")
for i in list_:
    if "crop" in i:
        path = "./basketballImgs/" + i
        image = Image.open(path)
        w, h = image.size
        if w == 245 or h == 245:
            print(path)