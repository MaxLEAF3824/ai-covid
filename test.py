import os
import cv2
import numpy as np
pic_dir = "D:\\Coding\\Others\\COVID"
all_dir = os.path.join(pic_dir, "all")
height, width, scale = 0, 0, 0
for pic in os.listdir(all_dir):
    # print(os.path.join(non_covid_dir, pic))
    img = cv2.imread(os.path.join(all_dir, pic))
    height += img.shape[0]
    width += img.shape[1]
    scale += width / height
print([height / 746, width / 746, scale / 746])
