import os
import numpy as np
import cv2


def testTheAccuracyOfNetwork(w, b, dir):
    count = 0
    width_trans = 425
    height_trans = 302
    pic_matrix = np.zeros((width_trans * height_trans,), dtype=np.float)
    NonCOVID_file_num = len(os.listdir(os.path.join(dir, "test_NonCOVID")))
    COVID_file_num = len(os.listdir(os.path.join(dir, "test_COVID")))
    label_matrix = np.hstack((np.zeros(NonCOVID_file_num), np.ones(COVID_file_num)))
    for pic in os.listdir(os.path.join(dir, "test_NonCOVID")):
        img = cv2.imread(os.path.join(dir, "test_NonCOVID", pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        score = img_1D.dot(w) + b
        if score[0][0] > score[0][1]:
            count += 1
    for pic in os.listdir(os.path.join(dir, "test_COVID")):
        img = cv2.imread(os.path.join(dir, "test_COVID", pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        pscore = img_1D.dot(w) + b
        if score[0][0] < score[0][1]:
            count += 1
    print("准确率：%d%%" % (count / (COVID_file_num + NonCOVID_file_num) * 100))
