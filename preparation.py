import os
import cv2
import numpy as np


def importAndPrepocessPictures(dir):
    width_trans = 425
    height_trans = 302
    pic_matrix = np.zeros((width_trans * height_trans,), dtype=np.float)
    NonCOVID_file_num = len(os.listdir(os.path.join(dir, "train_NonCOVID")))
    COVID_file_num = len(os.listdir(os.path.join(dir, "train_COVID")))
    label_matrix = np.hstack((np.zeros(NonCOVID_file_num), np.ones(COVID_file_num)))
    # read pic from train_NonCOVID
    for pic in os.listdir(os.path.join(dir, "train_NonCOVID")):
        img = cv2.imread(os.path.join(dir, "train_NonCOVID", pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        pic_matrix = np.vstack((pic_matrix, img_1D))
    # read pic from train_COVID
    for pic in os.listdir(os.path.join(dir, "train_COVID")):
        img = cv2.imread(os.path.join(dir, "train_COVID", pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        pic_matrix = np.vstack((pic_matrix, img_1D))
    pic_matrix = np.delete(pic_matrix, 0, axis=0)
    np.savetxt("pic_matrix.csv", pic_matrix, delimiter=',')
    np.savetxt("label_matrix.csv", label_matrix, delimiter=',')
    return pic_matrix, label_matrix
