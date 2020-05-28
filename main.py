# -*- coding: utf-8 -*-
from time import time

from check import testTheAccuracyOfNetwork
from preparation import importAndPrepocessPictures
from train import trainNetworkWithLinearClassifier

train_pic_num = 696
COVID_dir = "D:\\Coding\\Others\\COVID"
start_time = time()

# ----stage 1: Preparation and Prepossess----
pic_matrix, label_matrix = importAndPrepocessPictures(COVID_dir)
stage1_end_time = time()
print("阶段1用时：%.2f秒" % (stage1_end_time - start_time))

# ----stage 2: Study with Linear classifier----
# w, b = trainNetworkWithLinearClassifier(pic_matrix, label_matrix)

stage2_end_time = time()
print("阶段2用时：%.2f秒" % (stage2_end_time - stage1_end_time))

# ----stage 3: Check the accuracy----
# testTheAccuracyOfNetwork(w, b, COVID_dir)

stage3_end_time = time()
print("阶段3用时：%.2f秒" % (stage3_end_time - stage2_end_time))
