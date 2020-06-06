# -*- coding: utf-8 -*-
from time import time

import preparation as stage1
import train as stage2
import check as stage3
import config
import torch
import os

COVID_dir = config.COVID_dir

start_time = time()

# ----stage 1: Preparation and Preprocess ----

# pic_matrix, label_matrix = stage1.importAndPreprocessPictures_manually(COVID_dir)
train_data, valid_data = stage1.importAndPrepocessPictures(COVID_dir)

stage1_end_time = time()
print("阶段1用时：%.4f秒" % (stage1_end_time - start_time))

# ----stage 2: Study with Linear classifier----

# w, b = stage2.train_LinearClassifier(pic_matrix, label_matrix)
model, record = stage2.train_ResNet50(train_data, valid_data)
# torch.save(model, os.path.join(config.COVID_dir, "trained_model.pth"))

stage2_end_time = time()
print("阶段2用时：%.4f秒" % (stage2_end_time - stage1_end_time))

# ----stage 3: Check the accuracy----

# model = torch.load(os.path.join(config.COVID_dir, "trained_model.pth"))
stage3.testTheAccuracyOfNetwork(model, COVID_dir)

stage3_end_time = time()
print("阶段3用时：%.4f秒" % (stage3_end_time - stage2_end_time))
