import os
import cv2
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
import config


# 线性分类器的旧的导入设置，没有什么用了
def importAndPreprocessPictures_manually(dir, COVID_file="train\\train_COVID", NonCOVID_file="train\\train_NonCOVID"):
    width_trans = 425
    height_trans = 302
    pic_matrix = np.zeros((width_trans * height_trans,), dtype=np.float)
    NonCOVID_file_num = len(os.listdir(os.path.join(dir, NonCOVID_file)))
    COVID_file_num = len(os.listdir(os.path.join(dir, COVID_file)))
    label_matrix = np.hstack((np.zeros(NonCOVID_file_num), np.ones(COVID_file_num)))
    # read pic from train_NonCOVID
    for pic in os.listdir(os.path.join(dir, NonCOVID_file)):
        img = cv2.imread(os.path.join(dir, NonCOVID_file, pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        pic_matrix = np.vstack((pic_matrix, img_1D))
        cv2.imshow("123", img_gray)
        cv2.waitKey(0)
    # read pic from train_COVID
    for pic in os.listdir(os.path.join(dir, COVID_file)):
        img = cv2.imread(os.path.join(dir, COVID_file, pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        pic_matrix = np.vstack((pic_matrix, img_1D))
    pic_matrix = np.delete(pic_matrix, 0, axis=0)
    # np.savetxt("pic_matrix.csv", pic_matrix, delimiter=',')
    # np.savetxt("label_matrix.csv", label_matrix, delimiter=',')
    return pic_matrix, label_matrix


def readGeneratedData():
    # 从csv导入图片数据
    pic_matrix = np.loadtxt("pic_matrix.csv", delimiter=',')
    label_matrix = np.loadtxt("label_matrix.csv", delimiter=',')
    return pic_matrix, label_matrix

# ResNet50的图片导入
def importAndPrepocessPictures(dir):
    # 预处理
    batch_size = config.BATCH_SIZE
    train_directory = os.path.join(dir, "train")
    valid_directory = os.path.join(dir, "valid")
    train_transforms = transforms.Compose(
        [transforms.Resize(size=(302, 425)),  # 变形到302*425
         transforms.CenterCrop(size=(272, 395)),  # 中心裁剪到224*224
         transforms.ToTensor(),  # 转化成张量
         transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                              [0.229, 0.224, 0.225])
         ])

    # 读取train_data和valid_data
    train_datasets = datasets.ImageFolder(train_directory, transform=train_transforms)
    train_data_size = len(train_datasets)
    train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_datasets = datasets.ImageFolder(valid_directory, transform=train_transforms)
    valid_data_size = len(valid_datasets)
    valid_data = DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)

    # 看一下处理后的样子
    # print(train_data_size, valid_data_size)
    # for images, labels in train_data:
    #     print(labels)
    #     img = images[0]
    #     img = img.numpy()
    #     img = np.transpose(img, (1, 2, 0))
    #     cv2.imshow("123",img)
    #     cv2.waitKey(0)

    return train_data, valid_data, train_data_size, valid_data_size
