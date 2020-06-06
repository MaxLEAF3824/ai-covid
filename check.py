import os
import numpy as np
import cv2
import torch
from torch import optim, nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score
import config


def testTheAccuracyOfNetwork_Lin(w, b, dir, COVID_file, NonCOVID_file):
    count = 0
    width_trans = 425
    height_trans = 302
    # 无COVID检验集数量
    NonCOVID_file_num = len(os.listdir(os.path.join(dir, NonCOVID_file)))
    # COVID检验集数量
    COVID_file_num = len(os.listdir(os.path.join(dir, COVID_file)))

    for pic in os.listdir(os.path.join(dir, NonCOVID_file)):
        img = cv2.imread(os.path.join(dir, NonCOVID_file, pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        score = img_1D.dot(w) + b
        if score[0][0] > score[0][1]:
            count += 1
            print("Correct")
        else:
            print("Wrong")
    accuracy0 = count / NonCOVID_file_num * 100
    for pic in os.listdir(os.path.join(dir, COVID_file)):
        img = cv2.imread(os.path.join(dir, COVID_file, pic))
        img_resize = cv2.resize(img, (width_trans, height_trans))
        img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_1D = np.reshape(img_gray, height_trans * width_trans)
        score = img_1D.dot(w) + b
        if score[0][0] < score[0][1]:
            count += 1
            print("Correct")
        else:
            print("Wrong")
    accuracy1 = (count - accuracy0 / 100 * NonCOVID_file_num) / COVID_file_num * 100
    print("NonCOVID准确率：%d%%" % accuracy0)
    print("COVID准确率：%d%%" % accuracy1)
    print("综合准确率：%d%%" % (count / (COVID_file_num + NonCOVID_file_num) * 100))


def testTheAccuracyOfNetwork(model, COVID_dir):
    # 数据增强
    test_directory = os.path.join(COVID_dir, "test")  # 设置测试集目录
    test_transforms = transforms.Compose(  # 设置transform
        [transforms.Resize(size=(302, 425)),  # 缩放到302*425
         transforms.CenterCrop(size=(272, 395)),  # 中心裁剪到224*224
         transforms.ToTensor(),  # 转化成张量
         transforms.Normalize([0.485, 0.456, 0.406],  # 归一化
                              [0.229, 0.224, 0.225])
         ])
    train_datasets = datasets.ImageFolder(test_directory, transform=test_transforms)  # 用ImageFolder组织测试集
    test_num = len(train_datasets)
    test_data = DataLoader(train_datasets, batch_size=test_num, shuffle=True)  # 读取数据,batch_size设置为全部

    # 验证过程
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置设备
    model.eval()  # 设置model为验证模式
    accuracy = 0  # 初始化准确率

    # 开始验证
    for i, (inputs, labels) in enumerate(test_data):
        # 调整数据类型与model一致
        inputs = inputs.to(device)
        labels = labels.to(device)
        model = model.to(device)

        # 前向计算
        outputs = model(inputs)

        # 计算准确率
        # torch.max函数返回的是每行最大数ret和他们的索引位置predictions
        ret, predictions = torch.max(outputs.data, 1)
        accuracy = precision_score(labels.cpu(), predictions.cpu())
        recall = recall_score(labels.cpu(), predictions.cpu())

    print("准确率为：%.2f" % (accuracy * 100))
    print("召回率为：%.2f" % (recall * 100))
