import os

import torch
import numpy as np
import random

from torch import optim
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from sklearn.metrics import precision_score, recall_score
import config


def train_LinearClassifier(pic_matrix, label_matrix):
    # 训练图片数量
    train_pic_num = pic_matrix.shape[0]
    # 设置训练设备
    dtype = torch.double
    device = torch.device("cpu")
    # device = torch.device("cuda:0")

    class_num, pixel_num, batch_num = config.NUM_CLASSES, 128350, config.BATCH_SIZE
    # class_num stand for 2 kind of classifications
    # pixel_num stand for 302*425 pixels in every pictures
    # batch_num stand for the number of pictures every train_round the model use

    # Create random Tensors for weights.
    w = torch.randn(pixel_num, class_num, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(1, class_num, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-5
    learning_round = 1000
    loss_rec = []
    for t in range(learning_round):
        seed = random.randint(0, train_pic_num - 1)
        if seed <= train_pic_num - batch_num:
            x = torch.from_numpy(pic_matrix[seed:seed + batch_num, :])
            y = torch.from_numpy(label_matrix[seed:seed + batch_num])
        else:
            x = torch.from_numpy(
                np.vstack((pic_matrix[seed:seed + batch_num, :], pic_matrix[0:batch_num - train_pic_num + seed, :])))
            y = torch.from_numpy(
                np.hstack((label_matrix[seed:seed + batch_num], label_matrix[0:batch_num - train_pic_num + seed])))
        b_batch = b
        for i in range(batch_num - 1):
            b_batch = torch.cat([b, b_batch], dim=0)

        y_pred = x.mm(w) + b_batch

        # calculate the loss
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y.long())
        loss_rec.append(loss.item())
        print(t, loss.item())

        # backward
        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            # Manually zero the gradients after updating weights
            w.grad.zero_()
            b.grad.zero_()
    # 训练结束后作图
    x_axis = range(1, 1001)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Loss in Training")
    ax.set_ylabel("proportion")
    ax.set_xlabel("epoch")
    ax.grid(which='minor', alpha=0.5)  # 设置网格
    ax.grid(which='major', alpha=0.5)
    ax.plot(x_axis, loss_rec, 'r', label="Train Loss")
    ax.legend()
    plt.show()
    return w.detach().numpy(), b.detach().numpy()


def train_ResNet50(train_data, valid_data):
    # 设置device为cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 若有gpu可用则用gpu

    # 使用ResNet50
    model = models.resnet50(pretrained=True)

    # 冻结参数梯度
    for param in model.parameters():
        param.requires_grad = False

    # 调整神经网络输出层
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, config.NUM_CLASSES),  # 调整为2分类
        nn.LogSoftmax(dim=1)
    )

    # 设置损失函数和优化器
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练过程
    record = []  # 保存训练过程中的数据
    best_acc = 0.0  # 保存最佳准确率
    best_epoch = 0  # 保存最好轮次数
    epochs = config.NUM_EPOCHS  # 训练轮数

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        model.train()  # 调整model为训练模式

        # 初始化记录值
        train_loss = 0.0
        train_accuracy = 0.0
        train_recall = 0.0
        valid_loss = 0.0
        valid_accuracy = 0.0
        valid_recall = 0.0

        # 开始训练
        for i, (inputs, labels) in enumerate(train_data):  # 从train_data中读取input和label

            # 调整数据类型与device一致
            inputs = inputs.to(device)
            labels = labels.to(device)
            model = model.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向计算
            outputs = model(inputs)

            # 计算损失函数
            loss = loss_func(outputs, labels)

            # 反向传播
            loss.backward()

            # 优化器调整权重
            optimizer.step()

            # 记录train_loss和train_accuracy值
            train_loss += loss.item() * inputs.size(0)
            ret, predictions = torch.max(outputs.data, 1)
            acc = precision_score(labels.cpu(), predictions.cpu())
            recall = recall_score(labels.cpu(), predictions.cpu())
            train_accuracy += acc * inputs.size(0)
            train_recall += recall * inputs.size(0)

        # 验证集
        with torch.no_grad():
            model.eval()  # 设置model为验证模式
            for j, (inputs, labels) in enumerate(valid_data):
                # 调整数据类型与device一致
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 同上
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

                # 同上
                valid_loss += loss.item() * inputs.size(0)
                ret, predictions = torch.max(outputs.data, 1)
                v_acc = precision_score(labels.cpu(), predictions.cpu())
                v_recall = recall_score(labels.cpu(), predictions.cpu())
                valid_accuracy += v_acc * inputs.size(0)
                valid_recall += v_recall * inputs.size(0)

        # 计算平均损失和准确率并保存至record
        train_data_size = len(train_data.dataset)
        valid_data_size = len(valid_data.dataset)
        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_accuracy / train_data_size
        avg_train_recall = train_recall / train_data_size
        avg_valid_loss = valid_loss / valid_data_size
        avg_valid_acc = valid_accuracy / valid_data_size
        avg_valid_recall = valid_recall / valid_data_size
        print(avg_valid_recall)
        record.append(
            [avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc, avg_train_recall, avg_valid_recall])

        # 记录最高准确性的模型
        if avg_valid_acc > best_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        epoch_end = time.time()

        # 打印信息
        print(
            "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%\n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%. Time: {:.4f}s".format(
                epoch + 1, avg_valid_loss, avg_train_acc * 100, avg_train_recall * 100, avg_valid_loss,
                avg_valid_acc * 100, avg_valid_recall * 100, epoch_end - epoch_start))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))

    # 训练结束后作图
    record_T = list(map(list, zip(*record)))
    x_axis = range(1, config.NUM_EPOCHS + 1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Loss, Accuracy and Recall in Training")
    ax.set_ylabel("proportion")
    ax.set_xlabel("epoch")
    ax.grid(which='minor', alpha=0.5)  # 设置网格
    ax.grid(which='major', alpha=0.5)
    ax.plot(x_axis, record_T[0], 'r', label="Train Loss")
    ax.plot(x_axis, record_T[2], 'g', label="Train Accuracy")
    ax.plot(x_axis, record_T[4], 'y', label="Train Recall")
    ax.legend()
    plt.show()

    return model, record
