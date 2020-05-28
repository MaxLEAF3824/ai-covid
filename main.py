# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import cv2
import os
from Preparation import importAndPrepocessPictures

train_pic_num = 696
COVID_dir = "D:\\Coding\\Others\\COVID"
# ----stage 1: Preparation----
pic_matrix, label_matrix = importAndPrepocessPictures(COVID_dir)
# read pic from test_COVID
for pic in os.listdir(os.path.join(COVID_dir, "test_COVID")):
    img = cv2.imread(os.path.join(COVID_dir, "test_COVID", pic))
    cv2.imshow("1", img)
# ----stage 2: Prepossessing----

pic_matrix = np.ndarray([128350, train_pic_num])
label_matrix = np.ndarray([1, train_pic_num])
# ----stage 3: Study using Linear classifier----
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

D_in, H, D_out = 2, 128350, 100
# D_in stand for 2 kind of classifications
# H stand for 302*425 pixels in every pictures
# D_out stand for the number of pictures every round that the model use


x = torch.randn(D_in, D_out, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
b = torch.randn(D_in, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
learning_round = 500
for t in range(learning_round):
    seed = random.randint(0, train_pic_num - 1)
    if seed <= train_pic_num - D_out:
        x = pic_matrix[:, seed:seed + D_out]
        y = label_matrix[:, seed:seed + D_out]
    else:
        x = np.concatenate((pic_matrix[:, seed:seed + D_out], pic_matrix[:, 0:D_out - (train_pic_num - seed)]), axis=1)
        y = np.concatenate((label_matrix[:, seed:seed + D_out], label_matrix[:, 0:D_out - (train_pic_num - seed)]),
                           axis=1)
    # Forward pass: compute predicted y using operations on Tensors; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = w.mm(x) + b

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

        # Manually zero the gradients after updating weights
        w.grad.zero_()
        b.grad.zero_()
