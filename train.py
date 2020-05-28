import torch
import numpy as np
import random


def trainNetworkWithLinearClassifier(pic_matrix, label_matrix):
    train_pic_num = pic_matrix.shape[0]

    dtype = torch.double
    device = torch.device("cpu")
    # device = torch.device("cuda:0")  # Uncomment this to run on GPU

    class_num, pixel_num, batch_num = 2, 128350, 5
    # class_num stand for 2 kind of classifications
    # pixel_num stand for 302*425 pixels in every pictures
    # batch_num stand for the number of pictures every train_round the model use

    # x = torch.randn(class_num, batch_num, device=device, dtype=dtype)
    # y = torch.randn(batch_num, class_num, device=device, dtype=dtype)

    # Create random Tensors for weights.
    w = torch.randn(pixel_num, class_num, device=device, dtype=dtype, requires_grad=True)
    b = torch.randn(1, class_num, device=device, dtype=dtype, requires_grad=True)

    learning_rate = 1e-7
    learning_round = 1000
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

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y.long())
        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad
            # Manually zero the gradients after updating weights
            w.grad.zero_()
            b.grad.zero_()
    return w.detach().numpy(), b.detach().numpy()
