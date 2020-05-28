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
    b = torch.randn(batch_num, class_num, device=device, dtype=dtype, requires_grad=True)

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
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w) + b

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the scalar value held in the loss.
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y_pred, y.long())
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
