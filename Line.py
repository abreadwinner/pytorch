import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


def forward(x):
    return x * w


def loss(x, y):
    y_hat = forward(x)  # y_hat就是 y_predict
    return (y_hat - y) ** 2


if __name__ == "__main__":
    mse_list = []
    w_list = []
    for w in np.arange(0.0, 4.1, 0.1):
        print("w = ", w)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_predict = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_predict, loss_val)
        mse = l_sum / 3
        print('MSE = ', mse)
        w_list.append(w)
        mse_list.append(mse)
    plt.plot(w_list, mse_list)
    plt.ylabel('Loss')
    plt.xlabel('Mse')
    plt.show()
