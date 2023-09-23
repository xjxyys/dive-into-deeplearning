import torch
from torch import nn
from d2l import torch as d2l


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i + p_h, j:j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i + p_h, j:j + p_w].mean()
    return Y

if __name__ == '__main__':
    X =  torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    print(pool2d(X, (2, 2)))
    print(pool2d(X, (2, 2), 'avg'))

    # 把通道维度放在最前面
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
    print(X)
    # 用nn.module实现最大池化，默认的步幅和池化窗口形状相同，也就是说不扫描重叠区域
    pool2d = nn.MaxPool2d(3)
    print(pool2d(X))

    # 调整填充和步幅
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))

    # 指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅
    pool2d = nn.MaxPool2d((2, 3), padding=(1, 2), stride=(2, 3))

    # 多通道
    X = torch.cat((X, X + 1), 1)
    print(X)
    pool2d = nn.MaxPool2d(3, padding=1, stride=2)
    print(pool2d(X))
