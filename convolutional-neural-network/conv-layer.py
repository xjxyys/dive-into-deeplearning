import torch
from torch import nn 

# 二维互相关运算
def corr2d(X, K):
    """"计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y
# 测试1: 测试二维互相关运算
def test1():
    X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    print(corr2d(X, K))

# 二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


# 测试2: 图像中的目标边缘检测
def test2():
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(X)
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print(Y)
    # 这个二维线性相关只是检测水平边缘
    # 对垂直方向就是无效的
    print(corr2d(X.t(), K))

def learn_conv():
    # 前面两个参数代表channel
    conv2d = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=(1, 2), bias=False)
    # 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度），
    # 其中批量大小和通道数都为1
    
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)

    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2
    for i in range(10):
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        l.sum().backward()
        # 迭代卷积核
        conv2d.weight.data = conv2d.weight.data - lr * conv2d.weight.grad
        if (i + 1) % 2 == 0:
            print(f'batch {i+1}, loss {l.sum():.3f}')
    print(conv2d.weight.data.reshape((1, 2)))

if __name__ == "__main__":
    test1()
    test2()
    learn_conv()
