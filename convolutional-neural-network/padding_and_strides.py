import torch
from torch import nn


# 此函数初始化卷积层权重，并输入和输出提高和缩减的维数
def comp_conv2d(conv2d, X):
    # 这里的(1, 1)表示批量大小和通道数都为1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # omit the first two arguments
    return Y.reshape(Y.shape[2:])

# 测试1: 测试卷积层
def test1():
    # padding = 一个数时表示在两边都填充，两个数时表示在上下左右分别填充
    # kernel_size = 3 其实表示3 × 3的卷积核
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    X = torch.rand(size=(8, 8))
    print(comp_conv2d(conv2d, X).shape)

    # 一般选择kernel_size 为奇数，padding = h - 1
    conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding = (2, 1))
    print(comp_conv2d(conv2d, X).shape)

# 测试2: 测试步长
def test2():
    # torch.rand表示0~1的均匀分布，torch.randn表示N(0, 1)的标准正态分布
    X = torch.rand(size=(8, 8))
    conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride= 2)
    # 输出的维度为: (input_length  + padding - kernel_length) //stride + 1
    # 也即 (input_length + padding - kernel_length + stride) // stride
    print(comp_conv2d(conv2d, X).shape)

    # 一个稍微复杂的例子
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    print(comp_conv2d(conv2d, X).shape)

if __name__ == '__main__':
    test1()
    test2()