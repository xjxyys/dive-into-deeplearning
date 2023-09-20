import torch
from d2l import torch as d2l

# 多通道输入
def corr2d_multi_in(X, K):
    # First, iterate through the 0th dimension (channel dimension) of `X` and
    # `K`. Then, add them together

    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of `K`, and each time, perform
    # cross-correlation operations with input `X`. All of the results are
    # stacked together (in the 0th dimension)
    return torch.stack([corr2d_multi_in(X, k) for k in K])

# 1 × 1 卷积层其实等价于全连接层，这里用全连接来实现1 × 1卷积层
def corr2d_multi_in_out_1x1(X, K):
    # c_i 代表输入通道数，h, w 代表高和宽
    c_i, h, w = X.shape
    # c_o 代表输出通道数
    c_o = K.shape[0]
    # 把输入X给展平
    X = X.reshape((c_i, h * w))
    # 把核K给展平
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully-connected layer
    Y = torch.matmul(K, X)  # Y: (c_o, h * w)
    return Y.reshape((c_o, h, w))


# 测试1: 测试多输入通道
def test1(X, K):
    print(corr2d_multi_in(X, K))
# 测试2: 测试多输入多输出通道
def test2(X, K):
    print(corr2d_multi_in_out(X, K))

if __name__ == "__main__":
    # X 的维度是 input_channels × height × width，K 的维度是 (output_channels ×) input_channels × height × width
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    test1(X, K)

    K = torch.stack((K, K + 1, K + 2), 0)
    print(K.shape)
    test2(X, K)

    # 1 × 1 卷积层
    X = torch.normal(0, 1, (3, 3, 3))
    K = torch.normal(0, 1, (2, 3, 1, 1))
    Y1 = corr2d_multi_in_out_1x1(X, K)
    Y2 = corr2d_multi_in_out(X, K)
    assert float(torch.abs(Y1 - Y2).sum()) < 1e-6