import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    # 输入图片大小为 224x224
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    # 图片大小变为 54x54
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 图片大小变为 26x26
    # 减少卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    # 图片大小变为 26x26
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 图片大小变为 12x12
    # 使用三个连续的卷积层和较小的卷积窗口
    # 除了最后的卷积层外，进一步增大了输出通道数。
    # 前两个卷积层后不使用池化层来减小输入的高和宽
    # 而第三个卷积层后使用池化层来减半输入的高和宽
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 图片大小变为 5x5
    # 这里全连接层的输出个数比 LeNet 中的大数倍。使用丢弃层来缓解过拟合
    nn.Flatten(),
    nn.Linear(256 * 5 * 5, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # 最后是输出层。由于这里使用 Fashion-MNIST，所以用类别数为 10，而非论文中的 1000
    nn.Linear(4096, 10))

if __name__ == '__main__':
    X = torch.randn(size=(1, 1, 224, 224), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
    batch_size = 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    lr, num_epochs = 0.01, 10
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())