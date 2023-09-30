import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, 
                 use_1x1conv=False, strides=1):
        super().__init__()
        # if the strides is not 1, conv1 will reduce the width and height and increase the number of channels
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                                 kernel_size=3, padding=1)
        if use_1x1conv:
            # if use_1x1conv is True, then conv3 will increase the number of channels
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            # if use_1x1conv is False, then conv3 will not change the number of channels
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # if conv3 is not None, then X will be transformed by conv3
        if self.conv3:
            X = self.conv3(X)
        # element-wise addition 
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # if it is not the first block and is the first residual block, the strides is 2
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # if it is the first block or is not the first residual block, the strides is 1
            blk.append(Residual(num_channels, num_channels))
    return blk

if __name__ == '__main__':
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(Y.shape)
    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    print(blk(X).shape)

    # ResNet-18
    b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(64), nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    # * is used to unpack the list
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    # the output channels of the last block is 512
    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(), nn.Linear(512, 10))
    X = torch.rand(size=(1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)
    # train the model
    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    