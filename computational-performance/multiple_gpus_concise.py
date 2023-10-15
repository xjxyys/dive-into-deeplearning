import torch
from torch import nn
from d2l import torch as d2l

def resnet18(num_classes, in_channels=1):
    """slightly modify the original resnet18"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    d2l.Residual(in_channels, out_channels, use_1x1conv=True,
                                 strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)
    # This model uses a smaller convolution kernel, stride, and padding and
    # removes the maximum pooling layer
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", d2l.GlobalAvgPool2d(1,1))
    net.add_module("fc",
                   nn.Sequential(nn.Flatten(),
                                 nn.Linear(512, num_classes)))
    return net

def train(net, num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            # nn.init.normal_(m.weight, std=0.01)
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # split the model to multiple GPUs
    net = nn.DataParallel(net, device_ids=devices)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction="none")
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, 10], legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(10):
        # switch to training mode, this is unnecessary since we don't have Dropout in this example
        net.train()
        timer.start()
        for i, (X, y) in enumerate(train_iter):
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y).sum()
            l.backward()
            trainer.step()
        timer.stop()
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))
    print(f'test acc: {animator.Y[0][-1]:.2f},
           {timer.avg():.1f} sec/epoch on {str(devices)}')
if __name__ == "__main__":
    net = resnet18(10)
    # get the gpu list
    devices = d2l.try_all_gpus()    
    # split the model to multiple GPUs
    # train(net, num_gpus=len(devices), batch_size=256, lr=0.1)

    train(net, num_gpus=2, batch_size=256, lr=0.1*2)


