import torch
from torch import nn
from d2l import torch as d2l

class Reshape(nn.Module):
    """将输入数据形状转展平。"""
    def forward(self, x):
        # view和reshape的区别在于view会共享内存，reshape不会，reshape会复制一份
        return x.view(-1, 1, 28, 28)
    
net = nn.Sequential(
    Reshape(),
    # 输入数据大小为28*28
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # in_channels, out_channels, kernel_size
    # nn.Sigmoid(),
    nn.ReLU(),
    # 数据大小为28*28
    nn.AvgPool2d(kernel_size=2, stride=2),  # kernel_size, stride
    # 数据大小为14*14
    nn.Conv2d(6, 16, kernel_size=5), # 输出通道增大
    # nn.Sigmoid(),
    nn.ReLU(),

    # 数据大小为10*10
    nn.AvgPool2d(kernel_size=2, stride=2),
    # 数据大小为5*5
    nn.Flatten(), # 将四维的输出转成二维的输出，第一维是batch_size,faltten()不会影响第一维也即batch_size
    # 全连接层
    nn.Linear(16 * 5 * 5, 120), # in_features, out_features
    # nn.Sigmoid(),
    nn.ReLU(),

    nn.Linear(120, 64),
    # nn.Sigmoid(),
    nn.ReLU(),

    nn.Linear(64, 10)
)

# 在GPU上的评估函数
def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Sequential):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device # 如果没指定device就使用net的device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后会介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel()) # y.numel()返回y中元素的个数
    return metric[0] / metric[1]

# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight) # 初始化权重, xavier_uniform_是均匀分布防止梯度消失或者梯度爆炸

    net.apply(init_weights) # 初始化权重
    print('training on', device)
    net.to(device) # 模型放到device上
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 优化器
    loss = nn.CrossEntropyLoss() # 损失函数, 交叉熵损失函数(多分类问题)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc']) # 动画绘制器  
    timer, num_batches = d2l.Timer(), len(train_iter) # 计时器
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = d2l.Accumulator(3)
        net.train() # 设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad() # 梯度清零
            X, y = X.to(device), y.to(device) # 数据放到device上
            y_hat = net(X) # 前向传播
            l = loss(y_hat, y) # 计算损失
            l.backward() # 反向传播
            optimizer.step() # 更新参数
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2] # 训练损失
            train_acc = metric[1] / metric[2] # 训练准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter) # 测试准确率
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, 'f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec 'f'on {str(device)}')

if __name__ == "__main__":
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
    
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    lr, num_epochs = 0.1, 10
    train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
