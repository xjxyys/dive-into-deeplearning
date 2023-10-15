import torch
from torch import nn 
from torch.nn import functional as F
from d2l import torch as d2l


# initialize the parameter
scale = 0.01
# size: (out_channels, in_channels, kernel_size)
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    # fully connected layer
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat
loss = nn.CrossEntropyLoss(reduction='none')

def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_()
    return new_params
# test1: test the get_params function
def test1():
    new_params = get_params(params, d2l.try_gpu(0))
    print('b1 weight:', new_params[1])
    print('b1 grad:', new_params[1].grad)

def allreduce(data):
    # sum all the data
    for i in range(1, len(data)):
        data[0][:] += data[i].to(data[0].device)
    # broadcast the result
    for i in range(1, len(data)):
        data[i] = data[0].to(data[i].device)

# test2: test the allreduce function
def test2():
    # tensor([[1., 1.]], device='cuda:0')
    data = [torch.ones((1, 2), device=d2l.try_gpu(i)) * (i + 1) for i in range(2)]
    print('before allreduce:', data)
    allreduce(data)
    print('after allreduce:', data)

# test3: test the allocate function
def test3():
    data = torch.arange(20).reshape(4, 5)
    devices = [torch.device('cuda:0'), torch.device('cuda:1')]
    split = nn.parallel.scatter(data, devices)
    print('input:', data)
    print('load into', devices)
    print('output:', split)

def split_batch(X, y, devices):
    """Split X and y into multiple devices."""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
def train_batch(X, y, device_params, devices, lr):
    X_shards, y_shards = split_batch(X, y, devices)
    # run the forward and backward pass on each device
    # loss is a scalar
    ls = [loss(lenet(X_shards, device_W), y_shards).sum() 
          for X_shards, device_W in zip(
              X_shards, device_params)]
    for l in ls:
        l.backward()
    # aggregate the gradients on the GPU
    with torch.no_grad():
        # device_params[0][0] is the first layer's weight on the first device
        for i in range(len(device_params[0])):
            allreduce([device_params[c][i].grad for c in range(len(devices))])
    # update the parameters
    for param in device_params:
        d2l.sgd(param, lr, X.shape[0]) # sum over all devices
    
def train(num_gpus, batch_size, lr):
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # copy the parameters to num_gpus GPUs
    device_params = [get_params(params, d) for d in devices]
    num_epochs = 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        timer.start()
        for X, y in train_iter:
            train_batch(X, y, device_params, devices, lr)
            # synchronize all the devices before testing
            torch.cuda.synchronize()
        timer.stop()
        # aggregate the test results across all devices
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
            f'on {str(devices)}')

def test4():
    train(num_gpus=1, batch_size=256, lr=0.2)
    train(num_gpus=2, batch_size=256, lr=0.2)

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    test4()