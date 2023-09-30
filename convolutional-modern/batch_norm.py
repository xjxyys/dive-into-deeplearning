import torch
from torch import nn 
from d2l import torch as d2l

# the effect of batch normalization is making the distribution of each layer's inputs more stable and reducing the internal covariate shift, 
# It can also make the convergence faster and 

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """Batch normalization
    moving_mean and moving_var are for the global mean and variance
    """
    # Use torch.is_grad_enabled() to determine whether the current mode is training mode or prediction mode
    if not torch.is_grad_enabled():
        # if it is prediction mode, directly use the mean and variance obtained from the training set
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        # When using a fully connected layer, calculate the mean and variance on the feature dimension
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0) # (m, n) -> (1, n)
            
        # When using a two-dimensional convolutional layer, calculate the mean and variance on the channel dimension (axis=1)
        else:
            # Here we need to keep the shape of X, so that the broadcast operation can be carried out later
            # we get the mean and variance on the channel dimension and keep the shape of X
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True) # (m, c, h, w) -> (1, c, 1, 1)
        # In training mode, the current mean and variance are used for the standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance of the moving average, update the moving average using the momentum method
        # The momentum term is used to reduce the variance caused by the mini-batch approximation to the true mean and variance
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # Scale and shift
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer and the number of output channels for a convolutional layer
    # num_dims: 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            # Fully connected layer
            shape = (1, num_features)
        else:
            # Convolutional layer
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter are initialized to 0 and 1 respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # All the variables not involved in gradient calculation are initialized to 0
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)
    
    def forward(self, X):
        # if X is not on the main memory, copy moving_mean and moving_var to the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y

# Lenet with batch normalization
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5),
                    BatchNorm(6, num_dims=4),
                    nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5),
                    BatchNorm(16, num_dims=4),
                    nn.Sigmoid(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16*4*4, 120),
                    BatchNorm(120, num_dims=2),
                    nn.Sigmoid(),
                    nn.Linear(120, 84),
                    BatchNorm(84, num_dims=2),
                    nn.Sigmoid(),
                    nn.Linear(84, 10))
if __name__ == "__main__":
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    print(net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,)))

    # a concise implementation
    net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5),
                        nn.BatchNorm2d(6),
                        nn.Sigmoid(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Conv2d(6, 16, kernel_size=5),
                        nn.BatchNorm2d(16),
                        nn.Sigmoid(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.Flatten(),
                        nn.Linear(16*4*4, 120),
                        nn.BatchNorm1d(120),
                        nn.Sigmoid(),
                        nn.Linear(120, 84),
                        nn.BatchNorm1d(84),
                        nn.Sigmoid(),
                        nn.Linear(84, 10))
    
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    