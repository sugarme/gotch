# Linear Regression, NN, and CNN on MNIST dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sugarme/nb/blob/master/mnist/mnist.ipynb)

## MNIST

- MNIST files can be obtained from [this source](http://yann.lecun.com/exdb/mnist/) and put in `data/mnist` from
    root folder of this project.

- Load MNIST data using helper function at `vision` sub-package


## Linear Regression

- Run with `go clean -cache -testcache && go run . -model="linear"`


- Accuracy should be about **91.68%**.


## Neural Network (NN)

- Run with `go clean -cache -testcache && go run . -model="nn"`

- Accuracy should be about **94%**.


## Convolutional Neural Network (CNN)

- Run with `go clean -cache -testcache && go run . -model="cnn"`

- Accuracy should be about **99.3%**.

## Benchmark against Python

- Train batch size: 256
- Test batch size: 1000
- Adam optimizer, learning rate = 3*1e-4
- Epochs: 30

```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import time

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(1024, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(
                dim=1,
                keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size',
                        type=int,
                        default=256,
                        metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs',
                        type=int,
                        default=14,
                        metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model',
                        action='store_true',
                        default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        #  transforms.Normalize((0.1307, ), (0.3081, )),
    ])
    dataset1 = datasets.MNIST('../data',
                              train=True,
                              download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    end = time.time()

    print("taken time: {:.2f}mins".format((end - start) / 60.0))

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
```

```bash
Test set: Average loss: 0.1101, Accuracy: 9666/10000 (96.66%)
Test set: Average loss: 0.0697, Accuracy: 9779/10000 (97.79%)
Test set: Average loss: 0.0442, Accuracy: 9856/10000 (98.56%)
Test set: Average loss: 0.0384, Accuracy: 9873/10000 (98.73%)
Test set: Average loss: 0.0358, Accuracy: 9875/10000 (98.75%)
Test set: Average loss: 0.0323, Accuracy: 9898/10000 (98.98%)
Test set: Average loss: 0.0290, Accuracy: 9906/10000 (99.06%)
Test set: Average loss: 0.0272, Accuracy: 9910/10000 (99.10%)
Test set: Average loss: 0.0280, Accuracy: 9913/10000 (99.13%)
Test set: Average loss: 0.0295, Accuracy: 9908/10000 (99.08%)
Test set: Average loss: 0.0251, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0246, Accuracy: 9924/10000 (99.24%)
Test set: Average loss: 0.0258, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0296, Accuracy: 9911/10000 (99.11%)
Test set: Average loss: 0.0271, Accuracy: 9912/10000 (99.12%)
Test set: Average loss: 0.0251, Accuracy: 9918/10000 (99.18%)
Test set: Average loss: 0.0276, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0291, Accuracy: 9912/10000 (99.12%)
Test set: Average loss: 0.0291, Accuracy: 9920/10000 (99.20%)
Test set: Average loss: 0.0333, Accuracy: 9904/10000 (99.04%)
Test set: Average loss: 0.0268, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0265, Accuracy: 9931/10000 (99.31%)
Test set: Average loss: 0.0316, Accuracy: 9918/10000 (99.18%)
Test set: Average loss: 0.0299, Accuracy: 9917/10000 (99.17%)
Test set: Average loss: 0.0303, Accuracy: 9923/10000 (99.23%)
Test set: Average loss: 0.0327, Accuracy: 9914/10000 (99.14%)
Test set: Average loss: 0.0314, Accuracy: 9918/10000 (99.18%)
Test set: Average loss: 0.0316, Accuracy: 9920/10000 (99.20%)
Test set: Average loss: 0.0346, Accuracy: 9916/10000 (99.16%)
Test set: Average loss: 0.0308, Accuracy: 9923/10000 (99.23%)
taken time: 5.63mins
```

Gotch CNN performance

```bash
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.16      Test accuracy: 96.53%
Epoch: 1         Loss: 0.08      Test accuracy: 97.27%
Epoch: 2         Loss: 0.14      Test accuracy: 97.28%
Epoch: 3         Loss: 0.08      Test accuracy: 97.64%
Epoch: 4         Loss: 0.07      Test accuracy: 98.44%
Epoch: 5         Loss: 0.05      Test accuracy: 98.59%
Epoch: 6         Loss: 0.06      Test accuracy: 98.67%
Epoch: 7         Loss: 0.07      Test accuracy: 98.80%
Epoch: 8         Loss: 0.11      Test accuracy: 98.01%
Epoch: 9         Loss: 0.07      Test accuracy: 98.81%
Epoch: 10        Loss: 0.05      Test accuracy: 98.76%
Epoch: 11        Loss: 0.04      Test accuracy: 98.78%
Epoch: 12        Loss: 0.02      Test accuracy: 98.81%
Epoch: 13        Loss: 0.05      Test accuracy: 98.78%
Epoch: 14        Loss: 0.05      Test accuracy: 98.74%
Epoch: 15        Loss: 0.06      Test accuracy: 98.86%
Epoch: 16        Loss: 0.07      Test accuracy: 98.95%
Epoch: 17        Loss: 0.03      Test accuracy: 98.93%
Epoch: 18        Loss: 0.04      Test accuracy: 98.99%
Epoch: 19        Loss: 0.05      Test accuracy: 99.05%
Epoch: 20        Loss: 0.06      Test accuracy: 99.11%
Epoch: 21        Loss: 0.03      Test accuracy: 98.78%
Epoch: 22        Loss: 0.05      Test accuracy: 98.88%
Epoch: 23        Loss: 0.02      Test accuracy: 99.04%
Epoch: 24        Loss: 0.04      Test accuracy: 99.08%
Epoch: 25        Loss: 0.03      Test accuracy: 98.96%
Epoch: 26        Loss: 0.07      Test accuracy: 98.78%
Epoch: 27        Loss: 0.05      Test accuracy: 98.81%
Epoch: 28        Loss: 0.03      Test accuracy: 98.79%
Epoch: 29        Loss: 0.07      Test accuracy: 98.82%
Best test accuracy: 99.11%
Taken time:     2.81 mins
```

