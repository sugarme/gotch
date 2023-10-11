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


## New Implement with Go `Garbage Collection`

### Linear

```bash
go run -race . -model=linear -device=cuda
Epoch: 0 - Loss: 2.303 - Test accuracy: 68.08%
Epoch: 1 - Loss: 1.508 - Test accuracy: 60.77%
Epoch: 2 - Loss: 1.388 - Test accuracy: 52.54%
Epoch: 3 - Loss: 1.579 - Test accuracy: 64.46%
Epoch: 4 - Loss: 1.707 - Test accuracy: 60.47%
Epoch: 5 - Loss: 1.194 - Test accuracy: 61.55%
Epoch: 6 - Loss: 1.395 - Test accuracy: 70.66%
Epoch: 7 - Loss: 1.290 - Test accuracy: 70.43%
Epoch: 8 - Loss: 0.892 - Test accuracy: 66.76%
Epoch: 9 - Loss: 0.935 - Test accuracy: 71.84%
Epoch: 10 - Loss: 1.108 - Test accuracy: 73.77%
Epoch: 11 - Loss: 0.906 - Test accuracy: 78.11%
Epoch: 12 - Loss: 0.786 - Test accuracy: 79.07%
Epoch: 13 - Loss: 0.686 - Test accuracy: 80.85%
Epoch: 14 - Loss: 0.666 - Test accuracy: 79.68%
Epoch: 15 - Loss: 0.685 - Test accuracy: 82.69%
Epoch: 16 - Loss: 0.623 - Test accuracy: 82.13%
Epoch: 17 - Loss: 0.592 - Test accuracy: 82.36%
Epoch: 18 - Loss: 0.604 - Test accuracy: 80.09%
Epoch: 19 - Loss: 0.614 - Test accuracy: 80.69%
Epoch: 20 - Loss: 0.613 - Test accuracy: 78.71%
Epoch: 21 - Loss: 0.645 - Test accuracy: 81.83%
Epoch: 22 - Loss: 0.582 - Test accuracy: 81.11%
Epoch: 23 - Loss: 0.591 - Test accuracy: 83.56%
Epoch: 24 - Loss: 0.542 - Test accuracy: 83.58%
Epoch: 25 - Loss: 0.539 - Test accuracy: 85.27%
Epoch: 26 - Loss: 0.507 - Test accuracy: 85.70%
Epoch: 27 - Loss: 0.498 - Test accuracy: 86.79%
Epoch: 28 - Loss: 0.477 - Test accuracy: 87.10%
Epoch: 29 - Loss: 0.467 - Test accuracy: 87.83%
Epoch: 30 - Loss: 0.453 - Test accuracy: 88.09%
Epoch: 31 - Loss: 0.445 - Test accuracy: 88.44%
Epoch: 32 - Loss: 0.436 - Test accuracy: 88.52%
Epoch: 33 - Loss: 0.429 - Test accuracy: 88.78%
Epoch: 34 - Loss: 0.424 - Test accuracy: 88.78%
Epoch: 35 - Loss: 0.419 - Test accuracy: 89.20%
Epoch: 36 - Loss: 0.415 - Test accuracy: 89.10%
Epoch: 37 - Loss: 0.412 - Test accuracy: 89.49%
Epoch: 38 - Loss: 0.409 - Test accuracy: 89.37%
Epoch: 39 - Loss: 0.406 - Test accuracy: 89.62%
Epoch: 40 - Loss: 0.404 - Test accuracy: 89.58%
Epoch: 41 - Loss: 0.402 - Test accuracy: 89.71%
Epoch: 42 - Loss: 0.399 - Test accuracy: 89.68%
Epoch: 43 - Loss: 0.398 - Test accuracy: 89.85%
Epoch: 44 - Loss: 0.396 - Test accuracy: 89.77%
Epoch: 45 - Loss: 0.394 - Test accuracy: 89.87%
Epoch: 46 - Loss: 0.392 - Test accuracy: 89.89%
Epoch: 47 - Loss: 0.391 - Test accuracy: 89.95%
Epoch: 48 - Loss: 0.389 - Test accuracy: 89.96%
Epoch: 49 - Loss: 0.388 - Test accuracy: 90.05%
Epoch: 50 - Loss: 0.387 - Test accuracy: 90.04%
Epoch: 51 - Loss: 0.385 - Test accuracy: 90.13%
Epoch: 52 - Loss: 0.384 - Test accuracy: 90.11%
Epoch: 53 - Loss: 0.383 - Test accuracy: 90.24%
Epoch: 54 - Loss: 0.381 - Test accuracy: 90.19%
Epoch: 55 - Loss: 0.380 - Test accuracy: 90.27%
Epoch: 56 - Loss: 0.379 - Test accuracy: 90.22%
Epoch: 57 - Loss: 0.378 - Test accuracy: 90.30%
Epoch: 58 - Loss: 0.377 - Test accuracy: 90.28%
Epoch: 59 - Loss: 0.376 - Test accuracy: 90.33%
Epoch: 60 - Loss: 0.375 - Test accuracy: 90.33%
Epoch: 61 - Loss: 0.374 - Test accuracy: 90.37%
Epoch: 62 - Loss: 0.372 - Test accuracy: 90.38%
Epoch: 63 - Loss: 0.371 - Test accuracy: 90.41%
Epoch: 64 - Loss: 0.371 - Test accuracy: 90.42%
Epoch: 65 - Loss: 0.370 - Test accuracy: 90.43%
Epoch: 66 - Loss: 0.369 - Test accuracy: 90.45%
Epoch: 67 - Loss: 0.368 - Test accuracy: 90.51%
Epoch: 68 - Loss: 0.367 - Test accuracy: 90.52%
Epoch: 69 - Loss: 0.366 - Test accuracy: 90.53%
Epoch: 70 - Loss: 0.365 - Test accuracy: 90.53%
Epoch: 71 - Loss: 0.364 - Test accuracy: 90.55%
Epoch: 72 - Loss: 0.363 - Test accuracy: 90.57%
Epoch: 73 - Loss: 0.363 - Test accuracy: 90.57%
Epoch: 74 - Loss: 0.362 - Test accuracy: 90.58%
Epoch: 75 - Loss: 0.361 - Test accuracy: 90.59%
Epoch: 76 - Loss: 0.360 - Test accuracy: 90.59%
Epoch: 77 - Loss: 0.359 - Test accuracy: 90.62%
Epoch: 78 - Loss: 0.359 - Test accuracy: 90.67%
Epoch: 79 - Loss: 0.358 - Test accuracy: 90.67%
Epoch: 80 - Loss: 0.357 - Test accuracy: 90.70%
Epoch: 81 - Loss: 0.357 - Test accuracy: 90.71%
Epoch: 82 - Loss: 0.356 - Test accuracy: 90.72%
Epoch: 83 - Loss: 0.355 - Test accuracy: 90.77%
Epoch: 84 - Loss: 0.355 - Test accuracy: 90.77%
Epoch: 85 - Loss: 0.354 - Test accuracy: 90.78%
Epoch: 86 - Loss: 0.353 - Test accuracy: 90.80%
Epoch: 87 - Loss: 0.353 - Test accuracy: 90.82%
Epoch: 88 - Loss: 0.352 - Test accuracy: 90.83%
Epoch: 89 - Loss: 0.351 - Test accuracy: 90.82%
Epoch: 90 - Loss: 0.351 - Test accuracy: 90.84%
Epoch: 91 - Loss: 0.350 - Test accuracy: 90.87%
Epoch: 92 - Loss: 0.350 - Test accuracy: 90.88%
Epoch: 93 - Loss: 0.349 - Test accuracy: 90.87%
Epoch: 94 - Loss: 0.348 - Test accuracy: 90.89%
Epoch: 95 - Loss: 0.348 - Test accuracy: 90.89%
Epoch: 96 - Loss: 0.347 - Test accuracy: 90.91%
Epoch: 97 - Loss: 0.347 - Test accuracy: 90.94%
Epoch: 98 - Loss: 0.346 - Test accuracy: 90.94%
Epoch: 99 - Loss: 0.346 - Test accuracy: 90.96%
Epoch: 100 - Loss: 0.345 - Test accuracy: 90.96%
Epoch: 101 - Loss: 0.345 - Test accuracy: 90.96%
Epoch: 102 - Loss: 0.344 - Test accuracy: 91.02%
Epoch: 103 - Loss: 0.344 - Test accuracy: 91.02%
Epoch: 104 - Loss: 0.343 - Test accuracy: 91.04%
Epoch: 105 - Loss: 0.343 - Test accuracy: 91.05%
Epoch: 106 - Loss: 0.342 - Test accuracy: 91.05%
Epoch: 107 - Loss: 0.342 - Test accuracy: 91.06%
Epoch: 108 - Loss: 0.341 - Test accuracy: 91.07%
Epoch: 109 - Loss: 0.341 - Test accuracy: 91.08%
Epoch: 110 - Loss: 0.340 - Test accuracy: 91.09%
Epoch: 111 - Loss: 0.340 - Test accuracy: 91.11%
Epoch: 112 - Loss: 0.339 - Test accuracy: 91.14%
Epoch: 113 - Loss: 0.339 - Test accuracy: 91.15%
Epoch: 114 - Loss: 0.339 - Test accuracy: 91.14%
Epoch: 115 - Loss: 0.338 - Test accuracy: 91.15%
Epoch: 116 - Loss: 0.338 - Test accuracy: 91.17%
Epoch: 117 - Loss: 0.337 - Test accuracy: 91.18%
Epoch: 118 - Loss: 0.337 - Test accuracy: 91.19%
Epoch: 119 - Loss: 0.336 - Test accuracy: 91.20%
Epoch: 120 - Loss: 0.336 - Test accuracy: 91.22%
Epoch: 121 - Loss: 0.336 - Test accuracy: 91.23%
Epoch: 122 - Loss: 0.335 - Test accuracy: 91.28%
Epoch: 123 - Loss: 0.335 - Test accuracy: 91.30%
Epoch: 124 - Loss: 0.334 - Test accuracy: 91.29%
Epoch: 125 - Loss: 0.334 - Test accuracy: 91.32%
Epoch: 126 - Loss: 0.334 - Test accuracy: 91.35%
Epoch: 127 - Loss: 0.333 - Test accuracy: 91.37%
Epoch: 128 - Loss: 0.333 - Test accuracy: 91.36%
Epoch: 129 - Loss: 0.333 - Test accuracy: 91.36%
Epoch: 130 - Loss: 0.332 - Test accuracy: 91.36%
Epoch: 131 - Loss: 0.332 - Test accuracy: 91.35%
Epoch: 132 - Loss: 0.332 - Test accuracy: 91.36%
Epoch: 133 - Loss: 0.331 - Test accuracy: 91.37%
Epoch: 134 - Loss: 0.331 - Test accuracy: 91.38%
Epoch: 135 - Loss: 0.330 - Test accuracy: 91.39%
Epoch: 136 - Loss: 0.330 - Test accuracy: 91.40%
Epoch: 137 - Loss: 0.330 - Test accuracy: 91.43%
Epoch: 138 - Loss: 0.329 - Test accuracy: 91.44%
Epoch: 139 - Loss: 0.329 - Test accuracy: 91.45%
Epoch: 140 - Loss: 0.329 - Test accuracy: 91.46%
Epoch: 141 - Loss: 0.328 - Test accuracy: 91.47%
Epoch: 142 - Loss: 0.328 - Test accuracy: 91.46%
Epoch: 143 - Loss: 0.328 - Test accuracy: 91.48%
Epoch: 144 - Loss: 0.328 - Test accuracy: 91.46%
Epoch: 145 - Loss: 0.327 - Test accuracy: 91.46%
Epoch: 146 - Loss: 0.327 - Test accuracy: 91.46%
Epoch: 147 - Loss: 0.327 - Test accuracy: 91.47%
Epoch: 148 - Loss: 0.326 - Test accuracy: 91.47%
Epoch: 149 - Loss: 0.326 - Test accuracy: 91.48%
Epoch: 150 - Loss: 0.326 - Test accuracy: 91.48%
Epoch: 151 - Loss: 0.325 - Test accuracy: 91.50%
Epoch: 152 - Loss: 0.325 - Test accuracy: 91.50%
Epoch: 153 - Loss: 0.325 - Test accuracy: 91.52%
Epoch: 154 - Loss: 0.325 - Test accuracy: 91.52%
Epoch: 155 - Loss: 0.324 - Test accuracy: 91.52%
Epoch: 156 - Loss: 0.324 - Test accuracy: 91.51%
Epoch: 157 - Loss: 0.324 - Test accuracy: 91.51%
Epoch: 158 - Loss: 0.323 - Test accuracy: 91.52%
Epoch: 159 - Loss: 0.323 - Test accuracy: 91.51%
Epoch: 160 - Loss: 0.323 - Test accuracy: 91.51%
Epoch: 161 - Loss: 0.323 - Test accuracy: 91.50%
Epoch: 162 - Loss: 0.322 - Test accuracy: 91.51%
Epoch: 163 - Loss: 0.322 - Test accuracy: 91.53%
Epoch: 164 - Loss: 0.322 - Test accuracy: 91.54%
Epoch: 165 - Loss: 0.322 - Test accuracy: 91.54%
Epoch: 166 - Loss: 0.321 - Test accuracy: 91.54%
Epoch: 167 - Loss: 0.321 - Test accuracy: 91.56%
Epoch: 168 - Loss: 0.321 - Test accuracy: 91.56%
Epoch: 169 - Loss: 0.321 - Test accuracy: 91.56%
Epoch: 170 - Loss: 0.320 - Test accuracy: 91.57%
Epoch: 171 - Loss: 0.320 - Test accuracy: 91.59%
Epoch: 172 - Loss: 0.320 - Test accuracy: 91.59%
Epoch: 173 - Loss: 0.320 - Test accuracy: 91.60%
Epoch: 174 - Loss: 0.319 - Test accuracy: 91.60%
Epoch: 175 - Loss: 0.319 - Test accuracy: 91.61%
Epoch: 176 - Loss: 0.319 - Test accuracy: 91.61%
Epoch: 177 - Loss: 0.319 - Test accuracy: 91.61%
Epoch: 178 - Loss: 0.318 - Test accuracy: 91.60%
Epoch: 179 - Loss: 0.318 - Test accuracy: 91.60%
Epoch: 180 - Loss: 0.318 - Test accuracy: 91.60%
Epoch: 181 - Loss: 0.318 - Test accuracy: 91.60%
Epoch: 182 - Loss: 0.318 - Test accuracy: 91.60%
Epoch: 183 - Loss: 0.317 - Test accuracy: 91.60%
Epoch: 184 - Loss: 0.317 - Test accuracy: 91.60%
Epoch: 185 - Loss: 0.317 - Test accuracy: 91.62%
Epoch: 186 - Loss: 0.317 - Test accuracy: 91.63%
Epoch: 187 - Loss: 0.316 - Test accuracy: 91.66%
Epoch: 188 - Loss: 0.316 - Test accuracy: 91.66%
Epoch: 189 - Loss: 0.316 - Test accuracy: 91.65%
Epoch: 190 - Loss: 0.316 - Test accuracy: 91.65%
Epoch: 191 - Loss: 0.316 - Test accuracy: 91.65%
Epoch: 192 - Loss: 0.315 - Test accuracy: 91.65%
Epoch: 193 - Loss: 0.315 - Test accuracy: 91.67%
Epoch: 194 - Loss: 0.315 - Test accuracy: 91.66%
Epoch: 195 - Loss: 0.315 - Test accuracy: 91.66%
Epoch: 196 - Loss: 0.315 - Test accuracy: 91.67%
Epoch: 197 - Loss: 0.314 - Test accuracy: 91.68%
Epoch: 198 - Loss: 0.314 - Test accuracy: 91.68%
Epoch: 199 - Loss: 0.314 - Test accuracy: 91.68%
```

### NN

```bash
go run -race . -model=nn -device=cpu
Epoch: 0         Loss: 2.313     Test accuracy: 23.07%
Epoch: 1         Loss: 2.247     Test accuracy: 32.13%
Epoch: 2         Loss: 2.182     Test accuracy: 44.69%
Epoch: 3         Loss: 2.116     Test accuracy: 57.08%
Epoch: 4         Loss: 2.047     Test accuracy: 64.63%
Epoch: 5         Loss: 1.976     Test accuracy: 68.40%
Epoch: 6         Loss: 1.903     Test accuracy: 70.94%
Epoch: 7         Loss: 1.831     Test accuracy: 72.40%
Epoch: 8         Loss: 1.758     Test accuracy: 74.03%
Epoch: 9         Loss: 1.686     Test accuracy: 75.31%
Epoch: 10        Loss: 1.614     Test accuracy: 76.54%
Epoch: 11        Loss: 1.544     Test accuracy: 77.83%
Epoch: 12        Loss: 1.475     Test accuracy: 78.68%
Epoch: 13        Loss: 1.408     Test accuracy: 79.54%
Epoch: 14        Loss: 1.343     Test accuracy: 80.20%
Epoch: 15        Loss: 1.281     Test accuracy: 80.76%
Epoch: 16        Loss: 1.221     Test accuracy: 81.47%
Epoch: 17        Loss: 1.165     Test accuracy: 81.90%
Epoch: 18        Loss: 1.110     Test accuracy: 82.19%
Epoch: 19        Loss: 1.059     Test accuracy: 82.67%
Epoch: 20        Loss: 1.010     Test accuracy: 83.07%
Epoch: 21        Loss: 0.965     Test accuracy: 83.39%
Epoch: 22        Loss: 0.922     Test accuracy: 83.74%
Epoch: 23        Loss: 0.882     Test accuracy: 84.00%
Epoch: 24        Loss: 0.845     Test accuracy: 84.25%
Epoch: 25        Loss: 0.810     Test accuracy: 84.41%
Epoch: 26        Loss: 0.778     Test accuracy: 84.77%
Epoch: 27        Loss: 0.748     Test accuracy: 85.00%
Epoch: 28        Loss: 0.721     Test accuracy: 85.28%
Epoch: 29        Loss: 0.696     Test accuracy: 85.60%
Epoch: 30        Loss: 0.672     Test accuracy: 85.85%
Epoch: 31        Loss: 0.651     Test accuracy: 86.05%
Epoch: 32        Loss: 0.631     Test accuracy: 86.31%
Epoch: 33        Loss: 0.612     Test accuracy: 86.48%
Epoch: 34        Loss: 0.595     Test accuracy: 86.74%
Epoch: 35        Loss: 0.579     Test accuracy: 86.89%
Epoch: 36        Loss: 0.564     Test accuracy: 87.17%
Epoch: 37        Loss: 0.551     Test accuracy: 87.23%
Epoch: 38        Loss: 0.538     Test accuracy: 87.34%
Epoch: 39        Loss: 0.526     Test accuracy: 87.55%
Epoch: 40        Loss: 0.515     Test accuracy: 87.74%
Epoch: 41        Loss: 0.504     Test accuracy: 88.01%
Epoch: 42        Loss: 0.495     Test accuracy: 88.23%
Epoch: 43        Loss: 0.485     Test accuracy: 88.37%
Epoch: 44        Loss: 0.477     Test accuracy: 88.55%
Epoch: 45        Loss: 0.469     Test accuracy: 88.71%
Epoch: 46        Loss: 0.461     Test accuracy: 88.89%
Epoch: 47        Loss: 0.454     Test accuracy: 88.98%
Epoch: 48        Loss: 0.447     Test accuracy: 89.06%
Epoch: 49        Loss: 0.440     Test accuracy: 89.17%
Epoch: 50        Loss: 0.434     Test accuracy: 89.30%
Epoch: 51        Loss: 0.428     Test accuracy: 89.39%
Epoch: 52        Loss: 0.422     Test accuracy: 89.51%
Epoch: 53        Loss: 0.417     Test accuracy: 89.57%
Epoch: 54        Loss: 0.412     Test accuracy: 89.69%
Epoch: 55        Loss: 0.407     Test accuracy: 89.85%
Epoch: 56        Loss: 0.403     Test accuracy: 89.89%
Epoch: 57        Loss: 0.398     Test accuracy: 89.94%
Epoch: 58        Loss: 0.394     Test accuracy: 90.02%
Epoch: 59        Loss: 0.390     Test accuracy: 90.13%
Epoch: 60        Loss: 0.386     Test accuracy: 90.25%
Epoch: 61        Loss: 0.382     Test accuracy: 90.30%
Epoch: 62        Loss: 0.379     Test accuracy: 90.36%
Epoch: 63        Loss: 0.375     Test accuracy: 90.43%
Epoch: 64        Loss: 0.372     Test accuracy: 90.47%
Epoch: 65        Loss: 0.368     Test accuracy: 90.52%
Epoch: 66        Loss: 0.365     Test accuracy: 90.59%
Epoch: 67        Loss: 0.362     Test accuracy: 90.63%
Epoch: 68        Loss: 0.360     Test accuracy: 90.66%
Epoch: 69        Loss: 0.357     Test accuracy: 90.72%
Epoch: 70        Loss: 0.354     Test accuracy: 90.76%
Epoch: 71        Loss: 0.351     Test accuracy: 90.80%
Epoch: 72        Loss: 0.349     Test accuracy: 90.83%
Epoch: 73        Loss: 0.346     Test accuracy: 90.89%
Epoch: 74        Loss: 0.344     Test accuracy: 90.93%
Epoch: 75        Loss: 0.342     Test accuracy: 91.04%
Epoch: 76        Loss: 0.340     Test accuracy: 91.11%
Epoch: 77        Loss: 0.337     Test accuracy: 91.16%
Epoch: 78        Loss: 0.335     Test accuracy: 91.20%
Epoch: 79        Loss: 0.333     Test accuracy: 91.25%
Epoch: 80        Loss: 0.331     Test accuracy: 91.29%
Epoch: 81        Loss: 0.329     Test accuracy: 91.31%
Epoch: 82        Loss: 0.327     Test accuracy: 91.34%
Epoch: 83        Loss: 0.325     Test accuracy: 91.36%
Epoch: 84        Loss: 0.324     Test accuracy: 91.42%
Epoch: 85        Loss: 0.322     Test accuracy: 91.46%
Epoch: 86        Loss: 0.320     Test accuracy: 91.49%
Epoch: 87        Loss: 0.318     Test accuracy: 91.52%
Epoch: 88        Loss: 0.317     Test accuracy: 91.53%
Epoch: 89        Loss: 0.315     Test accuracy: 91.55%
Epoch: 90        Loss: 0.313     Test accuracy: 91.62%
Epoch: 91        Loss: 0.312     Test accuracy: 91.68%
Epoch: 92        Loss: 0.310     Test accuracy: 91.72%
Epoch: 93        Loss: 0.309     Test accuracy: 91.77%
Epoch: 94        Loss: 0.307     Test accuracy: 91.82%
Epoch: 95        Loss: 0.306     Test accuracy: 91.87%
Epoch: 96        Loss: 0.304     Test accuracy: 91.89%
Epoch: 97        Loss: 0.303     Test accuracy: 91.90%
Epoch: 98        Loss: 0.302     Test accuracy: 91.92%
Epoch: 99        Loss: 0.300     Test accuracy: 91.95%
Epoch: 100       Loss: 0.299     Test accuracy: 91.99%
Epoch: 101       Loss: 0.298     Test accuracy: 92.04%
Epoch: 102       Loss: 0.296     Test accuracy: 92.07%
Epoch: 103       Loss: 0.295     Test accuracy: 92.11%
Epoch: 104       Loss: 0.294     Test accuracy: 92.13%
Epoch: 105       Loss: 0.292     Test accuracy: 92.16%
Epoch: 106       Loss: 0.291     Test accuracy: 92.18%
Epoch: 107       Loss: 0.290     Test accuracy: 92.20%
Epoch: 108       Loss: 0.289     Test accuracy: 92.20%
Epoch: 109       Loss: 0.287     Test accuracy: 92.24%
Epoch: 110       Loss: 0.286     Test accuracy: 92.25%
Epoch: 111       Loss: 0.285     Test accuracy: 92.26%
Epoch: 112       Loss: 0.284     Test accuracy: 92.26%
Epoch: 113       Loss: 0.283     Test accuracy: 92.30%
Epoch: 114       Loss: 0.282     Test accuracy: 92.32%
Epoch: 115       Loss: 0.281     Test accuracy: 92.34%
Epoch: 116       Loss: 0.279     Test accuracy: 92.40%
Epoch: 117       Loss: 0.278     Test accuracy: 92.41%
Epoch: 118       Loss: 0.277     Test accuracy: 92.43%
Epoch: 119       Loss: 0.276     Test accuracy: 92.45%
Epoch: 120       Loss: 0.275     Test accuracy: 92.47%
Epoch: 121       Loss: 0.274     Test accuracy: 92.52%
Epoch: 122       Loss: 0.273     Test accuracy: 92.55%
Epoch: 123       Loss: 0.272     Test accuracy: 92.55%
Epoch: 124       Loss: 0.271     Test accuracy: 92.57%
Epoch: 125       Loss: 0.270     Test accuracy: 92.61%
Epoch: 126       Loss: 0.269     Test accuracy: 92.61%
Epoch: 127       Loss: 0.268     Test accuracy: 92.61%
Epoch: 128       Loss: 0.267     Test accuracy: 92.63%
Epoch: 129       Loss: 0.266     Test accuracy: 92.65%
Epoch: 130       Loss: 0.265     Test accuracy: 92.66%
Epoch: 131       Loss: 0.264     Test accuracy: 92.72%
Epoch: 132       Loss: 0.263     Test accuracy: 92.78%
Epoch: 133       Loss: 0.262     Test accuracy: 92.77%
Epoch: 134       Loss: 0.261     Test accuracy: 92.77%
Epoch: 135       Loss: 0.260     Test accuracy: 92.82%
Epoch: 136       Loss: 0.259     Test accuracy: 92.81%
Epoch: 137       Loss: 0.258     Test accuracy: 92.84%
Epoch: 138       Loss: 0.257     Test accuracy: 92.86%
Epoch: 139       Loss: 0.256     Test accuracy: 92.88%
Epoch: 140       Loss: 0.255     Test accuracy: 92.89%
Epoch: 141       Loss: 0.254     Test accuracy: 92.90%
Epoch: 142       Loss: 0.253     Test accuracy: 92.90%
Epoch: 143       Loss: 0.253     Test accuracy: 92.94%
Epoch: 144       Loss: 0.252     Test accuracy: 92.96%
Epoch: 145       Loss: 0.251     Test accuracy: 92.99%
Epoch: 146       Loss: 0.250     Test accuracy: 93.01%
Epoch: 147       Loss: 0.249     Test accuracy: 93.03%
Epoch: 148       Loss: 0.248     Test accuracy: 93.08%
Epoch: 149       Loss: 0.247     Test accuracy: 93.11%
Epoch: 150       Loss: 0.246     Test accuracy: 93.12%
Epoch: 151       Loss: 0.245     Test accuracy: 93.14%
Epoch: 152       Loss: 0.245     Test accuracy: 93.15%
Epoch: 153       Loss: 0.244     Test accuracy: 93.16%
Epoch: 154       Loss: 0.243     Test accuracy: 93.16%
Epoch: 155       Loss: 0.242     Test accuracy: 93.16%
Epoch: 156       Loss: 0.241     Test accuracy: 93.18%
Epoch: 157       Loss: 0.240     Test accuracy: 93.18%
Epoch: 158       Loss: 0.240     Test accuracy: 93.22%
Epoch: 159       Loss: 0.239     Test accuracy: 93.26%
Epoch: 160       Loss: 0.238     Test accuracy: 93.28%
Epoch: 161       Loss: 0.237     Test accuracy: 93.30%
Epoch: 162       Loss: 0.236     Test accuracy: 93.30%
Epoch: 163       Loss: 0.235     Test accuracy: 93.33%
Epoch: 164       Loss: 0.235     Test accuracy: 93.34%
Epoch: 165       Loss: 0.234     Test accuracy: 93.40%
Epoch: 166       Loss: 0.233     Test accuracy: 93.42%
Epoch: 167       Loss: 0.232     Test accuracy: 93.43%
Epoch: 168       Loss: 0.231     Test accuracy: 93.43%
Epoch: 169       Loss: 0.231     Test accuracy: 93.45%
Epoch: 170       Loss: 0.230     Test accuracy: 93.46%
Epoch: 171       Loss: 0.229     Test accuracy: 93.45%
Epoch: 172       Loss: 0.228     Test accuracy: 93.46%
Epoch: 173       Loss: 0.227     Test accuracy: 93.48%
Epoch: 174       Loss: 0.227     Test accuracy: 93.49%
Epoch: 175       Loss: 0.226     Test accuracy: 93.54%
Epoch: 176       Loss: 0.225     Test accuracy: 93.56%
Epoch: 177       Loss: 0.224     Test accuracy: 93.57%
Epoch: 178       Loss: 0.224     Test accuracy: 93.57%
Epoch: 179       Loss: 0.223     Test accuracy: 93.59%
Epoch: 180       Loss: 0.222     Test accuracy: 93.60%
Epoch: 181       Loss: 0.221     Test accuracy: 93.60%
Epoch: 182       Loss: 0.221     Test accuracy: 93.62%
Epoch: 183       Loss: 0.220     Test accuracy: 93.64%
Epoch: 184       Loss: 0.219     Test accuracy: 93.63%
Epoch: 185       Loss: 0.218     Test accuracy: 93.64%
Epoch: 186       Loss: 0.218     Test accuracy: 93.67%
Epoch: 187       Loss: 0.217     Test accuracy: 93.68%
Epoch: 188       Loss: 0.216     Test accuracy: 93.71%
Epoch: 189       Loss: 0.215     Test accuracy: 93.73%
Epoch: 190       Loss: 0.215     Test accuracy: 93.75%
Epoch: 191       Loss: 0.214     Test accuracy: 93.77%
Epoch: 192       Loss: 0.213     Test accuracy: 93.78%
Epoch: 193       Loss: 0.213     Test accuracy: 93.79%
Epoch: 194       Loss: 0.212     Test accuracy: 93.83%
Epoch: 195       Loss: 0.211     Test accuracy: 93.85%
Epoch: 196       Loss: 0.211     Test accuracy: 93.89%
Epoch: 197       Loss: 0.210     Test accuracy: 93.96%
Epoch: 198       Loss: 0.209     Test accuracy: 93.99%
Epoch: 199       Loss: 0.209     Test accuracy: 94.01%
```


### CNN

**BatchSize = 256 on CPU**

```
 go run . -model=cnn -device=cpu
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.15      Test accuracy: 96.69%
Epoch: 1         Loss: 0.20      Test accuracy: 94.54%
Epoch: 2         Loss: 0.17      Test accuracy: 95.38%
Epoch: 3         Loss: 0.15      Test accuracy: 95.46%
Epoch: 4         Loss: 0.12      Test accuracy: 97.09%
Epoch: 5         Loss: 0.17      Test accuracy: 97.17%
Epoch: 6         Loss: 0.09      Test accuracy: 97.17%
Epoch: 7         Loss: 0.06      Test accuracy: 97.17%
Epoch: 8         Loss: 0.10      Test accuracy: 97.16%
Epoch: 9         Loss: 0.11      Test accuracy: 97.16%
Epoch: 10        Loss: 0.14      Test accuracy: 97.16%
Epoch: 11        Loss: 0.11      Test accuracy: 97.16%
Epoch: 12        Loss: 0.08      Test accuracy: 97.16%
Epoch: 13        Loss: 0.10      Test accuracy: 97.16%
Epoch: 14        Loss: 0.13      Test accuracy: 97.16%
Epoch: 15        Loss: 0.08      Test accuracy: 97.16%
Epoch: 16        Loss: 0.10      Test accuracy: 97.16%
Epoch: 17        Loss: 0.12      Test accuracy: 97.16%
Epoch: 18        Loss: 0.13      Test accuracy: 97.16%
Epoch: 19        Loss: 0.08      Test accuracy: 97.16%
Epoch: 20        Loss: 0.09      Test accuracy: 97.16%
Epoch: 21        Loss: 0.05      Test accuracy: 97.16%
Epoch: 22        Loss: 0.10      Test accuracy: 97.16%
Epoch: 23        Loss: 0.11      Test accuracy: 97.16%
Epoch: 24        Loss: 0.14      Test accuracy: 97.16%
Epoch: 25        Loss: 0.15      Test accuracy: 97.16%
Epoch: 26        Loss: 0.07      Test accuracy: 97.16%
Epoch: 27        Loss: 0.08      Test accuracy: 97.16%
Epoch: 28        Loss: 0.13      Test accuracy: 97.16%
Epoch: 29        Loss: 0.10      Test accuracy: 97.16%
Best test accuracy: 97.17%
Taken time:     9.04 mins
 go run . -model=cnn -device=cpu
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.13      Test accuracy: 96.77%
Epoch: 1         Loss: 0.18      Test accuracy: 95.43%
Epoch: 2         Loss: 0.18      Test accuracy: 95.53%
Epoch: 3         Loss: 0.18      Test accuracy: 96.08%
Epoch: 4         Loss: 0.14      Test accuracy: 96.37%
Epoch: 5         Loss: 0.14      Test accuracy: 96.40%
Epoch: 6         Loss: 0.11      Test accuracy: 96.44%
Epoch: 7         Loss: 0.08      Test accuracy: 96.96%
Epoch: 8         Loss: 0.16      Test accuracy: 97.09%
Epoch: 9         Loss: 0.11      Test accuracy: 97.05%
Epoch: 10        Loss: 0.11      Test accuracy: 97.04%
Epoch: 11        Loss: 0.11      Test accuracy: 97.10%
Epoch: 12        Loss: 0.12      Test accuracy: 97.13%
Epoch: 13        Loss: 0.09      Test accuracy: 97.13%
Epoch: 14        Loss: 0.11      Test accuracy: 97.13%
Epoch: 15        Loss: 0.16      Test accuracy: 97.13%
Epoch: 16        Loss: 0.14      Test accuracy: 97.13%
Epoch: 17        Loss: 0.11      Test accuracy: 97.13%
Epoch: 18        Loss: 0.14      Test accuracy: 97.13%
Epoch: 19        Loss: 0.17      Test accuracy: 97.13%
Epoch: 20        Loss: 0.16      Test accuracy: 97.13%
Epoch: 21        Loss: 0.07      Test accuracy: 97.13%
Epoch: 22        Loss: 0.15      Test accuracy: 97.13%
Epoch: 23        Loss: 0.14      Test accuracy: 97.13%
Epoch: 24        Loss: 0.07      Test accuracy: 97.13%
Epoch: 25        Loss: 0.13      Test accuracy: 97.13%
Epoch: 26        Loss: 0.11      Test accuracy: 97.13%
Epoch: 27        Loss: 0.14      Test accuracy: 97.13%
Epoch: 28        Loss: 0.14      Test accuracy: 97.13%
Epoch: 29        Loss: 0.08      Test accuracy: 97.13%
Best test accuracy: 97.13%
Taken time:     9.37 mins
 go run . -model=cnn -device=cpu
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.13      Test accuracy: 97.03%
Epoch: 1         Loss: 0.14      Test accuracy: 97.43%
Epoch: 2         Loss: 0.10      Test accuracy: 97.37%
Epoch: 3         Loss: 0.13      Test accuracy: 97.35%
Epoch: 4         Loss: 0.15      Test accuracy: 97.37%
Epoch: 5         Loss: 0.06      Test accuracy: 97.68%
Epoch: 6         Loss: 0.12      Test accuracy: 97.19%
Epoch: 7         Loss: 0.08      Test accuracy: 97.68%
Epoch: 8         Loss: 0.13      Test accuracy: 97.89%
Epoch: 9         Loss: 0.10      Test accuracy: 97.32%
Epoch: 10        Loss: 0.10      Test accuracy: 98.25%
Epoch: 11        Loss: 0.07      Test accuracy: 98.26%
Epoch: 12        Loss: 0.07      Test accuracy: 98.39%
Epoch: 13        Loss: 0.09      Test accuracy: 98.43%
Epoch: 14        Loss: 0.07      Test accuracy: 98.44%
Epoch: 15        Loss: 0.09      Test accuracy: 98.49%
Epoch: 16        Loss: 0.06      Test accuracy: 98.48%
Epoch: 17        Loss: 0.05      Test accuracy: 98.48%
Epoch: 18        Loss: 0.06      Test accuracy: 98.48%
Epoch: 19        Loss: 0.04      Test accuracy: 98.48%
Epoch: 20        Loss: 0.08      Test accuracy: 98.48%
Epoch: 21        Loss: 0.11      Test accuracy: 98.48%
Epoch: 22        Loss: 0.09      Test accuracy: 98.48%
Epoch: 23        Loss: 0.06      Test accuracy: 98.48%
Epoch: 24        Loss: 0.06      Test accuracy: 98.48%
Epoch: 25        Loss: 0.05      Test accuracy: 98.48%
Epoch: 26        Loss: 0.05      Test accuracy: 98.48%
Epoch: 27        Loss: 0.05      Test accuracy: 98.48%
Epoch: 28        Loss: 0.07      Test accuracy: 98.48%
Epoch: 29        Loss: 0.10      Test accuracy: 98.48%
Best test accuracy: 98.49%
Taken time:     9.39 mins
 go run . -model=cnn -device=cpu
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.14      Test accuracy: 96.88%
Epoch: 1         Loss: 0.12      Test accuracy: 97.29%
Epoch: 2         Loss: 0.13      Test accuracy: 97.25%
Epoch: 3         Loss: 0.11      Test accuracy: 97.21%
Epoch: 4         Loss: 0.12      Test accuracy: 97.22%
Epoch: 5         Loss: 0.08      Test accuracy: 97.32%
Epoch: 6         Loss: 0.11      Test accuracy: 97.31%
Epoch: 7         Loss: 0.13      Test accuracy: 97.32%
Epoch: 8         Loss: 0.10      Test accuracy: 97.44%
Epoch: 9         Loss: 0.15      Test accuracy: 97.37%
Epoch: 10        Loss: 0.09      Test accuracy: 97.46%
Epoch: 11        Loss: 0.11      Test accuracy: 97.49%
Epoch: 12        Loss: 0.11      Test accuracy: 96.21%
Epoch: 13        Loss: 0.13      Test accuracy: 95.94%
Epoch: 14        Loss: 0.20      Test accuracy: 95.97%
Epoch: 15        Loss: 0.18      Test accuracy: 97.12%
Epoch: 16        Loss: 0.04      Test accuracy: 97.50%
Epoch: 17        Loss: 0.18      Test accuracy: 97.38%
Epoch: 18        Loss: 0.06      Test accuracy: 97.60%
Epoch: 19        Loss: 0.13      Test accuracy: 97.45%
Epoch: 20        Loss: 0.06      Test accuracy: 97.57%
Epoch: 21        Loss: 0.12      Test accuracy: 97.60%
Epoch: 22        Loss: 0.10      Test accuracy: 97.60%
Epoch: 23        Loss: 0.09      Test accuracy: 97.60%
Epoch: 24        Loss: 0.11      Test accuracy: 97.60%
Epoch: 25        Loss: 0.13      Test accuracy: 97.60%
Epoch: 26        Loss: 0.08      Test accuracy: 97.60%
Epoch: 27        Loss: 0.18      Test accuracy: 97.60%
Epoch: 28        Loss: 0.09      Test accuracy: 97.60%
Epoch: 29        Loss: 0.07      Test accuracy: 97.60%
Best test accuracy: 97.60%
Taken time:     9.41 mins

```

**BatchSize = 32 on CUDA**

```bash
 go run . -model=cnn -device=cuda
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.28      Test accuracy: 98.41%
Epoch: 1         Loss: 0.01      Test accuracy: 98.55%
Epoch: 2         Loss: 0.09      Test accuracy: 98.53%
Epoch: 3         Loss: 0.01      Test accuracy: 98.64%
Epoch: 4         Loss: 0.01      Test accuracy: 98.74%
Epoch: 5         Loss: 0.01      Test accuracy: 98.81%
Epoch: 6         Loss: 0.10      Test accuracy: 98.91%
Epoch: 7         Loss: 0.02      Test accuracy: 98.86%
Epoch: 8         Loss: 0.00      Test accuracy: 98.64%
Epoch: 9         Loss: 0.17      Test accuracy: 98.84%
Epoch: 10        Loss: 0.01      Test accuracy: 98.83%
Epoch: 11        Loss: 0.00      Test accuracy: 98.88%
Epoch: 12        Loss: 0.05      Test accuracy: 98.90%
Epoch: 13        Loss: 0.01      Test accuracy: 99.01%
Epoch: 14        Loss: 0.09      Test accuracy: 97.85%
Epoch: 15        Loss: 0.10      Test accuracy: 98.24%
Epoch: 16        Loss: 0.00      Test accuracy: 98.53%
Epoch: 17        Loss: 0.00      Test accuracy: 98.49%
Epoch: 18        Loss: 0.16      Test accuracy: 98.49%
Epoch: 19        Loss: 0.13      Test accuracy: 98.49%
Epoch: 20        Loss: 0.00      Test accuracy: 98.49%
Epoch: 21        Loss: 0.01      Test accuracy: 98.49%
Epoch: 22        Loss: 0.17      Test accuracy: 98.49%
Epoch: 23        Loss: 0.06      Test accuracy: 98.49%
Epoch: 24        Loss: 0.00      Test accuracy: 98.49%
Epoch: 25        Loss: 0.12      Test accuracy: 98.49%
Epoch: 26        Loss: 0.08      Test accuracy: 98.49%
Epoch: 27        Loss: 0.19      Test accuracy: 98.49%
Epoch: 28        Loss: 0.02      Test accuracy: 98.49%
Epoch: 29        Loss: 0.01      Test accuracy: 98.49%
Best test accuracy: 99.01%
Taken time:     8.89 mins


 go run . -model=cnn -device=cuda
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.05      Test accuracy: 98.40%
Epoch: 1         Loss: 0.01      Test accuracy: 98.92%
Epoch: 2         Loss: 0.10      Test accuracy: 98.97%
Epoch: 3         Loss: 0.03      Test accuracy: 98.79%
Epoch: 4         Loss: 0.02      Test accuracy: 98.81%
Epoch: 5         Loss: 0.15      Test accuracy: 98.85%
Epoch: 6         Loss: 0.01      Test accuracy: 98.82%
Epoch: 7         Loss: 0.03      Test accuracy: 98.83%
Epoch: 8         Loss: 0.01      Test accuracy: 98.56%
Epoch: 9         Loss: 0.00      Test accuracy: 98.85%
Epoch: 10        Loss: 0.22      Test accuracy: 98.51%
Epoch: 11        Loss: 0.78      Test accuracy: 98.37%
Epoch: 12        Loss: 0.01      Test accuracy: 98.47%
Epoch: 13        Loss: 0.55      Test accuracy: 98.48%
Epoch: 14        Loss: 0.00      Test accuracy: 98.45%
Epoch: 15        Loss: 0.13      Test accuracy: 98.47%
Epoch: 16        Loss: 0.01      Test accuracy: 98.49%
Epoch: 17        Loss: 0.00      Test accuracy: 98.35%
Epoch: 18        Loss: 0.08      Test accuracy: 98.41%
Epoch: 19        Loss: 0.63      Test accuracy: 98.58%
Epoch: 20        Loss: 0.22      Test accuracy: 98.59%
Epoch: 21        Loss: 0.00      Test accuracy: 98.63%
Epoch: 22        Loss: 0.80      Test accuracy: 98.63%
Epoch: 23        Loss: 0.19      Test accuracy: 98.63%
Epoch: 24        Loss: 0.00      Test accuracy: 98.63%
Epoch: 25        Loss: 0.00      Test accuracy: 98.63%
Epoch: 26        Loss: 0.00      Test accuracy: 98.63%
Epoch: 27        Loss: 0.00      Test accuracy: 98.63%
Epoch: 28        Loss: 0.09      Test accuracy: 98.63%
Epoch: 29        Loss: 0.02      Test accuracy: 98.63%
Best test accuracy: 98.97%
Taken time:     8.85 mins


 go run . -model=cnn -device=cuda
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.39      Test accuracy: 97.83%
Epoch: 1         Loss: 0.01      Test accuracy: 97.95%
Epoch: 2         Loss: 0.00      Test accuracy: 98.74%
Epoch: 3         Loss: 0.00      Test accuracy: 98.64%
Epoch: 4         Loss: 0.07      Test accuracy: 98.62%
Epoch: 5         Loss: 0.01      Test accuracy: 98.75%
Epoch: 6         Loss: 0.01      Test accuracy: 98.76%
Epoch: 7         Loss: 0.26      Test accuracy: 98.33%
Epoch: 8         Loss: 0.04      Test accuracy: 98.44%
Epoch: 9         Loss: 0.12      Test accuracy: 98.60%
Epoch: 10        Loss: 0.00      Test accuracy: 98.60%
Epoch: 11        Loss: 0.51      Test accuracy: 98.60%
Epoch: 12        Loss: 0.05      Test accuracy: 98.60%
Epoch: 13        Loss: 0.12      Test accuracy: 98.60%
Epoch: 14        Loss: 0.00      Test accuracy: 98.60%
Epoch: 15        Loss: 0.03      Test accuracy: 98.60%
Epoch: 16        Loss: 0.03      Test accuracy: 98.60%
Epoch: 17        Loss: 0.25      Test accuracy: 98.60%
Epoch: 18        Loss: 0.18      Test accuracy: 98.35%
Epoch: 19        Loss: 0.18      Test accuracy: 98.42%
Epoch: 20        Loss: 0.01      Test accuracy: 98.40%
Epoch: 21        Loss: 0.01      Test accuracy: 98.66%
Epoch: 22        Loss: 0.11      Test accuracy: 98.71%
Epoch: 23        Loss: 0.17      Test accuracy: 98.72%
Epoch: 24        Loss: 0.21      Test accuracy: 98.72%
Epoch: 25        Loss: 0.00      Test accuracy: 98.72%
Epoch: 26        Loss: 0.00      Test accuracy: 98.72%
Epoch: 27        Loss: 0.00      Test accuracy: 98.72%
Epoch: 28        Loss: 0.06      Test accuracy: 98.72%
Epoch: 29        Loss: 0.11      Test accuracy: 98.72%
Best test accuracy: 98.76%
Taken time:     8.84 mins


 go run . -model=cnn -device=cuda
testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.15      Test accuracy: 98.48%
Epoch: 1         Loss: 0.23      Test accuracy: 98.95%
Epoch: 2         Loss: 0.02      Test accuracy: 98.94%
Epoch: 3         Loss: 0.01      Test accuracy: 99.06%
Epoch: 4         Loss: 0.16      Test accuracy: 99.03%
Epoch: 5         Loss: 0.01      Test accuracy: 99.07%
Epoch: 6         Loss: 0.22      Test accuracy: 98.25%
Epoch: 7         Loss: 0.06      Test accuracy: 98.23%
Epoch: 8         Loss: 0.26      Test accuracy: 98.25%
Epoch: 9         Loss: 0.07      Test accuracy: 98.25%
Epoch: 10        Loss: 0.02      Test accuracy: 98.25%
Epoch: 11        Loss: 0.04      Test accuracy: 98.35%
Epoch: 12        Loss: 0.01      Test accuracy: 98.36%
Epoch: 13        Loss: 0.01      Test accuracy: 98.36%
Epoch: 14        Loss: 0.04      Test accuracy: 98.42%
Epoch: 15        Loss: 0.04      Test accuracy: 98.54%
Epoch: 16        Loss: 0.11      Test accuracy: 98.53%
Epoch: 17        Loss: 0.07      Test accuracy: 98.53%
Epoch: 18        Loss: 0.45      Test accuracy: 98.53%
Epoch: 19        Loss: 0.07      Test accuracy: 98.53%
Epoch: 20        Loss: 0.15      Test accuracy: 98.53%
Epoch: 21        Loss: 0.20      Test accuracy: 98.53%
Epoch: 22        Loss: 0.02      Test accuracy: 98.53%
Epoch: 23        Loss: 0.02      Test accuracy: 98.53%
Epoch: 24        Loss: 0.00      Test accuracy: 98.53%
Epoch: 25        Loss: 0.01      Test accuracy: 98.53%
Epoch: 26        Loss: 0.12      Test accuracy: 98.53%
Epoch: 27        Loss: 0.01      Test accuracy: 98.53%
Epoch: 28        Loss: 0.04      Test accuracy: 98.53%
Epoch: 29        Loss: 0.18      Test accuracy: 98.53%
Best test accuracy: 99.07%
Taken time:     8.82 mins


testImages: [10000 784]
testLabels: [10000]
Epoch: 0         Loss: 0.02      Test accuracy: 98.37%
Epoch: 1         Loss: 0.01      Test accuracy: 98.26%
Epoch: 2         Loss: 0.02      Test accuracy: 98.51%
Epoch: 3         Loss: 0.17      Test accuracy: 98.56%
Epoch: 4         Loss: 0.02      Test accuracy: 98.60%
Epoch: 5         Loss: 0.00      Test accuracy: 98.66%
Epoch: 6         Loss: 0.01      Test accuracy: 98.85%
Epoch: 7         Loss: 0.02      Test accuracy: 98.86%
Epoch: 8         Loss: 0.01      Test accuracy: 98.42%
Epoch: 9         Loss: 0.00      Test accuracy: 98.44%
Epoch: 10        Loss: 0.02      Test accuracy: 98.50%
Epoch: 11        Loss: 0.00      Test accuracy: 98.50%
Epoch: 12        Loss: 0.05      Test accuracy: 98.50%
Epoch: 13        Loss: 0.13      Test accuracy: 98.50%
Epoch: 14        Loss: 0.00      Test accuracy: 98.50%
Epoch: 15        Loss: 0.12      Test accuracy: 98.50%
Epoch: 16        Loss: 0.00      Test accuracy: 98.50%
Epoch: 17        Loss: 0.03      Test accuracy: 98.50%
Epoch: 18        Loss: 0.41      Test accuracy: 98.50%
Epoch: 19        Loss: 0.17      Test accuracy: 98.50%
Epoch: 20        Loss: 0.26      Test accuracy: 98.50%
Epoch: 21        Loss: 0.00      Test accuracy: 98.50%
Epoch: 22        Loss: 0.29      Test accuracy: 98.50%
Epoch: 23        Loss: 0.00      Test accuracy: 98.50%
Epoch: 24        Loss: 0.20      Test accuracy: 98.50%
Epoch: 25        Loss: 0.01      Test accuracy: 98.50%
Epoch: 26        Loss: 0.18      Test accuracy: 98.50%
Epoch: 27        Loss: 0.01      Test accuracy: 98.50%
Epoch: 28        Loss: 0.12      Test accuracy: 98.50%
Epoch: 29        Loss: 0.04      Test accuracy: 98.50%
Best test accuracy: 98.86%
Taken time:     8.77 mins

```
