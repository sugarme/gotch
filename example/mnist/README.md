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




