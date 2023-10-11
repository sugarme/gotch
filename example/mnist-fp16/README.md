# CNN MNIST training Float vs BFloat16

## BFloat16 - 16bit floating point

```bash
testImages: [10000 784]
testLabels: [10000]
Start eval...Epoch: 0    Loss: 0.05      Test accuracy: 98.05%
Start eval...Epoch: 1    Loss: 0.03      Test accuracy: 98.36%
Start eval...Epoch: 2    Loss: 0.03      Test accuracy: 98.44%
Start eval...Epoch: 3    Loss: 0.18      Test accuracy: 98.44%
Start eval...Epoch: 4    Loss: 0.01      Test accuracy: 98.52%
Start eval...Epoch: 5    Loss: 0.06      Test accuracy: 98.52%
Start eval...Epoch: 6    Loss: 0.21      Test accuracy: 98.52%
Start eval...Epoch: 7    Loss: 0.05      Test accuracy: 98.59%
Start eval...Epoch: 8    Loss: 0.12      Test accuracy: 98.52%
Start eval...Epoch: 9    Loss: 0.12      Test accuracy: 98.48%
Start eval...Epoch: 10   Loss: 0.04      Test accuracy: 98.52%
Start eval...Epoch: 11   Loss: 0.03      Test accuracy: 98.52%
Start eval...Epoch: 12   Loss: 0.04      Test accuracy: 98.48%
Start eval...Epoch: 13   Loss: 0.32      Test accuracy: 98.48%
Start eval...Epoch: 14   Loss: 0.06      Test accuracy: 98.52%
Start eval...Epoch: 15   Loss: 0.10      Test accuracy: 98.55%
Start eval...Epoch: 16   Loss: 0.02      Test accuracy: 98.52%
Start eval...Epoch: 17   Loss: 0.01      Test accuracy: 98.48%
Start eval...Epoch: 18   Loss: 0.01      Test accuracy: 98.67%
Start eval...Epoch: 19   Loss: 0.10      Test accuracy: 98.63%
Start eval...Epoch: 20   Loss: 0.05      Test accuracy: 98.71%
Start eval...Epoch: 21   Loss: 0.01      Test accuracy: 98.79%
Start eval...Epoch: 22   Loss: 0.05      Test accuracy: 98.71%
Start eval...Epoch: 23   Loss: 0.03      Test accuracy: 98.67%
Start eval...Epoch: 24   Loss: 0.03      Test accuracy: 98.67%
Start eval...Epoch: 25   Loss: 0.16      Test accuracy: 98.75%
Start eval...Epoch: 26   Loss: 0.07      Test accuracy: 98.75%
Start eval...Epoch: 27   Loss: 0.01      Test accuracy: 98.75%
Start eval...Epoch: 28   Loss: 0.15      Test accuracy: 98.63%
Start eval...Epoch: 29   Loss: 0.01      Test accuracy: 98.59%
Best test accuracy: 98.79%
Taken time:     8.67 mins
```


## Float - 32bit floating point

```bash
testImages: [10000 784]
testLabels: [10000]
Start eval...Epoch: 0    Loss: 0.27      Test accuracy: 98.42%
Start eval...Epoch: 1    Loss: 0.06      Test accuracy: 98.60%
Start eval...Epoch: 2    Loss: 0.01      Test accuracy: 98.68%
Start eval...Epoch: 3    Loss: 0.01      Test accuracy: 98.63%
Start eval...Epoch: 4    Loss: 0.11      Test accuracy: 98.82%
Start eval...Epoch: 5    Loss: 0.11      Test accuracy: 99.00%
Start eval...Epoch: 6    Loss: 0.00      Test accuracy: 98.93%
Start eval...Epoch: 7    Loss: 0.00      Test accuracy: 98.96%
Start eval...Epoch: 8    Loss: 0.01      Test accuracy: 99.02%
Start eval...Epoch: 9    Loss: 0.04      Test accuracy: 99.04%
Start eval...Epoch: 10   Loss: 0.06      Test accuracy: 99.07%
Start eval...Epoch: 11   Loss: 0.01      Test accuracy: 99.12%
Start eval...Epoch: 12   Loss: 0.00      Test accuracy: 99.12%
Start eval...Epoch: 13   Loss: 0.00      Test accuracy: 99.12%
Start eval...Epoch: 14   Loss: 0.04      Test accuracy: 99.14%
Start eval...Epoch: 15   Loss: 0.07      Test accuracy: 99.12%
Start eval...Epoch: 16   Loss: 0.00      Test accuracy: 99.08%
Start eval...Epoch: 17   Loss: 0.00      Test accuracy: 99.10%
Start eval...Epoch: 18   Loss: 0.08      Test accuracy: 99.16%
Start eval...Epoch: 19   Loss: 0.07      Test accuracy: 99.20%
Start eval...Epoch: 20   Loss: 0.00      Test accuracy: 99.06%
Start eval...Epoch: 21   Loss: 0.05      Test accuracy: 98.97%
Start eval...Epoch: 22   Loss: 0.01      Test accuracy: 99.13%
Start eval...Epoch: 23   Loss: 0.00      Test accuracy: 99.13%
Start eval...Epoch: 24   Loss: 0.01      Test accuracy: 99.16%
Start eval...Epoch: 25   Loss: 0.00      Test accuracy: 99.11%
Start eval...Epoch: 26   Loss: 0.09      Test accuracy: 99.13%
Start eval...Epoch: 27   Loss: 0.00      Test accuracy: 99.14%
Start eval...Epoch: 28   Loss: 0.00      Test accuracy: 99.13%
Start eval...Epoch: 29   Loss: 0.01      Test accuracy: 99.20%
Best test accuracy: 99.20%
Taken time:     3.06 mins
```
