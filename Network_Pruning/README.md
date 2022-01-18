# Network Prunning

## Introduction

The 

## Enviroment
- Model : RESNET18
- Dataset : CIFAR10


## Ratio

We set the parameters which small than the threshold to 0, let the numbers of zero is x%(x is a hyper-parameter)
## Coarse

We will set some row of RESNET model to 0, and retrain the model.
## Parameters Distributed

![](https://i.imgur.com/EiIXOBz.png)

## Final Result
|Methods |Accuracy|Pruning rate|
|-|-|-|
|Ratio|98.2%|90%|
|Coarse|70%|85%|