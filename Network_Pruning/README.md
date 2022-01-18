# Network Pruning

## Introduction

There is a technique called Network Pruning, which is set Model parameters to zero, but won't lose a lot accuracy.

It will let the network lightweight, and decrease training time.

## Enviroment
- Model : RESNET18
- Dataset : CIFAR10

To set up enviroment
```bash
$pip install -r ./requirements.txt
```

## Method

There are two method below
### Ratio

We set the parameters which small than the threshold to 0, let the numbers of zero is x%(x is a hyper-parameter)
### Coarse

We will set some row of RESNET model to 0, and retrain the model.
## Parameters Distributed

![](https://i.imgur.com/EiIXOBz.png)

## Final Result
|Methods |Accuracy|Pruning rate|
|-|-|-|
|Ratio|98.2%|90%|
|Coarse|70%|85%|