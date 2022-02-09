# Project 02: Skin Disease Detection

## Introduction

Skin Diseases may cause by different Virus, our target is 
**seprate different Skin Diseases** by ResNet18(Implement by us or Pytorch)
Another challenge is our dataset are unbalanced, thus it's easy to overfitting
## Enviroment
- Model : RESNET18
- Dataset : From Kaggle

To set up enviroment
```bash
$pip install -r ./requirements.txt
```

## Method

### Implement
Use Pytorch's Convolution block make small block, and stack small block to resnet18.

### Pytorch
Just call the command(but we'll compare the model with pretrained or not)
## Data Distributed

![](https://i.imgur.com/z67mkvY.png)

## Final Result
|Methods |Train Accuracy|Validation Accuracy|
|-|-|-|
|Non-pretrained Model|82.81%|72.62%|
|Pretrained Model|93.76%|88.5%|