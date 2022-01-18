# Project 03: Cityscapes Semantic Segmentation 
## What is semantic segmentation?

![](https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/koeln00.png)
Semantic segmentation is a task which we want to cluster parts of an image together of same object class. It can be viewed as a pixel-level prediction task since each pixlel is classified according to a category. 

## Method
We implement [UNet](https://arxiv.org/abs/1505.04597) in this project. Since images in Cityscapes dataset have high resolution, we choose to use UNet to achieve better efficiency. For instance, segmentation of a 512x512 image takes less than a second on a recent GPU as the author states. 

## Dataset
We are using [Cityscapes Dataset](https://www.cityscapes-dataset.com/) in this project. Please download the data [here]() and store it in folder dataset.


## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary libraries
```shell
pip install -r requirements.txt
```

## Train
```shell
python training.py -b batch_size -e num_epoch 
```
## Test
```bash
python testing.py -o output_directory -m model.pth
```

## Final Result
|Model |Accuracy|mIoU|
|-|-|-|
|UNet|74.8%|not available|
