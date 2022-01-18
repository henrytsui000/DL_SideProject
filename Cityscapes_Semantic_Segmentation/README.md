# Project 03: Cityscapes Semantic Segmentation 
## What is semantic segmentation?
Semantic segmentation is a task which we want to cluster parts of an image together of same object class. It can be viewed as a pixel-level prediction task since each pixlel is classified according to a category. 


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