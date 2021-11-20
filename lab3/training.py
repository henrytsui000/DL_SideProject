import argparse
import os
import time
import datetime
import json
import torch
import torch.optim as optim
from math import ceil
from torch.utils.data import DataLoader
from dataset.dataset import CityScapesDataset
#from demo_test_utils.dataset import CityScapesDataset
from dataset.augmentation import get_composed_augmentations
from utils.core_utils import *

from utils.metric import runningScore, averageMeter

from tensorboardX import SummaryWriter
from model.FCN import FCN
from model.UNet import UNet
from loss import cross_entropy2d

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def train(epoch, data_loader, Net, optimizer, loss_fn, log_file, Meter, writer):
	Net.train()
	timeStart = time.time()
	for i, (data, target) in enumerate(data_loader):
		data , target = data.to(device),target.to(device)
		
		##yourself
		pred = Net(data)
		# loss function
		loss   = loss_fn(pred,target)
		optimizer.zero_grad()  
		# do back propagation
		loss.backward()
		optimizer.step()
		
		training_loss = loss.item()
		pred = pred.data.max(1)[1]
		Meter['metric'].update(target.data.cpu().numpy(), pred.data.cpu().numpy())
		Meter['loss'].update(training_loss,data.size()[0])
	timeEnd = time.time()
	score, class_iou = Meter['metric'].get_scores()
	loss_avg = Meter['loss'].avg
	writer.add_scalars('loss',{'train':loss_avg},epoch)
	print_with_write(log_file ,'epoch %3d : %10s loss: %f OverallAcc: %f MeanAcc %f mIoU %f time: %f' %(epoch, ('training'), loss_avg, score['OverallAcc'], score['MeanAcc'], score['mIoU'], timeEnd-timeStart))

    

def val(epoch, data_loader, Net, loss_fn, log_file, Meter,writer):
	Net.eval()
	with torch.no_grad():
		for i, (data, target) in enumerate(data_loader):
			data, target = data.to(device), target.to(device)
			timeStart = time.time()
			pred = Net(data)
			timeEnd = time.time()
			validation_loss = loss_fn(pred,target).item()
			pred = pred.data.max(1)[1]
			Meter['metric'].update(target.data.cpu().numpy(), pred.data.cpu().numpy())
			Meter['loss'].update(validation_loss,data.size()[0])
			Meter['time'].update(timeEnd-timeStart,1)
	score, class_iou = Meter['metric'].get_scores()
	loss_avg = Meter['loss'].avg
	time_avg = Meter['time'].avg
	writer.add_scalars('loss',{'val':loss_avg},epoch)
	print_with_write(log_file ,'epoch %3d : %10s loss: %f OverallAcc: %f MeanAcc %f mIoU %f time: %f' 
        %(epoch, ('validation'), loss_avg, score['OverallAcc'], score['MeanAcc'], score['mIoU'], time_avg))




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-b','--batchsize',   type=int,            default=4,          help='input batch size')
	parser.add_argument('-e','--epoch',       type=int,            default=10,        help='number of epochs')
	parser.add_argument('-i','--img-size',    nargs='+', type=int, default=[256,512], help='resize to imgsize')
	parser.add_argument('-m','--model-name',  type=str,            default='model',    help='for name of save model')
	parser.add_argument('-o','--output-path', type=str,            default='log',      help='output directory(including log and savemodel)')
	parser.add_argument('-r','--resume',      type=str,            default=None,       help='the file name of checkpoint you want to resume')
	parser.add_argument('-t','--task',        type=str,            default='cat',      help='the training task: cat')
	opt = parser.parse_args()
	
	batchsize   = opt.batchsize*len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
	Epoch       = opt.epoch
	img_size    = tuple(opt.img_size)
	model_name  = opt.model_name
	output_path = opt.output_path
	resume      = opt.resume
	task        = opt.task
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
	assert task in ['cat'],'wrong value of task'
	if task=='cat':
		num_classes = 8
	writer = SummaryWriter(os.path.join(output_path,'events','graph'))
	training_meter = {'metric':runningScore(num_classes),'loss':averageMeter(),'time':averageMeter()}
	validation_meter = {'metric':runningScore(num_classes),'loss':averageMeter(),'time':averageMeter()}
	
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	if not os.path.exists(os.path.join(output_path,'savemodel')):
		os.makedirs(os.path.join(output_path,'savemodel'))
	if resume is None:
		log_file = open(os.path.join(output_path,'log_train.txt'),'w')
	else:
		log_file = open(os.path.join(output_path,'log_train.txt'),'a')
	
	print_with_write(log_file ,str(datetime.datetime.now()))
	print_with_write(log_file ,str(opt))
	
	with open('augmentation.json','r') as f:
		aug_dict = json.load(f)

	training_augmentation = get_composed_augmentations(aug_dict)
	TrainingDataset   = CityScapesDataset('data', 'training', img_size, task=task)#, augmentation=training_augmentation)
	ValidationDataset = CityScapesDataset('data', 'validation', img_size, task=task)
	
	TrainingLoader    = DataLoader(  TrainingDataset, batch_size=batchsize, shuffle=True, num_workers=8)
	ValidationLoader  = DataLoader(ValidationDataset, batch_size=batchsize, shuffle=False, num_workers=8)
	num_batch         = ceil(len(TrainingDataset)/batchsize)
	
	# define yout model
	#Net = FCN()
	Net = UNet()
	Net.to(device=device)
	torch.backends.cudnn.benchmark = True

    

	# define your loss
	loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255) 
	
	start_epoch = 0
	
	
	# define your optimizer
	lr=0.005
	optimizer = torch.optim.Adam(Net.parameters(),lr=lr)
	
	if resume is not None:
		Net, optimizer, start_epoch = load_model(Net, optimizer, resume, log_file)
		print_with_write(log_file,'Done!')
	
	for epoch in range(start_epoch, Epoch):
		for _, v in training_meter.items():
			v.reset()
		
		train(epoch, TrainingLoader, Net, optimizer, loss_fn, log_file, training_meter, writer)
		if (epoch+1)%10==0 or epoch==Epoch-1:
			for _, v in validation_meter.items():
				v.reset()
			print("val")
			val(epoch, ValidationLoader, Net, loss_fn, log_file, validation_meter, writer)
			save_model({'epoch':epoch,
                        'model_state_dict':Net.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
						},
						os.path.join(output_path,'savemodel'),model_name)
	print_with_write(log_file ,str(datetime.datetime.now()))
	writer.close()
