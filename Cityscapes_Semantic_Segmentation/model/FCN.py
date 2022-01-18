import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCN(nn.Module):
	
	def __init__(self, num_classes=8, testing=False):
		super(FCN, self).__init__()
		
		self.testing = testing
		self.features1 = nn.Sequential(
				nn.Conv2d( 3, 64,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(64),
				nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(64),
				nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
			)
				
		self.features2 = nn.Sequential(
				nn.Conv2d( 64,128,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(128),
				nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(128),
				nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False),
			)
		self.features3 = nn.Sequential(
				nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(256),
				nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(256),
				nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
			)
		self.features4 = nn.Sequential(
				nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(512),
				nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
				nn.BatchNorm2d(512,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True),
				nn.ReLU(512),
				nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,ceil_mode=False)
			)
		self.deconv1 = nn.Sequential(
				nn.ConvTranspose2d(512,256,kernel_size=3,stride=2,padding=1,output_padding=1),
				nn.ReLU(256),
				nn.BatchNorm2d(256,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
			)
		self.deconv2 = nn.Sequential(
				nn.ConvTranspose2d(256,128,kernel_size=3,stride=2,padding=1,output_padding=1),
				nn.ReLU(128),
				nn.BatchNorm2d(128,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
			)
		self.deconv3 = nn.Sequential(
				nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1),
				nn.ReLU(64),
				nn.BatchNorm2d(64,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
			)
		self.deconv4 = nn.Sequential(
				nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1),
				nn.ReLU(32),
				nn.BatchNorm2d(32,eps=1e-05,momentum=0.1,affine=True,track_running_stats=True)
			)
		self.classifier=nn.Conv2d(32,8,kernel_size=1,stride=1)
	def forward(self, x):
		
		out = self.features1(x)
		out = self.features2(out)
		out = self.features3(out)
		out = self.features4(out)
		out = self.deconv1(out)
		out = self.deconv2(out)
		out = self.deconv3(out)
		out = self.deconv4(out)
		
		if self.testing is True:
			out = F.interpolate(out, scale_factor=4)
		out = self.classifier(out)                    
		
		return out     
		
        
		
