import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

	def __init__(self,n_classes=8, testing=False):
		super(UNet, self).__init__()

		self.features1 = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features2 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, tride=2, padding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128, eps=1e-05, mentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features3 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(128, 256, kernel_size=3, tride=1, padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features4 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2, adding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)

		self.features5 = nn.Sequential(
			nn.MaxPool2d(kernel_size=2, stride=2, adding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU()
		)

		self.deconv1 = nn.Sequential(
			nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		)

		self.deconv2 = nn.Sequential(
			nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		)

		self.deconv3 = nn.Sequential(
			nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		)

		self.deconv4 = nn.Sequential(
			nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
		)

		self.features6 = nn.Sequential(
			nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features7 = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features8 = nn.Sequential(
			nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.features9 = nn.Sequential(
			nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(),
		)

		self.classifier=nn.Conv2d(64, n_classes,kernel_size=1, stride=1)

	def forward(self, x):

		x1 = self.features1(x)
		x2 = self.features2(x1)
		x3 = self.features3(x2)
		x4 = self.features4(x3)
		x5 = self.features5(x4)

		x5 = self.deconv1(x5)
		x5 = torch.cat([x4, x5], dim=1)

		x6 = self.features6(x5)
		x6 = self.deconv2(x6)
		x6 = torch.cat([x3, x6], dim=1)

		x7 = self.features7(x6)
		x7 = self.deconv3(x7)
		x7 = torch.cat([x2, x7], dim=1)

		x8 = self.features8(x7)
		x8 = self.deconv4(x8)
		x8 = torch.cat([x1, x8], dim=1)

		out = self.features9(x8)

		out = self.classifier(out)

		return out