import os
import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision

from .utils import ConvBNRelu, SeparableConvolution, ReluSeparableConvolution, ConvBN


class SeparableNet(nn.Module):
	def __init__(self, num_classes=2):
		super(SeparableNet, self).__init__()

		self.ConvBnRelu = ConvBNRelu(in_channels=3, out_channels=8, kernel_size=3, stride=2)

		self.Flow1_SeparableConv = SeparableConvolution(in_channels=8, out_channels=10)
		self.Flow1_ReluSeparableConv = ReluSeparableConvolution(in_channels=10, out_channels=10)
		self.Flow1_MaxPool2d = nn.MaxPool2d(kernel_size=2)
		self.Flow1_shortcut = ConvBN(in_channels=8, out_channels=10, kernel_size=1, stride=2)

		self.Flow2_ReluSeparableConv_1 = ReluSeparableConvolution(in_channels=10, out_channels=12)
		self.Flow2_ReluSeparableConv_2 = ReluSeparableConvolution(in_channels=12, out_channels=12)
		self.Flow2_MaxPool2d = nn.MaxPool2d(kernel_size=2)
		self.Flow2_shortcut = ConvBN(in_channels=10, out_channels=12, kernel_size=1, stride=2)

		self.Flow3_ReluSeparableConv_1 = ReluSeparableConvolution(in_channels=12, out_channels=16)
		self.Flow3_ReluSeparableConv_2 = ReluSeparableConvolution(in_channels=16, out_channels=16)
		self.Flow3_MaxPool2d = nn.MaxPool2d(kernel_size=2)
		self.Flow3_shortcut = ConvBN(in_channels=12, out_channels=16, kernel_size=1, stride=2)

		# Normal Layer
		self.relu = nn.ReLU(inplace=True)
		self.leakyrelu = nn.LeakyReLU(0.1)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
		self.maxpooling2 = nn.MaxPool2d(kernel_size=(2, 2))

		self.dropout = nn.Dropout2d(0.5)
		self.fc1 = nn.Linear(16 * 8 * 8, 16)
		self.fc2 = nn.Linear(16, num_classes)

	def Flow1(self, input):
		shortcut = self.Flow1_shortcut(input)

		x = self.Flow1_SeparableConv(input)
		x = self.Flow1_ReluSeparableConv(x)
		x = self.Flow1_MaxPool2d(x)

		return shortcut + x

	def Flow2(self, input):
		shortcut = self.Flow2_shortcut(input)

		x = self.Flow2_ReluSeparableConv_1(input)
		x = self.Flow2_ReluSeparableConv_2(x)
		x = self.Flow2_MaxPool2d(x)

		return shortcut + x

	def Flow3(self, input):
		shortcut = self.Flow3_shortcut(input)

		x = self.Flow3_ReluSeparableConv_1(input)
		x = self.Flow3_ReluSeparableConv_2(x)
		x = self.Flow3_MaxPool2d(x)

		return shortcut + x

	def forward(self, input):
		# input 3*256*256
		x = self.ConvBnRelu(input)  # (batch, 8, 128, 128)

		x = self.Flow1(x)  # (batch, 10, 64, 64)
		x = self.Flow2(x)  # (batch, 12, 32, 32)
		x = self.Flow3(x)  # (batch, 16, 16, 16)

		x = self.conv2(x)  # (batch, 16, 16, 16)
		x = self.relu(x)
		x = self.bn1(x)
		x = self.maxpooling2(x)  # (Batch, 16, 8, 8)
		feature = x

		x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
		x = self.dropout(x)
		x = self.fc1(x)  # (Batch, 16)
		x = self.leakyrelu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x, feature

