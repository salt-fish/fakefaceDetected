import torch
import torch.nn as nn

def ConvBN(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0 if kernel_size == 1 else (kernel_size-1)//2),
        nn.BatchNorm2d(out_channels)
    )

def ConvBNRelu(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size, stride),
        nn.ReLU6(inplace=False)
    )

def SeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    )

def SeparableConvolutionRelu(in_channels, out_channels):
    return nn.Sequential(
        SeparableConvolution(in_channels, out_channels),
        nn.ReLU6(inplace=False)
    )

def ReluSeparableConvolution(in_channels, out_channels):
    return nn.Sequential(
        nn.ReLU6(inplace=False),
        SeparableConvolution(in_channels, out_channels)
    )