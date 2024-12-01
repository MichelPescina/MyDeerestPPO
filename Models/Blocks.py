import numpy as np
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def layer_init_rnn(layer, std=np.sqrt(2), bias_const=0.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, bias_const)
        elif "weight" in name:
            nn.init.orthogonal_(param, std)
    return layer


class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            in_channels,
            squeeze_channels,
            squeeze_activ = nn.ReLU(),
            excitation_activ = nn.Sigmoid()
        ):
        super().__init__()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            squeeze_activ,
            nn.Conv2d(squeeze_channels, in_channels, 1),
            excitation_activ
        )
    
    def forward(self, x):
        return self.scale(x) * x


class DWSConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            activation,
            layer_init = lambda l: l
        ):
        super(DWSConv2d, self).__init__()
        assert stride == 1 or stride == 2, f"Stride value {stride} not supported!"
        padding = 'same' if stride == 1 else self._padding_stride2(kernel_size)
        self.depth = layer_init(nn.Conv2d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            groups = in_channels,
            stride= stride,
            padding = padding
        ))
        self.point = layer_init(nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
        ))
        self.activ = activation
    
    def _padding_stride2(self, kernel_size):
        p1 = kernel_size//2 - 1
        if kernel_size % 2 != 0:
            p1 = (kernel_size + 1)//2 -1
        padding = p1
        return padding



    def forward(self, x):
        x = self.activ(self.depth(x))
        x = self.activ(self.point(x))
        return x