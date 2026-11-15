import torch
import torch.nn as nn
from typeguard import typechecked
from ..utils.typing import ConvAcFuncTypes, ConvPaddingTypes
from ..utils.typing import InputChannels

class ConvolutionalBlock(nn.Module):
    
    @typechecked
    def __init__(
        self, 
        in_channels: InputChannels, 
        out_channels: int,
        /,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1, 
        *,
        pooling: bool = False,
        pooling_size: int = 2,
        batch_norm: bool = True,
        activ_func: ConvAcFuncTypes = "ReLU",
        conv_bias: bool | None = False,
        conv_padding_type: ConvPaddingTypes = 'replicate'):
        # --- Super Properties ---
        super(ConvolutionalBlock, self).__init__()
        # --- Convolution Layers ---
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, stride, padding, 
            bias = conv_bias,
            padding_mode = conv_padding_type)
        # --- Pooling ---
        self.pool_status = pooling
        if pooling:
            self.pool = nn.MaxPool2d(pooling_size)
        # --- Batch Normalization ---
        self.batch_status = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        # --- Activation Funcs ---
        if activ_func == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activ_func == "LeakyReLU":
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(
                f"Unsupported activation func: {activ_func}")

    @typechecked
    def forward(
            self, 
            x: torch.Tensor
            ) -> torch.Tensor:
        # --- Conv Layer ---
        x = self.conv(x)
        # --- Batch Norm ---
        x = self._batch_norm(x)
        # --- Pooling ---
        x = self.activation(x)
        # --- Pooling ---
        x = self._pooling(x)
        # --- Return ---
        return x

    def _pooling(self, x):
        if self.pool_status:
            x = self.pool(x)
            return x
        else:
            return x

    def _batch_norm(self, x):
        if self.batch_status:
            x = self.bn(x)
            return x
        else:
            return x

   