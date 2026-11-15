# --- Imports ---
import torch
import torch.nn as nn
from typeguard import typechecked
from .._convolution import ConvolutionalBlock
from ...utils.typing import InputChannels

# --- Single NN BackBone Class ---
class SingleStreamBackbone(nn.Module):
    # --- Methods ---
    @typechecked
    def __init__(
            self,
            /,
            in_channels: InputChannels = 3,
            ):
        # --- Super Class ---
        super().__init__()
        # --- Feature Extraction ---
        self.stages = nn.ModuleList([
            # --- Block 1 ---
            ConvolutionalBlock(
                in_channels,
                16,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2),
            # --- Block 2 ---
            ConvolutionalBlock(
                16,
                32,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = True,
                pooling_size = 2),
            # --- Block 3 ---
            ConvolutionalBlock(
                32,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2),
            # --- Block 4 ---
            ConvolutionalBlock(
                64,
                128,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2)
        ])


    def forward(self, x: torch.Tensor) -> tuple[dict, dict]:
        stl = self._stage_iter(x = x, stages = self.stages) 
        # --- Creating Return Dict ---
        r_tensor_dict, r_channel_dict = self._rddc(stl = stl)
        # --- Return ---
        return r_tensor_dict, r_channel_dict

    @staticmethod
    def _stage_iter(*args, **kwargs):
        # --- Value Unpacking ---
        x = kwargs['x']
        stages = kwargs['stages']
        # --- Iteration for C ---
        stl = [] #> Stage Tensors List
        for i, stage in enumerate(stages):  #> Context Stages
            x = stage(x)
            # --- Ignoring First Stage ---
            if i==0:
                continue
            # --- Appending to List ---
            stl.append(x)
        # --- Return ---
        return stl

    def _rddc(self, *args, **kwargs) -> tuple[dict, dict]: #> Return Data Dict Creator
        # --- Variable ---
        r_tensor_dict = {
            "c2": kwargs['stl'][0],
            "c3": kwargs['stl'][1],
            "c4": kwargs['stl'][2]
        }
        r_channel_dict = {
            "c2": 32,
            "c3": 64,
            "c4": 128
        }
        # --- Return ---
        return r_tensor_dict, r_channel_dict