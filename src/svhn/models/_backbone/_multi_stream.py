# --- Imports ---
import torch
import torch.nn as nn
from typeguard import typechecked
from .._convolution import ConvolutionalBlock
from ...utils.typing import InputChannels

# --- Single NN BackBone Class ---
class MultiStreamBackbone(nn.Module): 
    # --- Methods ---
    @typechecked
    def __init__(
            self,
            /,
            in_channels: InputChannels = 3,
            ):
        # --- Super Class ---
        super().__init__()
        # --- NN1 Stages ---
        self.nn1_stages = nn.Sequential(
            # --- Block 1 ---
            ConvolutionalBlock(
                in_channels,
                16,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2)
        )
        # --- NN2 Stages (First Detail Pathway) ---
        self.nn2_stages = nn.Sequential(
            # --- Block 1 ---
            ConvolutionalBlock(
                16,
                32,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2)
        )
        # --- NN3 Stages (First Detail Pathway) ---
        self.nn3_stages = nn.Sequential(
            # --- Block 2 (Refinement) ---
            ConvolutionalBlock(
                32,
                32,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2),
            # --- Block 3 (Refinement) ---
            ConvolutionalBlock(
                32,
                32,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2)
        )
        # --- NN4 Stages (Second Detail Pathway) ---
        self.nn4_stages = nn.Sequential(
            # --- Block 1 (Detail Stage D2) ---
            ConvolutionalBlock(
                32,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2),
            # --- Block 2 (Refinement) ---
            ConvolutionalBlock(
                64,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2),
            # --- Block 3 (Refinement) ---
            ConvolutionalBlock(
                64,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 1,
                pooling = False,
                pooling_size = 2)
        )
        # --- NN5 Stages (Context Pathway) ---
        self.nn5_stages = nn.ModuleList([
            # --- Block 1 ---
            ConvolutionalBlock(
                16,
                32,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = True,
                pooling_size = 2),
            # --- Block 2 ---
            ConvolutionalBlock(
                32,
                64,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2),
            # --- Block 3 ---
            ConvolutionalBlock(
                64,
                128,
                kernel_size = 3,
                padding = 1,
                stride = 2,
                pooling = False,
                pooling_size = 2),
        ])


    def forward(self, x: torch.Tensor) -> tuple[dict, dict]:
        x = self.nn1_stages(x) #> First Stage
        # --- Detail Pathway Stages ---
        d = self.nn2_stages(x) #> Parent Detail Stages
        d1 = self.nn3_stages(d) #> First Detail Stages
        d2 = self.nn4_stages(d) #> Second Detail Stages
        # --- Context Pathway ---
        stl = self._stage_iter(x = x, stages = self.nn5_stages) 
        # --- Creating Return Dict ---
        r_tensor_dict, r_channel_dict = self._rddc(
            d1 = d1, d2 = d2, stl = stl)
        # --- Return ---
        return r_tensor_dict, r_channel_dict

    @staticmethod
    def _stage_iter(*args, **kwargs):
        # --- Value Unpacking ---
        x = kwargs['x']
        stages = kwargs['stages']
        # --- Iteration for C ---
        stl = [] #> Stage Tensors List
        for stage in stages:  #> Context Stages
            x = stage(x)
            stl.append(x)
        # --- Return ---
        return stl

    def _rddc(self, *args, **kwargs) -> tuple[dict, dict]:
        # --- Variable ---
        r_tensor_dict = {
            "d1": kwargs['d1'],
            "d2": kwargs['d2'],
            "c2": kwargs['stl'][0],
            "c3": kwargs['stl'][1],
            "c4": kwargs['stl'][2]
        }
        r_channel_dict = {
            "d1": 32,
            "d2": 64,
            "c2": 32,
            "c3": 64,
            "c4": 128
        }
        # --- Return ---
        return r_tensor_dict, r_channel_dict