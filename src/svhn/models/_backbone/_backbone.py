# --- Imports ---
import torch
import torch.nn as nn
from typeguard import typechecked
from ._single_stream import SingleStreamBackbone
from ._multi_stream import MultiStreamBackbone
from .._convolution import ConvolutionalBlock

# --- BackBone Class ---
class BackBone(nn.Module):
    # --- Methods ---
    @typechecked
    def __init__(
            self,
            /,
            in_channels: int = 3,
            *,
            channel_selection_criteria: list = [2,3,4]
            ):
        # --- Super Class ---
        super().__init__()
        # --- Feature Extraction ---
        #> `nn.Sequential`` is a PyTorch container that connects multiple modules in a sequence. Useful when we have a simple sequence.
        #> `nn.ModuleList` Stores modules, but does not automatically execute them. We must iterate over them ourself.
        self.stages = nn.ModuleList([
            # --- Block 1 ---
            ConvolutionalBlock(
                in_channels,
                16,
                kernel_size=3,
                padding=1,
                pooling=False,
                pooling_size=2),
            # --- Block 2 ---
            ConvolutionalBlock(
                16,
                32,
                kernel_size=3,
                padding=1,
                pooling=False),
            # --- Block 3 ---
            ConvolutionalBlock(
                32,
                64,
                kernel_size=3,
                pooling=True,
                pooling_size=2),
            # --- Block 4 ---
            ConvolutionalBlock(
                64,
                128,
                kernel_size=3,
                padding=1,
                pooling=False)
        ])
        # --- Output Information ---
        self.out_channels = 128
        self.__csc = [i-1 for i in channel_selection_criteria] #> Channel Selection Criteria
        # --- Stage Info ---
        self.__STAGE_INFO = {
            "1": 16,
            "2": 32,
            "3": 64,
            "4": 128}


    def forward(
            self, 
            x: torch.Tensor
            ) -> dict:
        # --- Iteration ---
        for it, stage in enumerate(self.stages):
            x = stage(x)
            # --- Iteration Variable ---
            stage_tensors_lst = []
            # --- Saving Layer Results ---
            if it in self.__csc:
                stage_tensors_lst.append(x)
        # --- Creating Return Dict ---
        ret_dict = self._mddc(
            stl = stage_tensors_lst)
        # --- Return ---
        return ret_dict

    def _mddc(self, *args, **kwargs): #> Meta Data Dict Creator
        # --- Iteration Variable ---
        ret_dict = {}
        # --- Creating Return Dict ---
        for stage, tensor in enumerate(kwargs['stl']):
            ret_dict[f"{self.__csc[stage]+1}st"] = (tensor, 
                                        self.__STAGE_INFO[f"{self.__csc[stage]}"])
        # --- Return ---
        return ret_dict