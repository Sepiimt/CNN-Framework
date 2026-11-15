
import torch.nn as nn
import torch.nn.functional as F

from ._convolution import ConvolutionalBlock


class FeaturePyramidNeck(nn.Module):

    def __init__(
        self,
        /,
        low_channels: int, #> default = 32
        high_channels: int, #> default = 128
        *,
        out_channels: int = 64
        ):
        # --- Super Class ---
        super().__init__()
        # --- Lateral Convolutions ---
        self.lateral_low = nn.Conv2d( #> Lateral Convolutions to fix Channel Mismatch
            low_channels,
            out_channels,
            kernel_size=1
        )
        self.lateral_high = nn.Conv2d( #> Lateral Convolutions to fix Channel Mismatch
            high_channels,
            out_channels,
            kernel_size=1
        )
        # --- Smoothing Convolutions ---
        self.smooth_low = ConvolutionalBlock( #> Refinement Convolutions to Refine the Combined Representation
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            pooling=False
        )
        self.smooth_high = ConvolutionalBlock( #> Refinement Convolutions to Refine the Combined Representation
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            pooling=False
        )
        # --- Output Values ---
        self.out_channels = out_channels


    def forward(self, backbone_dict):
        # --- Receive Backbone Features ---
        low = backbone_dict["low"][0]
        high = backbone_dict["high"][0]
        # --- Lateral Connections ---
        low = self.lateral_low(low)
        high = self.lateral_high(high)
        # --- Top-Down Upsampling ---
        high_up = F.interpolate(  #> Upsampling to fix Spatial Dimensions Mismatch
            high,
            size=low.shape[-2:], #> Shape = (N,C,H,W)
            mode="nearest")
        # --- Feature Fusion ---
        fused_low = low + high_up #> Central Operation of the FPN
        # --- Feature Refinement ---
        output_low = self.smooth_low(fused_low)
        output_high = self.smooth_high(high)
        # --- Return Feature Pyramid ---
        return {
            "fine": output_low,
            "coarse": output_high}
