import torch
import torch.nn as nn
from torchvision.ops.misc import ConvNormActivation
from functools import partial

# Modified from S3D with conv2ds (https://github.com/pytorch/vision/blob/main/torchvision/models/video/s3d.py)

class TemporalSeparableConv(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer,
    ):
        super().__init__(
            ConvNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size),
                stride=(1, stride),
                padding=(0, padding),
                bias=False,
                norm_layer=norm_layer,
            ),
            ConvNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size, 1),
                stride=(stride, 1),
                padding=(padding, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
        )
        
class SepInceptionBlock2D(nn.Module):
    def __init__(
        self,
        in_planes: int,
        b0_out: int,
        b1_mid: int,
        b1_out: int,
        b2_mid: int,
        b2_out: int,
        b3_out: int,
        norm_layer,
    ):
        super().__init__()

        self.branch0 = ConvNormActivation(in_planes, b0_out, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.branch1 = nn.Sequential(
            ConvNormActivation(in_planes, b1_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(b1_mid, b1_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
        )
        self.branch2 = nn.Sequential(
            ConvNormActivation(in_planes, b2_mid, kernel_size=1, stride=1, norm_layer=norm_layer),
            TemporalSeparableConv(b2_mid, b2_out, kernel_size=3, stride=1, padding=1, norm_layer=norm_layer),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1),
            ConvNormActivation(in_planes, b3_out, kernel_size=1, stride=1, norm_layer=norm_layer),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class S2D(nn.Module):
    def __init__(self,n_frames):
        super().__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.001)
        self.features = nn.Sequential(
                                      TemporalSeparableConv(n_frames, 64, 7, 2, 3, norm_layer),
                                      nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
                                      ConvNormActivation(64, 64, kernel_size=1, stride=1, norm_layer=norm_layer,),
                                      TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer),
                                      nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
                                      SepInceptionBlock2D(192, 64, 96, 128, 16, 32, 32, norm_layer),
                                      SepInceptionBlock2D(256, 128, 128, 192, 32, 96, 64, norm_layer),
                                      nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                                      SepInceptionBlock2D(480, 192, 96, 208, 16, 48, 64, norm_layer),
                                      SepInceptionBlock2D(512, 160, 112, 224, 24, 64, 64, norm_layer),
                                      SepInceptionBlock2D(512, 128, 128, 256, 24, 64, 64, norm_layer),
                                      SepInceptionBlock2D(512, 112, 144, 288, 32, 64, 64, norm_layer),
                                      SepInceptionBlock2D(528, 256, 160, 320, 32, 128, 128, norm_layer),
                                      nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
                                      SepInceptionBlock2D(832, 256, 160, 320, 32, 128, 128, norm_layer),
                                      SepInceptionBlock2D(832, 384, 192, 384, 48, 128, 128, norm_layer),
                                  )
        self.avgpool = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Flatten(1))
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x