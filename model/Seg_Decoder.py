import torch
import torch.nn as nn
import torch.nn.functional as F
from .TRACER_specular import get_model_shape

class Decoder(nn.Module):
    def __init__(self, cfg, out_channel):  # 1046
        super(Decoder, self).__init__()
        _, self.channels = get_model_shape(cfg.arch) # [16, 24, 40, 112, 320]
        self.img_channel = out_channel
        self.pixelShuffleRatio = cfg.pixelShuffleRatio
        inner_in_channels, inner_out_channels = self.get_channels() # [320, 448, 160, 96, 64, 16] [448, 160, 96, 64, 64, 3]

        self.bottomConv = BasicConv2d(inner_in_channels[0], inner_out_channels[0], 3, padding=1)
        self.upmerge1 = BasicUpMerge(inner_in_channels[1], inner_out_channels[1], self.pixelShuffleRatio)
        self.upmerge2 = BasicUpMerge(inner_in_channels[2], inner_out_channels[2], self.pixelShuffleRatio)
        self.upmerge3 = BasicUpMerge(inner_in_channels[3], inner_out_channels[3], self.pixelShuffleRatio)
        self.upmerge4 = BasicUpMerge(inner_in_channels[4], inner_out_channels[4], self.pixelShuffleRatio)
        self.topPixelShuffle = nn.PixelShuffle(self.pixelShuffleRatio)
        self.topConv = nn.Conv2d(inner_in_channels[5], inner_out_channels[5], 3, stride=1, padding=1)
        self.acti = nn.Sigmoid()

    def get_channels(self):
        in_channels = []
        out_channels = []
        i = len(self.channels)-1
        while i >= 0:
            if i == len(self.channels)-1:
                in_channels.append(self.channels[i])
            elif i == 0:
                temp = self.channels[i]*self.pixelShuffleRatio*self.pixelShuffleRatio
                in_channels.append(temp)
                out_channels.append(temp)
                out_channels.append(temp)
            else:
                temp = self.channels[i]*self.pixelShuffleRatio*self.pixelShuffleRatio
                in_channels.append(temp)
                out_channels.append(temp)
            i -= 1
        in_channels.append(self.channels[0])
        out_channels.append(self.img_channel)
        return in_channels, out_channels

    def forward(self, features):
        x4 = self.bottomConv(features[4]) 
        x3 = self.upmerge1(x4, features[3])
        x2 = self.upmerge2(x3, features[2])
        x1 = self.upmerge3(x2, features[1])
        x0 = self.upmerge4(x1, features[0])

        x = self.topPixelShuffle(x0)
        x = self.acti(self.topConv(x))
        return x 

    

class BasicUpMerge(nn.Module):
    def __init__(self, in_channel, out_channel, pixelShuffleRatio=2):
        super(BasicUpMerge, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(pixelShuffleRatio)
        channel = in_channel//pixelShuffleRatio
        self.basicConv = BasicConv2d(channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = self.pixel_shuffle(x1)
        out = torch.cat([x, x2], dim=1)
        out = self.basicConv(out)
        return out#,x

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        return x
