import torch
import torch.nn as nn
import torch.nn.functional as F
from .TRACER_specular import get_model_shape

class RemDecoder(nn.Module):
    def __init__(self, cfg, out_channel):  # 1046
        super(RemDecoder, self).__init__()
        _, self.channels = get_model_shape(cfg.arch) # [16, 24, 40, 112, 320]
        self.img_channel = out_channel
        self.pixelShuffleRatio = cfg.pixelShuffleRatio
        inner_in_channels, inner_out_channels = self.get_channels() # [320, 448, 160, 96, 64, 16] [448, 160, 96, 64, 64, 3]
        # print(self.inner_in_channels, self.inner_out_channels)

        self.bottomConv = BasicConv2d(inner_in_channels[0], inner_out_channels[0], 3, padding=1)
        self.upmerge1 = BasicUpMerge(inner_in_channels[1], inner_out_channels[1], self.pixelShuffleRatio)
        self.upmerge2 = BasicUpMerge(inner_in_channels[2], inner_out_channels[2], self.pixelShuffleRatio)
        self.upmerge3 = BasicUpMerge(inner_in_channels[3], inner_out_channels[3], self.pixelShuffleRatio)
        self.upmerge4 = BasicUpMerge(inner_in_channels[4], inner_out_channels[4], self.pixelShuffleRatio)
        self.topPixelShuffle = nn.PixelShuffle(self.pixelShuffleRatio)
        self.topConv = nn.Conv2d(inner_in_channels[5], inner_out_channels[5], 3, stride=1, padding=1)
        if self.img_channel == 3:
            self.acti = nn.Tanh()

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
        
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class BasicUpMerge(nn.Module):
    def __init__(self, in_channel, out_channel, pixelShuffleRatio=2, nafblock=2):
        super(BasicUpMerge, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(pixelShuffleRatio)
        channel = in_channel//pixelShuffleRatio
        self.nafblock = nn.Sequential(
                    *[NAFBlock(channel) for _ in range(nafblock)]
                )
        self.basicConv = BasicConv2d(channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        x = self.pixel_shuffle(x1)
        out = torch.cat([x, x2], dim=1)
        out = self.nafblock(out)
        out = self.basicConv(out)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=False)
        # self.bn = nn.BatchNorm2d(out_channel)
        # self.selu = nn.SELU()

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.selu(x)

        return x
