import torch
import torch.nn as nn
import torch.nn.functional as F
from .Seg_Decoder import BasicConv2d
from .TRACER_specular import get_model_shape
import numpy as np

class HSI_Encoder(nn.Module):
    def __init__(self, cfg, in_channel, flag):
        super().__init__()
        self.args = cfg
        _, self.channels = get_model_shape(cfg.arch)
        self.flag = flag
        self.sequential = nn.ModuleList()
        for i in range(len(self.channels)):
            if i == 0:
                convLayer = nn.Conv2d(in_channel, self.channels[i], 3, 2, 1)
            else:
                convLayer = nn.Conv2d(self.channels[i-1], self.channels[i], 3, 2, 1)
            # if self.flag == "si":
            #     self.acti = nn.Tanh()
            # else:
            #     self.acti = nn.Sigmoid()
            self.sequential.append(convLayer)


    def forward(self, x):
        outs = []
        for i in range(len(self.channels)):
            x = self.sequential[i](x)
            outs.append(x)

        # if self.flag == "si":
        #     mask_copy = mask.clone()
        #     ratios = [self.args.set_number]*len(self.channels)
        #     for i in range(len(self.channels)):
        #         factor = np.power(0.5, i+1)
        #         mask_temp = F.interpolate(mask_copy, scale_factor=factor, mode='nearest')
        #         # mask_bool = mask_temp > 0
        #         outs[i] = (outs[i] * (1 - mask_temp).float()) + mask_temp*ratios[i]

        return outs

class Aggregation(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.args = cfg
        _, self.channels = get_model_shape(cfg.arch)
        self.sequential1 = nn.ModuleList()
        self.sequential2 = nn.ModuleList()
        # self.acti = nn.ModuleList()
        for i in range(len(self.channels)):
            in_channels = self.channels[i]
            reduction = self.args.reduction
            convLayer1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0),
                nn.LayerNorm((in_channels//reduction, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, bias=True, padding=0)
            )
            convLayer2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0),
                nn.LayerNorm((in_channels//reduction, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, bias=True, padding=0)
            )
            # gate_activation = nn.Sigmoid()
            self.sequential1.append(convLayer1)
            self.sequential2.append(convLayer2)
            # self.acti.append(gate_activation)

    def forward(self, x, f_h, f_si):
        outs_fh = []
        outs_fsi = []

        for i in range(len(self.channels)):
            attn_f_h = self.sequential1[i](f_h[i])
            attn_f_si = self.sequential2[i](f_si[i])
            attn_seg = F.sigmoid(attn_f_h + (1-attn_f_si)) 
            attn_rem = F.sigmoid(attn_f_si + (1-attn_f_h))
            res_seg = x[i] + x[i] * attn_seg
            res_rem = x[i] + x[i] * attn_rem
            
            outs_fh.append(res_seg)
            outs_fsi.append(res_rem)
        
        return outs_fh, outs_fsi
            

