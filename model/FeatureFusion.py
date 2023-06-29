import torch
import torch.nn as nn
import torch.nn.functional as F
from .TRACER_specular import get_model_shape
from .Seg_Decoder import BasicConv2d

class FeatureFusion(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.args = cfg
        _, self.channels = get_model_shape(cfg.arch)
        self.sequential_f = nn.ModuleList()
        self.sequential_h = nn.ModuleList()

        self.conv_f_p = nn.ModuleList()
        self.conv_f_n = nn.ModuleList()
        self.conv_h_p = nn.ModuleList()
        self.conv_h_n = nn.ModuleList()

        self.ratio_f = nn.ParameterDict()
        self.ratio_h = nn.ParameterDict()

        # self.acti = nn.ModuleList()
        for i in range(len(self.channels)-1):
            in_channel = self.channels[i]
            out_channel = self.channels[i+1]
            reduction = self.args.reduction
            seAttn1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, bias=True, padding=0),
                nn.LayerNorm((in_channel//reduction, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, bias=True, padding=0),
                nn.Sigmoid()
            )
    
            seAttn2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, bias=True, padding=0),
                nn.LayerNorm((in_channel//reduction, 1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, bias=True, padding=0),
                nn.Sigmoid()
            )
            self.sequential_f.append(seAttn1)
            self.sequential_h.append(seAttn2)

            convBlock1 = BasicConv2d(in_channel, out_channel, 3, 2, 1)
            convBlock2 = BasicConv2d(in_channel, out_channel, 3, 2, 1)
            convBlock3 = BasicConv2d(in_channel, out_channel, 3, 2, 1)
            convBlock4 = BasicConv2d(in_channel, out_channel, 3, 2, 1)

            self.conv_f_p.append(convBlock1)
            self.conv_f_n.append(convBlock2) 
            self.conv_h_p.append(convBlock3)
            self.conv_h_n.append(convBlock4) 

            alpha = nn.Parameter(torch.ones((2)), requires_grad=True)
            beta = nn.Parameter(torch.ones((2)), requires_grad=True)

            self.ratio_f[str(i)] = alpha
            self.ratio_h[str(i)] = beta  


    # def forward(self, x, y, features_seg=None):
    def forward(self, f, h):
        f_p, f_n = [], []
        h_p, h_n = [], []

        for i in range(len(self.channels)-1):
            attn_f = self.sequential_f[i](f[i])
            attn_h = self.sequential_h[i](h[i])

            f_p.append(self.conv_f_p[i](f[i] * attn_f))
            f_n.append(self.conv_f_n[i](f[i] * (1-attn_f)))

            h_p.append(self.conv_h_p[i](h[i] * attn_h))
            h_n.append(self.conv_h_n[i](h[i] * (1-attn_h)))

        outs_f, outs_h = [], []
        outs_f.append(f[0])
        outs_h.append(h[0])

        for i in range(1, len(self.channels)):
            j = i-1
            out_f = f[i] + self.ratio_f[str(j)][0]*f_p[j] + self.ratio_f[str(j)][1]*h_n[j]
            out_h = h[i] + self.ratio_h[str(j)][0]*h_p[j] + self.ratio_h[str(j)][1]*f_n[j]
            outs_f.append(out_f)
            outs_h.append(out_h)
        return outs_f, outs_h