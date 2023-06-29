import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .common import BaseNetwork
from .TRACER_specular import TRACER_Encoder
from .Seg_Decoder import Decoder
from .Rem_Decoder import RemDecoder
from .encoder_hsi import HSI_Encoder, Aggregation
from .FeatureFusion import FeatureFusion


class InpaintGenerator(nn.Module):
    def __init__(self, args):  # 1046
        super(InpaintGenerator, self).__init__()

        self.encoder = TRACER_Encoder(args)
        self.encoder_hsv = TRACER_Encoder(args)
        self.agg = FeatureFusion(args)
        self.seg_decoder = Decoder(args, 1)
        self.rem_decoder = RemDecoder(args, 3)
        self.padder_size = 2 ** 5

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


    def forward(self, x, hsv):
        B, C, H, W = x.shape
        x = self.check_image_size(x)
        hsv = self.check_image_size(hsv)

        features = self.encoder(x)
        features_hsv = self.encoder_hsv(hsv)
        features, features_hsv = self.agg(features, features_hsv)
        seg_maps = self.seg_decoder(features_hsv)
        rem = self.rem_decoder(features)
    
        return seg_maps[:, :, :H, :W], rem[:, :, :H, :W]


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat

