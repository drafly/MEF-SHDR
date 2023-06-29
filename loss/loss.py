import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .common import VGG19, gaussian_blur



class L1(): 
    def __init__(self,):
        self.calc = torch.nn.L1Loss()
    
    def __call__(self, x, y):
        return self.calc(x, y)


    
class Perceptual_clip(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual_clip, self).__init__()
        
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        
        content_loss = 0.0
        prefix = [0, 1, 2, 3, 4]
        ratio = [2, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x[f'layer_{prefix[i]}'], y[f'layer_{prefix[i]}'])
        return content_loss


class Style_clip(nn.Module):
    def __init__(self):
        super(Style_clip, self).__init__()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G
    
    def __call__(self, x, y):
        
        style_loss = 0.0
        prefix = [1, 2, 3, 4]
        ratio = [2, 3, 4, 5]

        for i in range(4):
            style_loss += self.criterion(
                self.compute_gram(x[f'layer_{prefix[i]}']), self.compute_gram(y[f'layer_{prefix[i]}']))
        return style_loss
    


class nsgan(): 
    def __init__(self, ):
        self.loss_fn = torch.nn.Softplus()
    
    def __call__(self, netD, fake, real):
        fake_detach = fake.detach()
        d_fake = netD(fake_detach)
        d_real = netD(real)
        dis_loss = self.loss_fn(-d_real).mean() + self.loss_fn(d_fake).mean()

        g_fake = netD(fake)
        gen_loss = self.loss_fn(-g_fake).mean()
        
        return dis_loss, gen_loss




