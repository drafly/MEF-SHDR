import os
import importlib
import numpy as np
from PIL import Image
from glob import glob
import time
import torch
from torchvision.transforms import ToTensor
import os 
from utils.option import args 
import cv2
from data import create_loader, create_loader_test

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

def postprocess_mask(mask):
        mask = mask * 255
        mask = torch.squeeze(mask)
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        return mask

def main_worker(args):
    
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location='cuda'))
    model.eval()
    os.makedirs(os.path.join(args.outputs,"pred"), exist_ok=True)
    os.makedirs(os.path.join(args.outputs,"mask"), exist_ok=True)

    # prepare dataset
    test_loader = create_loader_test(args)
    for i, data in enumerate(test_loader):
        image, gt, highlight_mask, HSV_img, filename = data
        image, gt, highlight_mask, HSV_img = image.cuda(), gt.cuda(), highlight_mask.cuda(), HSV_img.cuda()

        with torch.no_grad():
            seg_map, pred_img = model(image, HSV_img)

        filename = filename[0].split('_')[0]

        imgs = np.uint8(np.array(((pred_img+1)/2)[0].detach().permute(1, 2, 0).cpu())*255)
        imgs = Image.fromarray(imgs)
        imgs.save(os.path.join(args.outputs,"pred", filename+'.png'))
        seg_map = postprocess_mask(seg_map)
        seg_map.save(os.path.join(args.outputs, "mask", filename+'.png'))
        

if __name__ == '__main__':

    main_worker(args)
