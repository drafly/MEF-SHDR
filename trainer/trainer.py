import os
import importlib
from tqdm import tqdm
from glob import glob
from PIL import Image, ImageFilter
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from data import create_loader, create_loader_test
# from data.dataset import rgb2hsi
from loss import loss as loss_module
from .common import timer, reduce_loss_dict
import torch.nn.functional as F
from loss import clip
from torchvision.transforms import ToTensor
import numpy as np

class Trainer():
    def __init__(self, args):
        self.args = args 
        self.iteration = 0

        # setup data set and data loader
        self.dataloader = create_loader(args)

        # set up losses and metrics
        self.rec_loss_func = {
            key: getattr(loss_module, key)() for key, val in args.rec_loss.items()}
        self.adv_loss = getattr(loss_module, args.gan_type)()

        # Image generator input: [rgb(3) + mask(1)], discriminator input: [rgb(3)]
        net = importlib.import_module('model.'+args.model)
        
        self.netG = net.InpaintGenerator(args).cuda()

        self.optimG = torch.optim.Adam(
            self.netG.parameters(), lr=args.lrg, betas=(args.beta1, args.beta2))

        self.netD = net.Discriminator().cuda()
        self.optimD = torch.optim.Adam(
            self.netD.parameters(), lr=args.lrd, betas=(args.beta1, args.beta2))
        
        self.load()
        if args.distributed:
            self.netG = DDP(self.netG, device_ids= [args.local_rank], output_device=[args.local_rank])
            self.netD = DDP(self.netD, device_ids= [args.local_rank], output_device=[args.local_rank])
        
        if args.tensorboard: 
            self.writer = SummaryWriter(os.path.join(args.save_dir, 'log'))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.CLIP, _ = clip.load("RN50", device) 
            

    def load(self):
        try: 
            gpath = sorted(list(glob(os.path.join(self.args.save_dir, 'G*.pt'))))[-1]
            self.netG.load_state_dict(torch.load(gpath, map_location='cuda'))
            self.iteration = int(os.path.basename(gpath)[1:-3])
            if self.args.global_rank == 0: 
                print(f'[**] Loading generator network from {gpath}')
        except: 
            pass 
        
        try: 
            dpath = sorted(list(glob(os.path.join(self.args.save_dir, 'D*.pt'))))[-1]
            self.netD.load_state_dict(torch.load(dpath, map_location='cuda'))
            if self.args.global_rank == 0: 
                print(f'[**] Loading discriminator network from {dpath}')
        except: 
            pass
        
        try: 
            opath = sorted(list(glob(os.path.join(self.args.save_dir, 'O*.pt'))))[-1]
            data = torch.load(opath, map_location='cuda')
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            if self.args.global_rank == 0: 
                print(f'[**] Loading optimizer from {opath}')
        except: 
            pass


    def save(self, ):
        if self.args.global_rank == 0:
            print(f'\nsaving {self.iteration} model to {self.args.save_dir} ...')
            torch.save(self.netG.state_dict(),
                os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            torch.save(self.netD.state_dict(),
                os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            # torch.save(self.netG.module.state_dict(),
            #            os.path.join(self.args.save_dir, f'G{str(self.iteration).zfill(7)}.pt'))
            # torch.save(self.netD.module.state_dict(),
            #            os.path.join(self.args.save_dir, f'D{str(self.iteration).zfill(7)}.pt'))
            torch.save(
                {'optimG': self.optimG.state_dict(), 'optimD': self.optimD.state_dict()}, 
                os.path.join(self.args.save_dir, f'O{str(self.iteration).zfill(7)}.pt'))

    # def postprocess(self, image):
    #     image = torch.clamp(image, -1., 1.)
    #     image = (image + 1) / 2.0
    #     image = image.permute(1, 2, 0)
    #     image = image.cpu().numpy()
    #     return image
    
    def postprocess_mask(self, mask):
        mask = mask * 255
        mask = torch.squeeze(mask)
        mask = mask.cpu().numpy().astype(np.uint8)
        mask = Image.fromarray(mask)
        return mask
      
    def test_eval(self, ):
        self.netG.eval()
        self.netD.eval()
        os.makedirs(os.path.join(self.args.save_dir, 'test_output', str(self.iteration)), exist_ok=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        psnr, ssim = [], []

        test_loader = create_loader_test(self.args)
        for i, data in enumerate(test_loader):
            image, gt, highlight_mask, HSV_img, filename = data
            image, gt, highlight_mask, HSV_img = image.cuda(), gt.cuda(), highlight_mask.cuda(), HSV_img.cuda()

            with torch.no_grad():
                seg_map, pred_img = self.netG(image, HSV_img)

            filename = filename[0].split('_')[0]
            pred_img = torch.clamp(pred_img, -1., 1.)
            imgs = np.uint8(np.array(((pred_img+1)/2)[0].detach().permute(1, 2, 0).cpu())*255)
            gt = np.uint8(np.array(((gt+1)/2)[0].detach().permute(1, 2, 0).cpu())*255)

            psnr.append(peak_signal_noise_ratio(gt, imgs))
            ssim.append(structural_similarity(gt, imgs, channel_axis=2))

            if i % self.args.save_skip == 0:
                imgs = Image.fromarray(imgs)
                imgs.save(os.path.join(self.args.save_dir, 'test_output', str(self.iteration), filename+'.png'))
                seg_map = self.postprocess_mask(seg_map)
                seg_map.save(os.path.join(self.args.save_dir, 'test_output', str(self.iteration), filename+'_mask.png'))
        
        self.writer.add_scalar('psnr', np.mean(psnr), self.iteration)
        self.writer.add_scalar('ssim', np.mean(ssim), self.iteration)

        self.netG.train()
        self.netD.train()

                
    def train(self):
        pbar = range(self.iteration, self.args.iterations)
        if self.args.global_rank == 0: 
            pbar = tqdm(range(self.iteration, self.args.iterations+1), dynamic_ncols=True, smoothing=0.01)
            timer_data, timer_model = timer(), timer()
        
        for idx in pbar:
            self.iteration += 1

            images, gt_image, highlight, hsv, filename = next(self.dataloader)
            images, gt_image, highlight, hsv = images.cuda(), gt_image.cuda(), highlight.cuda(), hsv.cuda()
            # highlight_copy = highlight.detach()

            if self.args.global_rank == 0: 
                timer_data.hold()
                timer_model.tic()

            # in: [rgb(3) + edge(1)]
            seg_map, pred_img = self.netG(images, hsv)

            # reconstruction losses 
            losses = {}
            for name, weight in self.args.rec_loss.items(): 
                if "clip" in name:
                    x_clip, y_clip = self.CLIP.encode_image(pred_img), self.CLIP.encode_image(gt_image)
                    losses[name] = weight * self.rec_loss_func[name](x_clip, y_clip)
                else:
                    losses[name] = weight * self.rec_loss_func[name](pred_img, gt_image)
            
            # adversarial loss 
            highlight = highlight.type_as(images)
            seg_loss = F.binary_cross_entropy(seg_map, highlight)
            # seg_loss = loss_module.adaptive_pixel_intensity_loss(seg_map, highlight)
            losses["seg_loss"] = seg_loss
            if self.args.gan_type == 'nsgan':
                dis_loss, gen_loss = self.adv_loss(self.netD, pred_img, gt_image)
            # else:
            #     dis_loss, gen_loss = self.adv_loss(self.netD, pred_img, images, masks)
            losses[f"advg"] = gen_loss * self.args.adv_weight
            
            # backforward 
            self.optimG.zero_grad()
            self.optimD.zero_grad()
            sum(losses.values()).backward()
            losses[f"advd"] = dis_loss 

            dis_loss.backward()
            self.optimG.step()
            self.optimD.step()

            if self.args.global_rank == 0:
                timer_model.hold()
                timer_data.tic()

            # logs
            scalar_reduced = reduce_loss_dict(losses, self.args.world_size)
            if self.args.global_rank == 0 and (self.iteration % self.args.print_every == 0): 
                pbar.update(self.args.print_every)
                description = f'mt:{timer_model.release():.1f}s, dt:{timer_data.release():.1f}s, '
                for key, val in losses.items(): 
                    description += f'{key}:{val.item():.3f}, '
                    if self.args.tensorboard: 
                        self.writer.add_scalar(key, val.item(), self.iteration)
                pbar.set_description((description))
            
            if self.args.global_rank == 0 and (self.iteration % self.args.save_every) == 0: 
                self.save()
                if self.args.tensorboard: 
                    # self.writer.add_image('mask', make_grid(masks), self.iteration)
                    self.writer.add_image('highlight_mask', make_grid(highlight), self.iteration)
                    self.writer.add_image('seg_map', make_grid(seg_map), self.iteration)
                    self.writer.add_image('ori', make_grid((gt_image+1.0)/2.0), self.iteration)
                    self.writer.add_image('pred', make_grid((pred_img+1.0)/2.0), self.iteration)
                    # self.writer.add_image('comp', make_grid((comp_img+1.0)/2.0), self.iteration)
                    self.test_eval()


