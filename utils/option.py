import argparse

parser = argparse.ArgumentParser(description='Image Inpainting')

# data specifications 
parser.add_argument('--dir_image', type=str, default='',
                    help='image dataset directory')
parser.add_argument('--dir_test', type=str, default='',
                    help='image dataset directory')
parser.add_argument('--image_size', type=int, default=200,
                    help='image size used during training')
parser.add_argument('--mask_type', type=str, default='0328_',
                    help='mask used during training')
parser.add_argument('--threshold', type=int, default=10,
                    help='image size used during training')

# model specifications 
parser.add_argument('--model', type=str, default='network',
                    help='model name')
parser.add_argument('--gan_type', type=str, default='nsgan',
                    help='discriminator types')

parser.add_argument('--reduction',  type=int, default=8, 
                    help='the number of SE Module reduction')

# hardware specifications 
parser.add_argument('--seed', type=int, default=2021,
                    help='random seed')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers used in data loader')

# optimization specifications 
parser.add_argument('--lrg', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lrd', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 in optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 in optimier')

# loss specifications 
parser.add_argument('--rec_loss', type=str, default='1*L1+250*Style_clip+0.1*Perceptual_clip',
                    help='losses for reconstruction')
parser.add_argument('--adv_weight', type=float, default=0.01,
                    help='loss weight for adversarial loss')

# training specifications 
parser.add_argument('--iterations', type=int, default=1e5,
                    help='the number of iterations for training') 
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size in each mini-batch')
parser.add_argument('--batch_size_test', type=int, default=1,
                    help='batch size in each mini-batch')
parser.add_argument('--port', type=int, default=22334,
                    help='tcp port for distributed training')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous iteration')

parser.add_argument('--arch', type=str, default='0', help='Backbone Architecture')
parser.add_argument('--out_channel', type=int, default=3, help='total out channel')
parser.add_argument('--pixelShuffleRatio', type=int, default=2, help='the ratio of pixel shuffle')


# log specifications 
parser.add_argument('--print_every', type=int, default=10,
                    help='frequency for updating progress bar')
parser.add_argument('--save_every', type=int, default=2000,
                    help='frequency for saving models')  # 1e4
parser.add_argument('--save_dir', type=str, default='./experiments',
                    help='directory for saving models and logs')
parser.add_argument('--tensorboard', action='store_true',
                    help='default: false, since it will slow training. use it for debugging')
parser.add_argument('--save_skip', type=int, default=200,
                    help='image size used during training')
# test and demo specifications 
parser.add_argument('--pre_train', type=str, default='',
                    help='path to pretrained models')
parser.add_argument('--outputs', type=str, default='./outputs', 
                    help='path to save results')
parser.add_argument('--thick',  type=int, default=15, 
                    help='the thick of pen for free-form drawing')
parser.add_argument('--painter', default='freeform', choices=('freeform', 'bbox'),
                    help='different painters for demo ')

parser.add_argument('--date', type=str, default='0318')


# ----------------------------------
args = parser.parse_args()
args.iterations = int(args.iterations)

args.rates = list(map(int, list(args.rates.split('+'))))

losses = list(args.rec_loss.split('+'))
args.rec_loss = {}
for l in losses: 
    weight, name = l.split('*')
    args.rec_loss[name] = float(weight)
