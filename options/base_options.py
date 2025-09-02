import argparse
import os
import time
import util
import torch
#import models
#import data
 

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='architecture for binary classification')

        # data augmentation
        parser.add_argument('--rz_interp', default='bilinear')
        parser.add_argument('--blur_prob', type=float, default=0)
        parser.add_argument('--blur_sig', default='0.5')
        parser.add_argument('--jpg_prob', type=float, default=0)
        parser.add_argument('--jpg_method', default='cv2')
        parser.add_argument('--jpg_qual', default='75')

        parser.add_argument('--dataroot', default='./dataset/', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--classes', default='', help='image classes to train on')
        parser.add_argument('--class_bal', action='store_true')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        parser.add_argument('--delr_freq', type=int, default=20, help='frequency of changing lr')
        
        parser.add_argument('--features', type=str, default='edge,texture',
                            help='comma-separated features to use: edge,texture,other')
        parser.add_argument('--proc_mode', type=str, default='concat', choices=['concat','dict'],
                            help='how to combine multi-features for model input')
        parser.add_argument('--proc_config', type=str, default='',
                            help='JSON path for FeatureManager configuration (optional)')
        parser.add_argument('--max_videos_per_class', type=int, default=-1,
                            help='limit #videos per class during indexing (-1: no limit)')
        parser.add_argument('--sort_frames_numeric', action='store_true',
                            help='sort frame files numerically if possible')
        parser.add_argument('--pin_memory', action='store_true',
                            help='enable pin_memory=True for DataLoader')

        # === 추가: 시퀀스 샘플링 ===
        parser.add_argument('--frame_max', type=int, default=-1,
                            help='max frames per video (-1: use all)')
        parser.add_argument('--frame_stride', type=int, default=1,
                            help='sample every Nth frame (>=1)')
        parser.add_argument('--frame_start', type=int, default=0,
                            help='start frame index offset (>=0)')

        # === 추가: 재현성/성능 ===
        parser.add_argument('--seed', type=int, default=1337, help='random seed')
        parser.add_argument('--deterministic', action='store_true', help='torch.backends.cudnn.deterministic=True')
        parser.add_argument('--amp', action='store_true', help='enable mixed-precision training (torch.cuda.amp)')
        parser.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation steps')
        parser.add_argument('--grad_clip', type=float, default=0.0, help='clip grad norm if > 0')

        # 나머지 기존 옵션 그대로 ...
        self.initialized = True
        return parser
        

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return opt #parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        opt.name = opt.name + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        # additional
        opt.classes = opt.classes.split(',')
        opt.rz_interp = opt.rz_interp.split(',')
        opt.blur_sig = [float(s) for s in opt.blur_sig.split(',')]
        opt.jpg_method = opt.jpg_method.split(',')
        opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(',')]
        if len(opt.jpg_qual) == 2:
            opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
        elif len(opt.jpg_qual) > 2:
            raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")

        opt.features = [s for s in opt.features.split(',') if s] if isinstance(opt.features, str) else opt.features
        if opt.max_videos_per_class is not None and opt.max_videos_per_class < 0:
            opt.max_videos_per_class = None
        if not opt.proc_config:
            opt.proc_config = None
        if opt.frame_max is not None and opt.frame_max < 0:
            opt.frame_max = None
        if opt.frame_stride < 1:
            opt.frame_stride = 1
        if opt.frame_start < 0:
            opt.frame_start = 0

        # 재현성
        if opt.seed is not None:
            import random
            import numpy as np
            random.seed(opt.seed)
            np.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            if len(opt.gpu_ids) > 0:
                torch.cuda.manual_seed_all(opt.seed)
        if opt.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.opt = opt
        return self.opt
