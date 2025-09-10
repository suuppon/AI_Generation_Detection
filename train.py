import os
import sys
import time
import torch
import torch.nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger
from tqdm import tqdm
from torchvision import transforms as T

SAVE_COUNT = 0
MAX_SAVE = 20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denorm_imgnet(t):
    # t: [C,H,W], normalized → [0,1]로 복원
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(-1,1,1)
    x = t*std + mean
    return x.clamp(0,1)

to_pil = T.ToPILImage()

def save_side_by_side(tensor_img, orig_path, mode="texture"):
    """tensor_img: normalized tensor [C,H,W], orig_path: 파일 경로"""
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return
    try:
        # 원본 로드
        orig = Image.open(orig_path).convert("RGB")
        W, H = orig.size

        # 텐서 denorm → PIL
        den = denorm_imgnet(tensor_img.detach().cpu())
        transformed = to_pil(den).resize((W, H))

        # 합쳐 저장
        combined = Image.new("RGB", (W*2, H))
        combined.paste(orig, (0, 0))
        combined.paste(transformed, (W, 0))

        base = os.path.splitext(os.path.basename(orig_path))[0]
        out_dir = os.path.join("samples", base)
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{mode}_comparison.png")
        combined.save(save_path)
        print(f"✅ 샘플 저장: {save_path}")
        SAVE_COUNT += 1
    except Exception as e:
        print(f"[WARN] sample save failed for {orig_path}: {e}")

import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True

    return val_opt


if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)) )
    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)
    data_loader = create_dataloader(opt)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    model = Trainer(opt)
    if torch.cuda.device_count() > 1:
        print(f"⚡ Using {torch.cuda.device_count()} GPUs with DataParallel")
        model.model = torch.nn.DataParallel(model.model)
    # def testmodel():
    #     print('*'*25);accs = [];aps = []
    #     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    #     for v_id, val in enumerate(vals):
    #         Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
    #         Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
    #         Testopt.no_resize = False
    #         Testopt.no_crop = True
    #         acc, ap, _, _, _, _ = validate(model.model, Testopt)
    #         accs.append(acc);aps.append(ap)
    #         print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    #     print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 
    #     print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
    # model.eval();testmodel();
    model.train()
    print(f'cwd: {os.getcwd()}')
    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{opt.niter}")
        for i, batch in pbar:
            # 🔹 배치 구조 정규화: (img, label, path) 혹은 (img, label)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, labels, paths = batch
            else:
                imgs, labels = batch
                paths = None

            # ✅ 첫 에폭에서만 앞 20개 저장
            if epoch == 0 and paths is not None and SAVE_COUNT < MAX_SAVE:
                # 배치 내에서 남은 만큼만 저장
                for b in range(imgs.size(0)):
                    if SAVE_COUNT >= MAX_SAVE:
                        break
                    save_side_by_side(imgs[b], paths[b], mode=getattr(opt, "transform_mode", "texture"))

            # 모델 입력은 (img,label) 그대로 유지
            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input((imgs, labels))
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                log_str = f"loss: {model.loss:4f} | step: {model.total_steps} | lr: {model.lr:.6f}"
                pbar.set_postfix_str(log_str)
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()), 'changing lr at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.adjust_learning_rate()
            
        if (epoch+1) % 10 == 0:
            # Validation
            model.eval()
            acc, ap = validate(model.model, val_opt)[:2]
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))
            # testmodel()
            model.train()

    # model.eval();testmodel()
    model.save_networks('last')
    
