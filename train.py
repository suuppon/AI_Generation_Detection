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
import random

# ------------------------
# 유틸: 샘플 저장
# ------------------------
SAVE_COUNT = 0
MAX_SAVE = 20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def denorm_imgnet(t):
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(-1,1,1)
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device).view(-1,1,1)
    x = t*std + mean
    return x.clamp(0,1)

to_pil = T.ToPILImage()

def save_side_by_side(tensor_img, orig_path, mode="texture"):
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return
    try:
        orig = Image.open(orig_path).convert("RGB")
        W, H = orig.size
        den = denorm_imgnet(tensor_img.detach().cpu())
        transformed = to_pil(den).resize((W, H))

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

# ------------------------
# 시드 고정
# ------------------------
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

# ------------------------
# 체크포인트 저장 (state_dict만)
# ------------------------
def save_state_dict_only(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    sd = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    path = os.path.join(save_dir, filename)
    torch.save(sd, path)
    print(f"💾 Saved state_dict: {path}")

# ------------------------
# 검증 옵션
# ------------------------
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

# ------------------------
# main
# ------------------------
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

    model.train()
    print(f'cwd: {os.getcwd()}')

    # 체크포인트 디렉토리
    ckpt_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Early Stopping 설정 ----
    patience_limit = int(getattr(opt, "patience", getattr(opt, "earlystop_epoch", 0)))
    best_metric = -float("inf")
    best_epoch  = 0
    stale_count = 0
    stopped_early = False
    VAL_EVERY = 5
    MONITOR = "AP"

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        epoch_iter = 0

        pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}/{opt.niter}")
        for i, batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                imgs, labels, paths = batch
            else:
                imgs, labels = batch
                paths = None

            if epoch == 0 and paths is not None and SAVE_COUNT < MAX_SAVE:
                for b in range(imgs.size(0)):
                    if SAVE_COUNT >= MAX_SAVE:
                        break
                    save_side_by_side(imgs[b], paths[b], mode=getattr(opt, "transform_mode", "texture"))

            model.total_steps += 1
            epoch_iter += opt.batch_size
            model.set_input((imgs, labels))
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                log_str = f"loss: {model.loss:4f} | step: {model.total_steps} | lr: {model.lr:.6f}"
                pbar.set_postfix_str(log_str)
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  f'changing lr at the end of epoch {epoch}, iters {model.total_steps}')
            model.adjust_learning_rate()
            
        # ---- 5에폭마다 검증 ----
        if (epoch+1) % VAL_EVERY == 0:
            model.eval()
            acc, ap = validate(model.model, val_opt)[:2]
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print(f"(Val @ epoch {epoch+1}) acc: {acc:.6f}; ap: {ap:.6f}")

            # 현재 에폭 저장
            save_state_dict_only(model.model, ckpt_dir, f"model_epoch_{epoch+1}.pth")

            # 모니터 지표: ap
            metric = ap
            if metric > best_metric + 1e-8:
                best_metric = metric
                best_epoch = epoch + 1
                stale_count = 0
                save_state_dict_only(model.model, ckpt_dir, "model_epoch_best.pth")
                print(f"🔥 New best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}")
            else:
                stale_count += 1
                print(f"⏳ No {MONITOR} improvement ({stale_count}/{patience_limit})")
                if patience_limit > 0 and stale_count >= patience_limit:
                    print(f"🛑 Early stopping triggered "
                          f"(no {MONITOR} improvement for {patience_limit} validations).")
                    save_state_dict_only(model.model, ckpt_dir, f"model_epoch_{epoch+1}_earlystop.pth")
                    stopped_early = True
                    break
            model.train()

        if stopped_early:
            break

    # 마지막 저장
    if not stopped_early:
        save_state_dict_only(model.model, ckpt_dir, "model_epoch_last.pth")

    print(f"✅ Training finished. Best {MONITOR}: {best_metric:.6f} @ epoch {best_epoch}")