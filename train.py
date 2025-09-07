# # train.py (main)
# import os
# import sys
# import time
# import torch
# import argparse
# from PIL import Image
# from tensorboardX import SummaryWriter
# import numpy as np

# from validate import validate
# from data import create_dataloader
# from utils.trainer import Trainer          # ★ 경로 수정
# from options.train_options import TrainOptions
# from options.test_options import TestOptions
# from util import Logger

# import random

# def seed_torch(seed=1029):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # multi-GPU
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False

# # test config
# vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
# multiclass = [1, 1, 1, 0, 1, 0, 0, 0]

# def get_val_opt(base_opt):
#     # base_opt를 카피해서 val 세팅만 변경
#     val_opt = TrainOptions().parse(print_options=False)
#     # 학습 파이프 옵션이 val에 영향을 주지 않도록 최소만 세팅
#     val_opt.dataroot = f'{base_opt.dataroot_root}/{base_opt.val_split}/'
#     val_opt.isTrain = False
#     val_opt.no_resize = False
#     val_opt.no_crop = False
#     val_opt.serial_batches = True
#     return val_opt

# if __name__ == '__main__':
#     opt = TrainOptions().parse()
#     seed_torch(100)

#     # 학습/검증 경로 정리
#     opt.dataroot_root = opt.dataroot.rstrip('/')

#     Testdataroot = os.path.join(opt.dataroot_root, 'test')
#     opt.dataroot = f'{opt.dataroot_root}/{opt.train_split}/'

#     Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
#     print('  '.join(list(sys.argv)) )

#     # dict 배치 파이프를 기본값으로 권장 (옵션에서 변경했다면 그대로 사용)
#     if not hasattr(opt, 'proc_mode') or not opt.proc_mode:
#         opt.proc_mode = 'dict'
#     if not hasattr(opt, 'features') or not opt.features:
#         # 최소 1개 feature 필요. 기존 전처리와 가장 가까운 edge+texture 권장
#         opt.features = ['edge', 'texture']

#     # DataLoader
#     data_loader = create_dataloader(opt)

#     # SummaryWriter
#     train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
#     val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

#     # Trainer
#     model = Trainer(opt)

#     # ---- Validation Option (val split) ----
#     val_opt = get_val_opt(opt)

#     # ---- 테스트 셋 루틴 ----
#     Testopt = TestOptions().parse(print_options=False)

#     def testmodel():
#         print('*'*25)
#         accs = []; aps = []
#         print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
#         for v_id, val in enumerate(vals):
#             Testopt.dataroot = f'{Testdataroot}/{val}'
#             Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
#             Testopt.no_resize = False
#             Testopt.no_crop = True
#             # validate가 4D 입력(B,C,H,W) 기반이면, MultiTowerFromCloned도 4D를 지원
#             acc, ap, _, _, _, _ = validate(model.model, Testopt)
#             accs.append(acc); aps.append(ap)
#             print(f"({v_id} {val:10}) acc: {acc*100:.1f}; ap: {ap*100:.1f}")
#         print(f"({v_id+1} {'Mean':10}) acc: {np.mean(accs)*100:.1f}; ap: {np.mean(aps)*100:.1f}")
#         print('*'*25)
#         print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

#     # 사전 점검용 테스트 1회
#     model.eval(); testmodel(); model.train()

#     print(f'cwd: {os.getcwd()}')
#     for epoch in range(opt.niter):
#         epoch_start_time = time.time()
#         epoch_iter = 0

#         for i, data in enumerate(data_loader):
#             model.total_steps += 1
#             epoch_iter += opt.batch_size

#             # data: (x_dict_or_list, labels)  — dataloader가 dict 모드면 dict가 들어옴
#             model.set_input(data)
#             model.optimize_parameters()

#             if model.total_steps % opt.loss_freq == 0:
#                 lr_now = model.optimizer.param_groups[0]['lr']
#                 print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
#                       f"Train loss: {model.loss:.6f} at step: {model.total_steps} lr {lr_now:g}")
#                 train_writer.add_scalar('loss', float(model.loss), model.total_steps)
#                 train_writer.add_scalar('lr', float(lr_now), model.total_steps)

#         # 스케줄링
#         if epoch % opt.delr_freq == 0 and epoch != 0:
#             print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
#                   f'changing lr at the end of epoch {epoch}, iters {model.total_steps}')
#             model.adjust_learning_rate()

#         # Validation (val split)
#         model.eval()
#         acc, ap = validate(model.model, val_opt)[:2]
#         val_writer.add_scalar('accuracy', float(acc), model.total_steps)
#         val_writer.add_scalar('ap', float(ap), model.total_steps)
#         print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")

#         # 추가 테스트셋 루틴
#         testmodel()
#         model.train()

#     model.eval(); testmodel()
#     model.save_networks('last')

# train.py
import os, sys, time, random
import torch
import numpy as np
from tensorboardX import SummaryWriter
from tqdm.auto import tqdm

from data import create_dataloader
from utils.trainer import Trainer
from options.train_options import TrainOptions
from util import Logger

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

if __name__ == '__main__':
    opt = TrainOptions().parse()
    seed_torch(100)

    # dataroot는 images 루트로 바로 지정 (예: ./images)
    # ex) python train.py --dataroot ./images
    opt.isTrain = True
    if not hasattr(opt, 'proc_mode') or not opt.proc_mode:
        opt.proc_mode = 'dict'
    if not hasattr(opt, 'features') or not opt.features:
        opt.features = ['edge','texture']

    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))

    # DataLoader (images/{real,fake})
    # TODO : CHECK
    data_loader = create_dataloader(opt)

    # Writers
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))

    # Model
    # TODO : CHECK
    model = Trainer(opt)
    model.train()
    print(f'cwd: {os.getcwd()}')

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        running_loss = 0.0
        num_steps = 0

        # tqdm 진행바: 에폭/총에폭, 현재 LR, 최근 loss 표시
        pbar = tqdm(
            data_loader,
            desc=f"Epoch {epoch+1}/{opt.niter}",
            dynamic_ncols=True,
            leave=False
        )

        for i, data in enumerate(pbar):
            model.total_steps += 1
            #data ({k1:-,k2:-,k3:-},labels)
            # TODO : CHECK
            model.set_input(data)
            model.optimize_parameters()

            # 통계/표시용
            loss_val = float(model.loss)
            running_loss += loss_val
            num_steps += 1
            lr_now = model.optimizer.param_groups[0]['lr']

            # 진행바 우측 postfix 업데이트
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{lr_now:.2e}",
                "step": model.total_steps
            })

            # 텐서보드 로깅(기존 주기 유지)
            if model.total_steps % opt.loss_freq == 0:
                print(
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                    f"Train loss: {loss_val:.6f} at step: {model.total_steps} lr {lr_now:g}"
                )
                train_writer.add_scalar('loss', loss_val, model.total_steps)
                train_writer.add_scalar('lr', float(lr_now), model.total_steps)

        # 에폭 끝나고 평균 loss 간단 출력
        if num_steps > 0:
            avg_loss = running_loss / num_steps
            print(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                f"[Epoch {epoch+1}/{opt.niter}] avg_loss: {avg_loss:.6f}, steps: {model.total_steps}"
            )

        # 주기적 LR decay
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                f'changing lr at the end of epoch {epoch}, iters {model.total_steps}'
            )
            model.adjust_learning_rate()

        #정확도 Train

        #정확도 Val
        
    # 마지막 저장
    model.eval()
    model.save_networks('last')