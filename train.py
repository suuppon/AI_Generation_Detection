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
<<<<<<< HEAD
import validate
=======
from validate import validate
>>>>>>> b43a8d5702cf4e6c97e45fff026e7860d067166a


def seed_torch(seed: int = 1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def main():
    opt = TrainOptions().parse()
    seed_torch(100)

    # ===== 기본 세팅 보정 =====
    opt.isTrain = True
    if not getattr(opt, "proc_mode", None):
        opt.proc_mode = "dict"  # dict 파이프 권장
    if not getattr(opt, "features", None):
        opt.features = ["edge", "texture"]  # 최소 1개 이상 필요

    # 로그 파일
    Logger(os.path.join(opt.checkpoints_dir, opt.name, "log.log"))
    print("  ".join(list(sys.argv)))
    print(f"[INFO] fusion=ON (built-in at MultiTower), features={opt.features}")

    # ===== DataLoader =====
    data_loader = create_dataloader(opt)

    # ===== SummaryWriter =====
    tb_dir = os.path.join(opt.checkpoints_dir, opt.name, "train")
    os.makedirs(tb_dir, exist_ok=True)
    train_writer = SummaryWriter(tb_dir)

    # ===== Model =====
    model = Trainer(opt)
    model.train()
    print(f"cwd: {os.getcwd()}")

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        num_steps = 0

        pbar = tqdm(
            data_loader,
            desc=f"Epoch {epoch+1}/{opt.epochs}",
            dynamic_ncols=True,
            leave=False,
        )

        for i, data in enumerate(pbar):
            model.total_steps += 1

            # data: (views, labels)
            #   - dict 모드: {'edge':(B,F,C,H,W), 'texture':(...), ...}, labels
            #   - list/tuple 모드: [(B,F,C,H,W), ...], labels
            model.set_input(data)
            model.optimize_parameters()

            # 통계/표시
            loss_val = float(model.loss)
            running_loss += loss_val
            num_steps += 1
            lr_now = model.optimizer.param_groups[0]["lr"]

            pbar.set_postfix({"loss": f"{loss_val:.4f}", "lr": f"{lr_now:.2e}", "step": model.total_steps})

            if model.total_steps % opt.loss_freq == 0:
                print(
                    time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                    f"Train loss: {loss_val:.6f} at step: {model.total_steps} lr {lr_now:g}",
                )
                train_writer.add_scalar("loss", loss_val, model.total_steps)
                train_writer.add_scalar("lr", float(lr_now), model.total_steps)

        # Epoch 통계
        if num_steps > 0:
            avg_loss = running_loss / num_steps
            print(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                f"[Epoch {epoch+1}/{opt.epochs}] avg_loss: {avg_loss:.6f}, steps: {model.total_steps}",
            )
            train_writer.add_scalar("epoch_avg_loss", float(avg_loss), epoch + 1)

        # 주기적 LR decay
        if epoch % getattr(opt, "delr_freq", 1) == 0 and epoch != 0:
            print(
                time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                f"changing lr at the end of epoch {epoch}, iters {model.total_steps}",
            )
            model.adjust_learning_rate()

        if epoch % opt.val_epoch == 0 and epoch != 0:
            model.eval()

            cc, ap, r_acc, f_acc, y_true, y_pred = validate(model ,opt)
            
            model.train()



    # 마지막 저장
    model.eval()
    model.save_networks("last")
    print("[DONE] training finished.")


if __name__ == "__main__":
    main()