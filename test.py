import os
import csv
import random
from collections import OrderedDict
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from options.test_options import TestOptions
from networks.resnet import resnet50
from networks.multi_tower import MultiTowerFromCloned
from xai_run import save_cams_for_folder


# ==============================
# Reproducibility
# ==============================
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


seed_torch(100)


# ==============================
# Helpers
# ==============================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _list_images(folder: str):
    paths = []
    for ext in IMG_EXTS:
        paths += glob(os.path.join(folder, f"*{ext}"))
    return sorted(paths)


def _build_transform(no_resize: bool, no_crop: bool):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    ops = []
    if not no_resize:
        ops.append(T.Resize(256))
    if not no_crop:
        ops.append(T.CenterCrop(224))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def _clean_state_dict(sd: dict) -> OrderedDict:
    drop_prefixes = ("fusion_head.", "module.fusion_head.", "ema.", "teacher.")
    new_sd = OrderedDict()
    for k, v in sd.items():
        k2 = k[7:] if k.startswith("module.") else k
        if any(k2.startswith(p) for p in drop_prefixes):
            continue
        new_sd[k2] = v
    return new_sd


# ==============================
# Pick logits safely
# ==============================
def _pick_logits_from_output(out_obj: object) -> torch.Tensor:
    if isinstance(out_obj, dict):
        for k in ["logits", "logit", "y", "scores"]:
            if k in out_obj and torch.is_tensor(out_obj[k]):
                t = out_obj[k]
                break
        else:
            for k in ["logits_seq", "scores_seq", "probs_seq"]:
                if k in out_obj and torch.is_tensor(out_obj[k]):
                    t = out_obj[k]
                    if t.dim() == 3:
                        B, T, C = t.shape
                        t = t.mean(dim=1)
                        t = t.squeeze(1) if C == 1 else t[:, 0]
                    elif t.dim() == 2:
                        t = t.mean(dim=1)
                    return t
            tensor_vals = [v for v in out_obj.values() if torch.is_tensor(v)]
            if len(tensor_vals) == 1:
                t = tensor_vals[0]
            else:
                raise RuntimeError(f"Cannot find logits in dict. keys={list(out_obj.keys())}")
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1)
            t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if isinstance(out_obj, (tuple, list)):
        for item in out_obj:
            if torch.is_tensor(item):
                t = item
                break
            if isinstance(item, dict):
                return _pick_logits_from_output(item)
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1)
            t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if torch.is_tensor(out_obj):
        t = out_obj
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1)
            t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    raise RuntimeError(f"Unsupported output type: {type(out_obj)}")


# ==============================
# Parse options
# ==============================
opt = TestOptions().parse(print_options=False)
print(f"Model_path {opt.model_path}")
print(f"Dataroot   {getattr(opt, 'dataroot', None)}")

if not hasattr(opt, "no_resize"):
    opt.no_resize = False
if not hasattr(opt, "no_crop"):
    opt.no_crop = True

# ==============================
# Build & load model  (combine 인자 제거 버전)
# ==============================
base_model = resnet50(num_classes=1)

# 일부 버전은 num_classes 인자도 없을 수 있으니 안전하게 처리
try:
    model = MultiTowerFromCloned(base_model=base_model, n_towers=3, num_classes=1, main_device=0)
except TypeError:
    # 더 예전 시그니처: (base_model, n_towers=2, ...)
    model = MultiTowerFromCloned(base_model=base_model, n_towers=3, main_device=0)
    if not hasattr(model, "num_classes"):
        model.num_classes = 1

# 체크포인트 로드
device = torch.device("cuda:0")
ckpt = torch.load(opt.model_path, map_location=device)
state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

# 필요 없는 키 정리
from collections import OrderedDict
def _clean_state_dict(sd: dict) -> OrderedDict:
    drop_prefixes = ("fusion_head.", "module.fusion_head.", "ema.", "teacher.")
    new_sd = OrderedDict()
    for k, v in sd.items():
        k2 = k[7:] if k.startswith("module.") else k
        if any(k2.startswith(p) for p in drop_prefixes):
            continue
        new_sd[k2] = v
    return new_sd

state = _clean_state_dict(state)
incompat = model.load_state_dict(state, strict=False)
print("[load] missing keys:", incompat.missing_keys)
print("[load] unexpected keys:", incompat.unexpected_keys)

# 안전 기본값(런타임 속성) — 과거/커스텀 구현 호환용
if not hasattr(model, "num_classes"): model.num_classes = 1
if not hasattr(model, "topk"):        model.topk = 1
if not hasattr(model, "vote"):        model.vote = "mean"
if not hasattr(model, "threshold"):   model.threshold = 0.5
# ← 생성자에 없던 combine은 속성으로 지정
if not hasattr(model, "combine"):     model.combine = "avg_logits"

model.cuda().eval()



# ==============================
# Inference on a single folder
# ==============================
@torch.no_grad()
def infer_folder(model, folder, out_csv, batch_size=32, no_resize=False, no_crop=True):
    device = next(model.parameters()).device
    tfm = _build_transform(no_resize, no_crop)
    paths = _list_images(folder)
    if not paths:
        raise FileNotFoundError(f"이미지가 없습니다: {folder}")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "logit", "prob"])

        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i+batch_size]
            imgs = [tfm(Image.open(p).convert("RGB")) for p in chunk]
            x = torch.stack(imgs, dim=0)
            x.to(device)
            model.to(device)
            out = model(x)
            logits = _pick_logits_from_output(out)
            probs = torch.sigmoid(logits)

            for pth, lo, pr in zip(chunk, logits.tolist(), probs.tolist()):
                w.writerow([os.path.basename(pth), f"{lo:.6f}", f"{pr:.6f}"])


# ==============================
# 실행부
# ==============================
src = opt.dataroot
if not os.path.isdir(src):
    raise FileNotFoundError(f"--dataroot 는 '이미지 폴더'여야 합니다: {src}")

leaf = src
print(f"[infer] single folder: {leaf}")

leaf_name = os.path.basename(os.path.normpath(leaf))
os.makedirs("preds", exist_ok=True)
out_csv = os.path.join("preds", f"{leaf_name}.csv")
infer_folder(model, leaf, out_csv, batch_size=32,
             no_resize=opt.no_resize, no_crop=opt.no_crop)
print(f"[infer] saved CSV: {out_csv}")

out_cam_dir = os.path.join("cams", leaf_name)
save_cams_for_folder(
    model=model,
    folder=leaf,
    out_dir=out_cam_dir,
    no_resize=opt.no_resize,
    no_crop=opt.no_crop,
    max_images=999999,
)
print(f"[xai] saved CAMs: {out_cam_dir}")

import sys
sys.exit(0)
