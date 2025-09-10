# test.py
import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random

# 🔥 Grad-CAM 유틸
from xai_gradcam import save_cams_for_folder

# --------------------------
# 공통 설정
# --------------------------
BASE_DATAROOT = "/data/Deepfake_train_val/test"   # 요청한 dataroot
# 이미지 전처리 기본값 (필요하면 바꿔도 됨)
NO_RESIZE_DEFAULT = False               # 다양한 해상도면 False 권장
NO_CROP_DEFAULT   = True

# Grad-CAM 설정
DO_GRADCAM   = True
GRADCAM_MAX  = 16
GRADCAM_DIR  = "./runs/gradcam"

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

# --------------------------
# 옵션 파싱 & 모델 로딩
# --------------------------
opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

model = resnet50(num_classes=1)

# --- robust state_dict loader ---
ckpt = torch.load(opt.model_path, map_location='cpu')

# 1) 체크포인트 안에서 state_dict 추출 (형태 가리지 않기)
if isinstance(ckpt, dict):
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
    elif 'model' in ckpt and isinstance(ckpt['model'], dict):
        sd = ckpt['model']
    else:
        # 이미 state_dict 그 자체로 저장된 경우
        sd = ckpt
else:
    sd = ckpt

# 2) DataParallel 프리픽스 제거
new_sd = {}
for k, v in sd.items():
    nk = k
    if nk.startswith('module.'):
        nk = nk[len('module.'):]
    new_sd[nk] = v

# (필요시) 마지막 레이어 이름 정합성 체크 (fc vs fc1 등)
# 현재 모델은 fc1을 사용하므로, 만약 키가 fc.* 형태라면 fc1.*로 매핑
if any(k.startswith('fc.') for k in new_sd.keys()) and \
   not any(k.startswith('fc1.') for k in new_sd.keys()):
    tmp = {}
    for k, v in new_sd.items():
        if k.startswith('fc.'):
            tmp['fc1.' + k[len('fc.'):]] = v
        else:
            tmp[k] = v
    new_sd = tmp

# 3) 로드 (엄격 모드로 시도 → 실패 시 느슨하게)
missing, unexpected = model.load_state_dict(new_sd, strict=False)
if missing or unexpected:
    print("[load_state_dict] warning")
    if missing:
        print("  missing:", missing)
    if unexpected:
        print("  unexpected:", unexpected)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# --------------------------
# 테스트 루프
# --------------------------
if not os.path.isdir(BASE_DATAROOT):
    raise FileNotFoundError(f"[test] dataroot not found: {BASE_DATAROOT}")

subsets = sorted([d for d in os.listdir(BASE_DATAROOT)
                  if os.path.isdir(os.path.join(BASE_DATAROOT, d))])

printSet(os.path.basename(BASE_DATAROOT))
accs, aps = [], []
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

for v_id, val in enumerate(subsets):
    opt.dataroot = os.path.join(BASE_DATAROOT, val)
    opt.classes  = ''  # 단일 이진 분류 가정
    opt.no_resize = NO_RESIZE_DEFAULT
    opt.no_crop   = NO_CROP_DEFAULT

    # 평가
    acc, ap, _, _, _, _ = validate(model, opt)
    accs.append(acc); aps.append(ap)
    print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc * 100, ap * 100))

    # Grad-CAM 저장
    if DO_GRADCAM:
        out_dir = os.path.join(GRADCAM_DIR, os.path.basename(BASE_DATAROOT), val)
        try:
            save_cams_for_folder(
                model=model,
                folder=opt.dataroot,     # 서브셋 루트에서 재귀적으로 이미지 수집
                out_dir=out_dir,
                no_resize=opt.no_resize,
                no_crop=opt.no_crop,
                max_images=GRADCAM_MAX,
            )
        except Exception as e:
            print(f"[xai] Grad-CAM 실패 ({val}): {e}")

print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(
    len(subsets), 'Mean', np.array(accs).mean() * 100, np.array(aps).mean() * 100))
print('*' * 25)