# xai_gradcam.py
import os
import random
import traceback
from glob import glob
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

DEBUG_XAI = False  # 필요시 True로 전환

# ============================================================
# Target layer 자동 탐색 (ResNet 계열 우선, 실패 시 마지막 Conv2d)
# ============================================================
def _auto_target_layer(model: nn.Module) -> nn.Module:
    # ResNet 스타일
    try:
        block = model.layer4[-1]
        for name in ["conv3", "bn3", "conv2", "bn2"]:
            if hasattr(block, name):
                return getattr(block, name)
    except Exception:
        pass

    # Multi-tower 모델(예: model.towers[0].layer4[-1]) 대응
    try:
        block = model.towers[0].layer4[-1]
        for name in ["conv3", "bn3", "conv2", "bn2"]:
            if hasattr(block, name):
                return getattr(block, name)
    except Exception:
        pass

    # 마지막 Conv2d fallback
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Grad-CAM target layer를 찾지 못했습니다.")
    return last_conv


# ============================================================
# Dataset 파이프라인과 동일한 추가 변환들 (texture/edge/sharpen)
# ============================================================
def texture_transform(img: Image.Image) -> Image.Image:
    """Sobel magnitude 기반 텍스처 부각"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mag_rgb = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mag_rgb)

def edge_transform(img: Image.Image) -> Image.Image:
    """Canny edge 기반 엣지 부각"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def sharpen_transform(img: Image.Image) -> Image.Image:
    """언샤프 마스크 유사한 샤픈(간단 커널)"""
    arr = np.array(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(sharp)

# (선택) dataset의 custom_resize를 그대로 사용하려면 아래를 활용
rz_dict = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic' : InterpolationMode.BICUBIC,
    'lanczos' : InterpolationMode.LANCZOS,
    'nearest' : InterpolationMode.NEAREST,
}
def custom_resize(img: Image.Image, opt) -> Image.Image:
    """
    dataset 쪽에서 사용하던 custom_resize를 재현.
    opt.rz_interp가 ['bilinear', ...] 형태라면 첫 값을 사용(혹은 네가 원하는 샘플링 로직으로 교체 가능)
    """
    interp = opt.rz_interp[0] if isinstance(getattr(opt, "rz_interp", "bilinear"), (list, tuple)) \
                               else getattr(opt, "rz_interp", "bilinear")
    return TF.resize(img, (opt.loadSize, opt.loadSize), interpolation=rz_dict.get(interp, InterpolationMode.BILINEAR))


# ============================================================
# Dataset의 binary_dataset 규칙을 그대로 따른 Transform 빌더
# ============================================================
def _build_transform_from_opt(
    opt,
    *,
    use_custom_resize: bool = False,
    override_transform_mode: Optional[str] = None
) -> transforms.Compose:
    """
    dataset_folder(binary_dataset)와 동일 규칙:
      - if opt.isTrain: RandomCrop(opt.cropSize)
        else (no_crop? 그대로 : CenterCrop)
      - if opt.isTrain and not opt.no_flip: RandomHorizontalFlip()
      - if (not opt.isTrain) and opt.no_resize: resize 생략
        else (opt.loadSize, opt.loadSize)로 Resize 또는 custom_resize
      - 추가변환(opt.transform_mode): 'texture' | 'edge' | 'sharpen'
      - ToTensor + Normalize(ImageNet)
    """
    # Crop
    if getattr(opt, "isTrain", False):
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif getattr(opt, "no_crop", False):
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    # Flip
    if getattr(opt, "isTrain", False) and not getattr(opt, "no_flip", False):
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)

    # Resize
    if (not getattr(opt, "isTrain", False)) and getattr(opt, "no_resize", False):
        rz_func = transforms.Lambda(lambda img: img)
    else:
        if use_custom_resize and hasattr(opt, "rz_interp"):
            rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        else:
            rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    # 추가 변환(mode)
    mode = override_transform_mode if override_transform_mode is not None else getattr(opt, "transform_mode", None)
    if   mode == 'texture':  additional_transform = texture_transform
    elif mode == 'edge':     additional_transform = edge_transform
    elif mode == 'sharpen':  additional_transform = sharpen_transform
    else:
        raise NotImplementedError(f"Unknown transform_mode: {mode}")

    return transforms.Compose([
        rz_func,
        crop_func,
        flip_func,
        additional_transform,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# ============================================================
# 유틸: 폴더 내 이미지 재귀 수집 / 모델 출력에서 logits 추출
# ============================================================
def _list_images_recursive(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    paths: List[str] = []
    for ext in exts:
        paths += glob(os.path.join(folder, "**", f"*{ext}"), recursive=True)
    return sorted(paths)

def _pick_logits_from_output(out_obj: Union[torch.Tensor, dict, list, tuple]) -> torch.Tensor:
    """
    다양한 모델 출력 포맷에서 logits(또는 점수 텐서)을 꺼내어 [B] 또는 [B,C] 형태로 정리
    """
    if isinstance(out_obj, dict):
        for k in ["logits", "logit", "y", "scores"]:
            if k in out_obj and torch.is_tensor(out_obj[k]):
                t = out_obj[k]
                break
        else:
            for k in ["logits_seq", "scores_seq", "probs_seq"]:
                if k in out_obj and torch.is_tensor(out_obj[k]):
                    t = out_obj[k]
                    if t.dim() == 3:  # [B,T,C]
                        B, T, C = t.shape
                        t = t.mean(dim=1)
                        t = t.squeeze(1) if C == 1 else t[:, 0]
                    elif t.dim() == 2:  # [B,T]
                        t = t.mean(dim=1)
                    return t
            tv = [v for v in out_obj.values() if torch.is_tensor(v)]
            if len(tv) == 1:
                t = tv[0]
            else:
                raise RuntimeError("Cannot find logits in dict output.")
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1)
            t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if isinstance(out_obj, (tuple, list)):
        t = None
        for item in out_obj:
            if torch.is_tensor(item):
                t = item
                break
            if isinstance(item, dict):
                return _pick_logits_from_output(item)
        if t is None:
            raise RuntimeError("Cannot find tensor in tuple/list output.")
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


# ============================================================
# 파일 저장 유틸
# ============================================================
@torch.no_grad()
def _save_image(path: str, arr_bgr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, arr_bgr)


# ============================================================
# 메인: 폴더 단위로 CAM 저장
# ============================================================
def save_cams_for_folder(
    model: nn.Module,
    folder: str,
    out_dir: str,
    # 호환 보존(무시해도 됨): dataset 규칙을 쓰므로 opt 기반이 우선
    no_resize: bool = False,
    no_crop: bool = True,
    # 동작 옵션
    max_images: Optional[int] = None,
    class_idx: Optional[int] = None,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    shuffle_seed: Optional[int] = 0,
    # dataset 규칙 동기화를 위한 opt
    opt=None,                        # 외부에서 넘겨주는 옵션 객체 (필수)
    use_custom_resize: bool=False,   # dataset의 custom_resize 경로 그대로 사용
    override_transform_mode: Optional[str]=None,  # opt.transform_mode 임시 덮어쓰기
):
    """
    folder 내 이미지를 dataset과 동일한 전처리 규칙으로 모델에 통과시켜
    Grad-CAM을 계산하고, 원본 이미지 위에 heatmap overlay를 저장.
    """
    assert opt is not None, "save_cams_for_folder: opt를 전달해야 합니다 (opt.transform_mode 등)."

    device = next(model.parameters()).device
    model.eval()

    # dataset과 동일 규칙의 transform
    tfm = _build_transform_from_opt(
        opt,
        use_custom_resize=use_custom_resize,
        override_transform_mode=override_transform_mode
    )

    target_layer = _auto_target_layer(model)

    img_paths = _list_images_recursive(folder)
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
        random.shuffle(img_paths)
    if isinstance(max_images, int) and max_images > 0:
        img_paths = img_paths[:max_images]

    if not img_paths:
        print(f"[xai] no images found under {folder}")
        return
    os.makedirs(out_dir, exist_ok=True)

    # forward hook에서 target layer의 출력 텐서를 잡고 retain_grad()
    act_tensor_holder: List[torch.Tensor] = []

    def forward_hook(_module, _inp, out):
        if isinstance(out, torch.Tensor):
            out.retain_grad()
            act_tensor_holder.clear()
            act_tensor_holder.append(out)
        else:
            raise RuntimeError("Target layer output is not a Tensor.")

    h_fwd = target_layer.register_forward_hook(forward_hook)

    try:
        for p in img_paths:
            try:
                if DEBUG_XAI:
                    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
                    os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")

                img = Image.open(p).convert("RGB")

                # ⬇ dataset과 동일 파이프라인
                x = tfm(img).unsqueeze(0).to(device, non_blocking=True)

                with torch.enable_grad():
                    x = x.requires_grad_(True)
                    out = model(x)
                    raw = _pick_logits_from_output(out)  # [B] or [B,C]
                    if raw.dim() == 1:
                        score = raw[0]
                    else:
                        cidx = class_idx if (class_idx is not None and 0 <= class_idx < raw.size(1)) \
                               else int(torch.argmax(raw[0]).item())
                        score = raw[0, cidx]

                    model.zero_grad(set_to_none=True)
                    score.backward(retain_graph=False)

                    if not act_tensor_holder:
                        raise RuntimeError("Activation tensor not captured.")

                    act  = act_tensor_holder[0]   # [B,C,h,w]
                    grad = act.grad               # [B,C,h,w]
                    if grad is None:
                        raise RuntimeError("Activation grad is None. (check retain_grad)")

                    # shape 정렬
                    if act.shape[1] != grad.shape[1]:
                        C = min(act.shape[1], grad.shape[1])
                        act  = act[:, :C, ...]
                        grad = grad[:, :C, ...]
                    if act.shape[-2:] != grad.shape[-2:]:
                        if DEBUG_XAI:
                            print(f"[xai][{os.path.basename(p)}] resizing grad {tuple(grad.shape[-2:])} -> {tuple(act.shape[-2:])}")
                        grad = F.interpolate(grad, size=act.shape[-2:], mode="nearest")

                    # Grad-CAM
                    weights = grad.mean(dim=(2, 3), keepdim=True)   # [B,C,1,1]
                    cam = (weights * act).sum(dim=1, keepdim=True)  # [B,1,h,w]
                    cam = F.relu(cam)

                # 원본 해상도로 업샘플
                H, W = img.size[1], img.size[0]  # PIL: (W,H)
                cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
                cam_up = cam_up.detach().cpu().numpy()

                cmin, cmax = float(cam_up.min()), float(cam_up.max())
                if cmax - cmin < 1e-8:
                    print(f"[xai] flat CAM skipped: {os.path.basename(p)}")
                    continue
                cam_up = (cam_up - cmin) / (cmax - cmin + 1e-8)

                # overlay (원본 위에 오버레이; 전처리 이미지를 쓰고 싶다면 img → 전처리 결과로 교체 가능)
                heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), colormap)
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                if heatmap.shape[:2] != img_cv.shape[:2]:
                    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                overlay = np.uint8(alpha * heatmap + (1 - alpha) * img_cv)

                out_path = os.path.join(out_dir, os.path.basename(p))
                _save_image(out_path, overlay)

            except Exception as e:
                print(f"[xai] 실패({os.path.basename(p)}): {e}")
                traceback.print_exc()
                if DEBUG_XAI:
                    raise
                continue
    finally:
        h_fwd.remove()

    print(f"[xai] CAMs saved to {out_dir}")