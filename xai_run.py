# import os
# from glob import glob
# import torch
# import torch.nn as nn
# from torchvision import transforms as T
# from PIL import Image
# import numpy as np
# import cv2
# from torchcam.methods import GradCAM


# def save_cams_for_folder(
#     model,
#     folder,
#     out_dir,
#     no_resize=False,
#     no_crop=True,
#     max_images=16,
# ):
#     """폴더 내 이미지들에 대해 Grad-CAM 저장 (ResNet/MultiTower 모두 지원)."""

#     # --- target_layer 자동 탐색 ---
#     try:
#         target_layer = model.layer4[-1].conv3
#     except AttributeError:
#         try:
#             target_layer = model.towers[0].layer4[-1].conv3
#         except Exception:
#             last_conv = None
#             for m in model.modules():
#                 if isinstance(m, nn.Conv2d):
#                     last_conv = m
#             if last_conv is None:
#                 raise RuntimeError("Grad-CAM target layer를 찾지 못했습니다.")
#             target_layer = last_conv

#     # --- transform ---
#     IMAGENET_MEAN = [0.485, 0.456, 0.406]
#     IMAGENET_STD = [0.229, 0.224, 0.225]
#     ops = []
#     if not no_resize:
#         ops.append(T.Resize(256))
#     if not no_crop:
#         ops.append(T.CenterCrop(224))
#     ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
#     tfm = T.Compose(ops)

#     # --- 이미지 로드 ---
#     exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
#     img_paths = []
#     for ext in exts:
#         img_paths += glob(os.path.join(folder, f"*{ext}"))
#     img_paths = sorted(img_paths)[:max_images]
#     if not img_paths:
#         print(f"[xai] no images in {folder}")
#         return
#     os.makedirs(out_dir, exist_ok=True)

#     # --- Grad-CAM ---
#     cam_extractor = GradCAM(model, target_layer=target_layer)
#     device = next(model.parameters()).device
#     model.eval()

#     for img_path in img_paths:
#         img = Image.open(img_path).convert("RGB")
#         x = tfm(img).unsqueeze(0).to(device)

#         scores = model(x)
#         cams = cam_extractor(class_idx=0, scores=scores)
#         cam = cams[0].cpu().numpy()

#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#         cam = cv2.resize(cam, img.size)

#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#         img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         overlay = np.uint8(0.5 * heatmap + 0.5 * img_cv)

#         out_path = os.path.join(out_dir, os.path.basename(img_path))
#         cv2.imwrite(out_path, overlay)

#     print(f"[xai] CAMs saved to {out_dir}")

import os
from glob import glob
from typing import List

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2


def _auto_target_layer(model: nn.Module) -> nn.Module:
    """ResNet/MultiTower에서 Grad-CAM 타깃 레이어를 자동 탐색."""
    # 1) 일반 ResNet
    try:
        return model.layer4[-1].conv3
    except Exception:
        pass

    # 2) MultiTower 형태
    try:
        return model.towers[0].layer4[-1].conv3
    except Exception:
        pass

    # 3) Fallback: 모델 내 마지막 Conv2d
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Grad-CAM target layer를 찾지 못했습니다.")
    return last_conv


def _build_transform(no_resize: bool, no_crop: bool):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    ops: List[nn.Module] = []
    if not no_resize:
        ops.append(T.Resize(256))
    if not no_crop:
        ops.append(T.CenterCrop(224))
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def _list_images(folder: str):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
    paths = []
    for ext in exts:
        paths += glob(os.path.join(folder, f"*{ext}"))
    return sorted(paths)


def _pick_logits_from_output(out_obj: object) -> torch.Tensor:
    """
    모델 출력이 dict/tuple/list/tensor 무엇이든 [B] 로짓으로 정규화.
    우선 키: logits, logit, y, scores, (logits_seq/scores_seq/probs_seq는 T-mean)
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
                        t = t.mean(dim=1)            # [B,C]
                        t = t.squeeze(1) if C == 1 else t[:, 0]
                    elif t.dim() == 2:               # [B,T]
                        t = t.mean(dim=1)            # [B]
                    return t
            tv = [v for v in out_obj.values() if torch.is_tensor(v)]
            if len(tv) == 1:
                t = tv[0]
            else:
                raise RuntimeError("Cannot find logits in dict output.")
        # 텐서 보정
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
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
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    if torch.is_tensor(out_obj):
        t = out_obj
        if t.dim() == 3:
            B, T, C = t.shape
            t = t.mean(dim=1); t = t.squeeze(1) if C == 1 else t[:, 0]
        elif t.dim() == 2:
            t = t.squeeze(1) if t.size(1) == 1 else t[:, 0]
        return t

    raise RuntimeError(f"Unsupported output type: {type(out_obj)}")


def save_cams_for_folder(
    model: nn.Module,
    folder: str,
    out_dir: str,
    no_resize: bool = False,
    no_crop: bool = True,
    max_images: int = 16,
):
    """
    폴더 내 이미지들에 대해 Grad-CAM overlay를 저장합니다.
    (pure PyTorch hooks; torchcam 필요 없음)
    - class_idx=0 기준으로 CAM을 계산합니다.
    """
    device = next(model.parameters()).device
    model.eval()

    target_layer = _auto_target_layer(model)
    tfm = _build_transform(no_resize, no_crop)

    img_paths = _list_images(folder)[:max_images]
    if not img_paths:
        print(f"[xai] no images in {folder}")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Hooks
    activations = []
    gradients = []

    def f_hook(module, inp, out):
        activations.append(out.detach())

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_backward_hook(b_hook)

    try:
        for p in img_paths:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)

            activations.clear()
            gradients.clear()

            # forward
            out = model(x)
            logits = _pick_logits_from_output(out)  # [B]
            # class_idx = 0 가정 (fake 클래스)
            score = logits.view(-1)[0]

            # backward
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            # Grad-CAM
            act = activations[0]          # [B,C,H,W]
            grad = gradients[0]           # [B,C,H,W]
            weights = grad.mean(dim=(2, 3), keepdim=True)      # [B,C,1,1]
            cam = (weights * act).sum(dim=1, keepdim=True)     # [B,1,H,W]
            cam = torch.relu(cam)[0, 0].cpu().numpy()          # [H,W]

            # normalize & resize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = cv2.resize(cam, img.size)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            overlay = np.uint8(0.5 * heatmap + 0.5 * img_cv)

            out_path = os.path.join(out_dir, os.path.basename(p))
            cv2.imwrite(out_path, overlay)
    finally:
        h1.remove()
        h2.remove()

    print(f"[xai] CAMs saved to {out_dir}")
