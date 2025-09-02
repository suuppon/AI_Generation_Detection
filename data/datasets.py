# -*- coding: utf-8 -*-
# multiprocess_dataset.py
import os
from typing import List, Tuple, Optional, Dict, Union
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# === 네가 제공한 파일을 import ===
from utils.feature_manager import FeatureManager

Tensor = torch.Tensor


def _load_image_any(image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    입력이 path/ndarray/torch.Tensor 어떤 것이든 RGB uint8 ndarray로 통일.
    """
    if isinstance(image, str):
        if cv2 is not None:
            arr = cv2.imread(image, cv2.IMREAD_COLOR)  # BGR
            if arr is None:
                raise ValueError(f"Failed to read image: {image}")
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return arr
        else:
            with Image.open(image) as im:
                return np.array(im.convert("RGB"), copy=True)
    elif isinstance(image, np.ndarray):
        return image
    elif torch.is_tensor(image):
        arr = image.detach().cpu().numpy()
        # 채널/타입은 호출부에서 맞춘다고 가정
        return arr
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

# -----------------------------
# Processor Builder
# -----------------------------
class ProcessorBuilder:
    """
    FeatureManager를 감싸서, 사용자가 원하는 feature 리스트만 선택적으로 사용.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        features: Optional[List[str]] = None,  # 예: ["edge","texture","other"]
    ):
        self.manager = FeatureManager(config_path=config_path)
        # 사용 가능한 키 검증
        all_keys = set(self.manager.preprocessors.keys())  # {'texture','edge','other'}
        if features is None:
            self.features = sorted(list(all_keys))
        else:
            unknown = set(features) - all_keys
            if unknown:
                raise ValueError(f"Unknown features: {unknown} (available: {sorted(list(all_keys))})")
            self.features = features

    def process_one(self, image: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, Tensor]:
        """
        단일 이미지에 대해 선택된 features를 모두 적용, {"edge":C,H,W, ...} 반환
        - 이미지는 1회만 로드하여 ndarray로 만든 뒤 모든 feature에 재사용
        """
        return self.manager.preprocess_selected(image, self.features)  # {"edge":..., "texture":..., ...}


# -----------------------------
# Dataset
# -----------------------------
class MultiProcVideoDataset(Dataset):
    """
    폴더 구조:
      root/
        fake/
          vid001/*.png
          ...
        real/
          vidABC/*.png
          ...
    각 비디오를 프레임 순회 -> ProcessorBuilder로 여러 전처리를 적용.
    mode:
      - "concat": features를 C축으로 concat하여 [T, C*, H, W]
      - "dict"  : feature별 [T, C, H, W] 딕셔너리 반환
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        fake_dir: str,
        real_dir: Optional[str] = None,
        processor: Optional[ProcessorBuilder] = None,
        mode: str = "concat",            # "concat" | "dict"
        max_videos_per_class: Optional[int] = None,
        sort_frames_numeric: bool = True,
        # ↓ 추가: 프레임 샘플링 옵션
        frame_max: Optional[int] = None, # 비디오당 최대 프레임 수(-1/None: 제한없음)
        frame_stride: int = 1,          # N프레임 간격 샘플링(최소 1)
        frame_start: int = 0,           # 시작 offset(최소 0)
    ):
        assert mode in ("concat", "dict")
        self.fake_dir = fake_dir
        self.real_dir = real_dir
        self.mode = mode
        self.max_videos_per_class = max_videos_per_class
        self.sort_frames_numeric = sort_frames_numeric

        # 샘플링 옵션
        self.frame_max = frame_max if (frame_max is None or frame_max >= 0) else None
        self.frame_stride = max(1, frame_stride)
        self.frame_start = max(0, frame_start)

        self.processor = processor or ProcessorBuilder(config_path=None, features=None)

        # (vdir, label)
        self.index: List[Tuple[str, int]] = []
        self._index_class(fake_dir, label=0)
        if real_dir is not None:
            self._index_class(real_dir, label=1)

        if not self.index:
            raise RuntimeError("No videos found. Check paths.")

    def _index_class(self, class_dir: str, label: int):
        if not os.path.isdir(class_dir):
            return
        subdirs = [os.path.join(class_dir, d) for d in os.listdir(class_dir)]
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        if self.max_videos_per_class is not None:
            subdirs = subdirs[: self.max_videos_per_class]
        # 이미지 파일 존재하는 폴더만 포함
        for vdir in subdirs:
            if any(os.path.splitext(f)[1].lower() in self.IMG_EXTS for f in os.listdir(vdir)):
                self.index.append((vdir, label))

    def __len__(self) -> int:
        return len(self.index)

    def _sorted_frames(self, vdir: str) -> List[str]:
        files = [f for f in os.listdir(vdir) if os.path.splitext(f)[1].lower() in self.IMG_EXTS]
        if self.sort_frames_numeric:
            try:
                files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            except Exception:
                files.sort()
        else:
            files.sort()
        paths = [os.path.join(vdir, f) for f in files]

        # === 프레임 샘플링 적용 ===
        paths = paths[self.frame_start::self.frame_stride]
        if self.frame_max is not None:
            paths = paths[: self.frame_max]
        return paths

    def __getitem__(self, idx: int):
        vdir, label = self.index[idx]
        frame_paths = self._sorted_frames(vdir)

        # 각 프레임에 대해 다중 전처리 적용
        # features_map: {"edge":[T,C,H,W], "texture":[T,C,H,W], ...}
        features_map: Dict[str, List[Tensor]] = {k: [] for k in self.processor.features}

        for fp in frame_paths:
            try:
                processed = self.processor.process_one(fp)  # {"edge":C,H,W, ...}
                for k, t in processed.items():
                    features_map[k].append(t)
            except Exception:
                # 손상 프레임이 있으면 비디오 전체를 스킵하고 다른 샘플로 대체
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)

        # 스택: feature별 [T,C,H,W]
        for k in list(features_map.keys()):
            if len(features_map[k]) == 0:
                # 비어있으면 다른 샘플로 대체
                new_idx = np.random.randint(0, len(self))
                return self.__getitem__(new_idx)
            features_map[k] = torch.stack(features_map[k], dim=0)  # [T,C,H,W]

        if self.mode == "dict":
            return features_map, label

        # concat 모드: C축으로 합치기
        videos: List[Tensor] = []
        for k in self.processor.features:
            videos.append(features_map[k])  # [T,C,H,W]
        # 채널 합치기: [T, sumC, H, W]
        video_concat = torch.cat(videos, dim=1)
        return video_concat, label


# -----------------------------
# Collate functions (제로패딩)
# -----------------------------
def collate_concat(batch: List[Tuple[Tensor, int]]):
    """
    batch: List of (video[T,C,H,W], label)
    -> videos[B, T_max, C, H, W], labels[B]
    """
    videos, labels = zip(*batch)
    T_max = max(v.shape[0] for v in videos)
    B = len(videos)
    C, H, W = videos[0].shape[1:]

    out = videos[0].new_zeros((B, T_max, C, H, W))
    for i, v in enumerate(videos):
        T = v.shape[0]
        out[i, :T] = v
    return out, torch.tensor(labels, dtype=torch.long)


def collate_dict(batch: List[Tuple[Dict[str, Tensor], int]]):
    """
    batch: List of ({"edge": [T,C,H,W], "texture":[T,C,H,W], ...}, label)
    -> (videos_dict, labels)
       where videos_dict[k] = [B, T_max, C, H, W]
    """
    feats_list, labels = zip(*batch)  # tuple of dicts, labels
    keys = list(feats_list[0].keys())

    # T_max 계산
    T_max = 0
    for d in feats_list:
        T_max = max(T_max, max(v.shape[0] for v in d.values()))

    out_dict: Dict[str, Tensor] = {}
    B = len(feats_list)
    for k in keys:
        C, H, W = feats_list[0][k].shape[1:]
        out = feats_list[0][k].new_zeros((B, T_max, C, H, W))
        for i, d in enumerate(feats_list):
            v = d[k]                # [T,C,H,W]
            T = v.shape[0]
            out[i, :T] = v
        out_dict[k] = out

    return out_dict, torch.tensor(labels, dtype=torch.long)