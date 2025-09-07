# dataset.py
# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Optional, Dict, Union
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# === FeatureManager 불러오기 ===
from utils.feature_manager import FeatureManager

Tensor = torch.Tensor


class ProcessorBuilder:
    """
    FeatureManager를 감싸서, 사용자가 원하는 feature 리스트만 선택적으로 사용.
    - 단일 프레임 처리: 'texture','edge','other'
    - 시퀀스 처리: 'temporal' (경로 리스트만 넘기면 프로세서가 앞/뒤까지 직접 로드)
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        features: Optional[List[str]] = None,  # 예: ["edge","texture","other"]
    ):
        self.manager = FeatureManager(config_path=config_path)

        # 단일 프레임 기본 features (temporal 제외)
        if features is None:
            self.features = ['texture', 'edge', 'other']
        else:
            all_keys = set(['texture', 'edge', 'other'])  # 여기선 단일 프레임만
            self.features = [f for f in features if f in all_keys]

        # 시퀀스(temporal) 사용 여부
        self.temporal_enabled = True

    def process_one(self, image: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, Tensor]:
        """
        단일 프레임에 대해 선택된 features 처리 -> {"edge":C,H,W, ...}
        """
        return self.manager.preprocess_selected(image, self.features)

    def process_sequence_from_paths(self, frame_paths: List[str], features: Optional[List[str]] = None) -> Dict[str, Tensor]:
        """
        경로 리스트만 넘기면 프로세서가 앞/뒤까지 직접 로드해서 시퀀스를 만든다.
        반환: {'temporal': [Seq, F, C, H, W]} (+ 선택 시 spatial도 [T,C,H,W])
        """
        if not self.temporal_enabled:
            raise RuntimeError("Temporal processing is disabled.")
        use_feats = ['temporal'] if features is None else features
        return self.manager.preprocess_sequence_selected_paths(frame_paths, use_feats)


class MultiProcVideoDataset(Dataset):
    """
    폴더 구조:
      root_fake/
        vid001/*.png
        ...
      root_real/
        vidABC/*.png
        ...

    mode:
      - "concat": features를 C축으로 concat하여 [T, C*, H, W]
      - "dict"  : feature별 [T, C, H, W] 딕셔너리 반환
      - "sequence": ★ temporal 전용 [Seq, F, C, H, W] 딕셔너리 반환 (키: 'temporal')

    프레임 샘플링:
      - frame_start, frame_stride, frame_max
    """
    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    def __init__(
        self,
        fake_dir: str,
        real_dir: Optional[str] = None,
        processor: Optional[ProcessorBuilder] = None,
        mode: str = "concat",            # "concat" | "dict" | "sequence"
        max_videos_per_class: Optional[int] = None,
        sort_frames_numeric: bool = True,
        # 프레임 샘플링 옵션
        frame_max: Optional[int] = None, # 비디오당 최대 프레임 수(None: 제한없음)
        frame_stride: int = 1,           # N프레임 간격 샘플링(최소 1)
        frame_start: int = 0,            # 시작 offset(최소 0)
        # 시퀀스에서 spatial도 함께 뽑고 싶을 때 지정 (예: ['temporal','edge'])
        sequence_features: Optional[List[str]] = None,
    ):
        assert mode in ("concat", "dict", "sequence")
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
        self.sequence_features = sequence_features or ['temporal']

        # (vdir, label)
        self.index: List[Tuple[str, int]] = []
        self._index_class(fake_dir, label=0)
        if real_dir is not None:
            self._index_class(real_dir, label=1)

        if not self.index:
            raise RuntimeError("No videos found. Check paths.")

        # 미리 가공 데이터를 담아둘 저장소
        self.tank: List[Tuple[Union[Dict[str, Tensor], Tensor], int]] = []

        # 비디오 단위로 가공
        for vdir, label in self.index:
            frame_paths = self._sorted_frames(vdir)
            if not frame_paths:
                raise RuntimeError(f"비디오에서 프레임을 찾을 수 없습니다: {vdir}")

            if self.mode == "sequence":
                # 경로 리스트를 그대로 넘겨서 내부에서 앞/뒤 로드 → 시퀀스 생성
                out_dict = self.processor.process_sequence_from_paths(frame_paths, features=self.sequence_features)
                # 최소 'temporal' 키는 있어야 함
                if 'temporal' not in out_dict:
                    raise RuntimeError("sequence mode requires 'temporal' output.")
                self.tank.append((out_dict, label))
                continue

            # 기존 dict/concat 경로 (단일 프레임 반복)
            features_map: Dict[str, List[torch.Tensor]] = {k: [] for k in self.processor.features}
            for fp in tqdm(frame_paths, desc=f"Processing {vdir}"):
                processed = self.processor.process_one(fp)
                for k, t in processed.items():
                    features_map[k].append(t)

            # 유효성 검사
            if any(len(v) == 0 for v in features_map.values()):
                raise RuntimeError(f"처리된 특징 맵이 비어있습니다. 손상된 비디오일 수 있습니다: {vdir}")

            # 모든 특징들을 텐서로 스택
            for k in list(features_map.keys()):
                features_map[k] = torch.stack(features_map[k], dim=0)  # [T,C,H,W]

            if self.mode == "dict":
                self.tank.append((features_map, label))
            else:  # "concat"
                # C축으로 concat
                keys = sorted(features_map.keys())
                cat = torch.cat([features_map[k] for k in keys], dim=1)  # [T, sumC, H, W]
                self.tank.append((cat, label))

    def _index_class(self, class_dir: str, label: int):
        if not os.path.isdir(class_dir):
            return
        subdirs = [os.path.join(class_dir, d) for d in os.listdir(class_dir)]
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        if self.max_videos_per_class is not None:
            subdirs = subdirs[: self.max_videos_per_class]
        # 이미지 파일 존재하는 폴더만 포함
        for vdir in subdirs:
            try:
                if any(os.path.splitext(f)[1].lower() in self.IMG_EXTS for f in os.listdir(vdir)):
                    self.index.append((vdir, label))
            except Exception:
                # 권한/깨진 디렉토리 등 무시
                continue

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
        # 유효한 샘플을 찾을 때까지 무한 루프
        while True:
            try:
                feats_or_tensor, label = self.tank[idx]
                return feats_or_tensor, label
            except Exception as e:
                print(f"[WARN] 인덱스 {idx} 비디오 처리 중 오류: {e}. 다른 샘플로 대체합니다.")
                idx = np.random.randint(0, len(self))


# -----------------------------
# Collate functions
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
    batch: List of ({"edge":[T,C,H,W], "texture":[T,C,H,W], ...}, label)
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


def collate_sequence(batch: List[Tuple[Dict[str, Tensor], int]]):
    """
    batch: [({'temporal':[S,F,C,H,W], ...}, label), ...]
    -> x: [B, S_max, F, C, H, W], y: [B]
    (추가로 spatial이 함께 들어온 경우, 필요 시 별도 꺼내 쓰면 됨)
    """
    dicts, labels = zip(*batch)
    key = 'temporal'
    S_max = max(d[key].shape[0] for d in dicts)
    B = len(dicts)
    F, C, H, W = dicts[0][key].shape[1:]
    out = dicts[0][key].new_zeros((B, S_max, F, C, H, W))
    for i, d in enumerate(dicts):
        S = d[key].shape[0]
        out[i, :S] = d[key]
    return out, torch.tensor(labels, dtype=torch.long)