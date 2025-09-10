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
import random

# === FeatureManager 불러오기 ===
from utils.feature_manager import FeatureManager

### CPU 별렬 처리
from multiprocessing import Pool
import time

import signal
import pickle

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

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
        mode: str = "concat",                 # "concat" | "dict" | "sequence"
        max_videos_per_class: Optional[int] = None,
        sort_frames_numeric: bool = True,
        # 프레임 샘플링 옵션
        frame_max: Optional[int] = None,      # 비디오당 최대 프레임 수(None: 제한없음)
        frame_stride: int = 1,                # N프레임 간격 샘플링(최소 1)
        frame_start: int = 0,                 # 시작 offset(최소 0)
        # 시퀀스에서 spatial도 함께 뽑고 싶을 때 지정 (예: ['temporal','edge'])
        sequence_features: Optional[List[str]] = None,
        # dict 모드에서 temporal 키 포함 및 옵션
        include_temporal_in_dict: bool = True,
        temporal_channels: int = 3,           # temporal 채널 수 (1=grayscale, 3=RGB)
        normalize_temporal: bool = False,     # True면 [0,1] 정규화
        temporal_num_frames: int = 8,         # num_frames(F)
        temporal_hop: int = 4,                # 윈도우 hop
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

        # 프로세서 & 시퀀스 피처
        self.processor = processor or ProcessorBuilder(config_path=None, features=None)
        self.sequence_features = sequence_features or ['temporal']

        # temporal 옵션
        self.include_temporal_in_dict = include_temporal_in_dict
        self.temporal_channels = temporal_channels
        self.normalize_temporal = normalize_temporal
        self.temporal_num_frames = max(1, temporal_num_frames)
        self.temporal_hop = max(1, temporal_hop)

        # (vdir, label) 인덱싱
        self.index: List[Tuple[str, int]] = []
        self._index_class(fake_dir, label=0)
        if real_dir is not None:
            self._index_class(real_dir, label=1)

        if not self.index:
            raise RuntimeError("No videos found. Check paths.")

        # 미리 가공 저장
        self.tank: List[Tuple[Union[Dict[str, Tensor], Tensor], int]] = []

        # 비디오 단위로 가공
        #시그널
        signal.signal(signal.SIGALRM, handler)
        #랜덤성 부여
        random.shuffle(self.index)

        for vdir, label in self.index:
            print(f"{vdir} is processing ... ")
            start = time.time()
            frame_paths = self._sorted_frames(vdir)
            if not frame_paths:
                raise RuntimeError(f"비디오에서 프레임을 찾을 수 없습니다: {vdir}")

            # --- sequence 모드: FeatureManager가 [S,F,C,H,W] 생성 ---
            if self.mode == "sequence":
                out_dict = self.processor.process_sequence_from_paths(
                    frame_paths, features=self.sequence_features
                )
                if 'temporal' not in out_dict:
                    raise RuntimeError("sequence mode requires 'temporal' output.")
                self.tank.append((out_dict, label))
                continue

            # 기존 dict/concat 경로 (단일 프레임 반복)
            #features_map: Dict[str, List[torch.Tensor]] = {k: [] for k in self.processor.features}
            

            #비정상적으로 길게 걸리는 이미지 처리
            fp = frame_paths[0]
            img = cv2.imread(fp)
            if img is None:
                print(f"읽기 실패, 건너뜀: {fp}")
                continue  # 읽기 실패하면 다음 프레임으로 넘어감
            
            try:
                signal.alarm(300)
                with Pool(processes=4) as pool:  # CPU 코어 수
                    processed = pool.map(self.processor.process_one, frame_paths)
                    features_map = {key: torch.from_numpy(np.stack([item[key] for item in processed], axis=0)) for key in self.processor.features}
                signal.alarm(0)
            except TimeoutException:
                print(f"{vdir} 처리 시간 초과, 건너뜀")
                continue
            
            end = time.time()
            print(f"{vdir} is processed, in {round(end - start)} s")
            # 유효성 검사
            if any(len(v) == 0 for v in features_map.values()):
                raise RuntimeError(f"처리된 특징 맵이 비어있습니다. 손상된 비디오일 수 있습니다: {vdir}")

            # 스택: [T,C,H,W]
            for k in list(features_map.keys()):
                features_map[k] = torch.stack(features_map[k], dim=0)

            # --- dict 모드에서 temporal을 [S,F,C,H,W]로 추가 ---
            if self.mode == "dict" and self.include_temporal_in_dict:
                any_key = next(iter(features_map.keys()))
                _, H, W = features_map[any_key].shape[1:]
                temporal_stack = self._build_temporal_windows(
                    frame_paths, (H, W),
                    F=self.temporal_num_frames,
                    hop=self.temporal_hop
                )  # [S,F,C,H,W]
                features_map['temporal'] = temporal_stack

            # 저장
            if self.mode == "dict":
                self.tank.append((features_map, label))
            else:  # "concat"
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

    @torch.no_grad()
    def _build_temporal_windows(self, frame_paths: List[str], target_hw: Tuple[int, int],
                                F: int, hop: int) -> Tensor:
        """
        frame_paths → [S, F, C, H, W]
        S = 1 + floor((T - F) / hop) (T < F이면 S=1로 패딩 없이 마지막 잘라내기 대신, F>T면 F=T로 다운셋)
        """
        H, W = target_hw
        T = len(frame_paths)
        if T == 0:
            raise RuntimeError("No frames to build temporal windows.")

        # 필요한 경우 F를 T로 축소 (짧은 비디오 대응)
        F_eff = min(F, T)

        # 모든 프레임 로드 (한 번만)
        frames = []
        for fp in frame_paths:
            img = Image.open(fp).convert("RGB")
            if img.size != (W, H):
                img = img.resize((W, H), Image.BILINEAR)
            arr = np.asarray(img)  # H,W,3
            t = torch.from_numpy(arr).permute(2, 0, 1).float()  # [3,H,W]
            if self.temporal_channels == 1:
                y = (0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]).unsqueeze(0)
                t = y
            if self.normalize_temporal:
                t = t / 255.0
            frames.append(t)  # [C,H,W]
        stack = torch.stack(frames, dim=0)  # [T,C,H,W]

        # 윈도우 추출
        if T <= F_eff:
            windows = stack.unsqueeze(0)  # [1,T,C,H,W]
            # 앞쪽으로 0-padding해서 길이를 F로 맞추기(원하면 유지도 가능)
            if T < F:
                pad = stack.new_zeros((F - T, *stack.shape[1:]))
                windows = torch.cat([windows[:, :0], torch.cat([stack, pad], dim=0).unsqueeze(0)], dim=1)  # [1,F,C,H,W]
            return windows  # [1,F,C,H,W] (또는 [1,T,C,H,W] if T==F)

        starts = list(range(0, T - F_eff + 1, hop))
        windows = []
        for s in starts:
            clip = stack[s:s+F_eff]  # [F_eff,C,H,W]
            # F_eff < F이면 뒤쪽 0패딩
            if F_eff < F:
                pad = stack.new_zeros((F - F_eff, *stack.shape[1:]))
                clip = torch.cat([clip, pad], dim=0)
            windows.append(clip.unsqueeze(0))  # [1,F,C,H,W]
        return torch.cat(windows, dim=0)  # [S,F,C,H,W]

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


# collate_dict 수정: temporal은 [B, S_max, F, C, H, W]로, 나머지는 [B, T_max, C, H, W]
def collate_dict(batch):
    dicts, labels = zip(*batch)
    keys = list(dicts[0].keys())

    out_dict: Dict[str, Tensor] = {}
    B = len(dicts)

    for k in keys:
        sample = dicts[0][k]
        if sample.ndim == 5:
            # temporal: [S, F, C, H, W] -> [B, S_max, F, C, H, W]
            S_max = max(d[k].shape[0] for d in dicts)
            F, C, H, W = sample.shape[1:]
            out = sample.new_zeros((B, S_max, F, C, H, W))
            for i, d in enumerate(dicts):
                S = d[k].shape[0]
                out[i, :S] = d[k]
            out_dict[k] = out
        else:
            # 일반: [T, C, H, W] -> [B, T_max, C, H, W]
            T_max = max(d[k].shape[0] for d in dicts)
            C, H, W = sample.shape[1:]
            out = sample.new_zeros((B, T_max, C, H, W))
            for i, d in enumerate(dicts):
                v = d[k]
                T = v.shape[0]
                out[i, :T] = v
            out_dict[k] = out

    return out_dict, torch.tensor(labels, dtype=torch.long)