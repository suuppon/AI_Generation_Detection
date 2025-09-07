# feature_manager.py
import os
import json
from typing import Dict, Any, Union, List, Optional
import torch
import numpy as np
from PIL import Image
import cv2

# utils.data_processor 안에 있는 전처리기들 사용
from utils.data_processor import (
    TexturePreprocessor,
    EdgePreprocessor,
    OtherPreprocessor,
    TemporalWindowPreprocessor,  # ★ 경로 리스트를 입력받아 내부에서 앞/뒤 프레임을 직접 로드
)

ImageLike = Union[str, np.ndarray, torch.Tensor]


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
        return arr
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


class FeatureManager:
    """Feature 전처리 통합 관리자"""

    def __init__(self, config_path: Optional[str] = None):
        self.preprocessors: Dict[str, Any] = {}
        self.config = self._load_config(config_path) if config_path else {}
        self._initialize_preprocessors()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _initialize_preprocessors(self):
        default_config = {
            # ----- spatial features -----
            'texture': {
                'target_size': (224, 224),
                'gabor_angles': [0, 45, 90, 135],
                'gabor_freq': 0.1,
                'lbp_radius': 1,
                'lbp_points': 8,
                'use_gabor': True,
                'use_lbp': True,
                'texture_weight': 0.3,
                'normalize': True
            },
            'edge': {
                'target_size': (224, 224),
                'canny_low': 50,
                'canny_high': 150,
                'sobel_kernel': 3,
                'laplacian_kernel': 3,
                'edge_method': 'canny',   # 'canny' | 'sobel' | 'laplacian'
                'edge_weight': 0.4,
                'edge_enhancement': False,
                'normalize': True
            },
            'other': {
                'target_size': (224, 224),
                'histogram_equalization': True,
                'contrast_enhancement': True,
                'noise_reduction': False,
                'blur': False,
                'blur_kernel': 3,
                'brightness_adjustment': 0.0,
                'noise_factor': 0.0,
                'normalize': True
            },
            # ----- temporal (sequence) -----
            'temporal': {
                'target_size': (224, 224),
                'frames_per_step': 3,  # 예: [t-1, t, t+1]
                'stride': 1,
                'pad': 'edge',         # 'edge' | 'zero'
                'normalize': True
            }
        }

        cfg = {**default_config, **self.config}

        # 개별 전처리기 초기화
        self.preprocessors['texture']  = TexturePreprocessor(**cfg['texture'])
        self.preprocessors['edge']     = EdgePreprocessor(**cfg['edge'])
        self.preprocessors['other']    = OtherPreprocessor(**cfg['other'])
        self.preprocessors['temporal'] = TemporalWindowPreprocessor(**cfg['temporal'])  # ★

    # -----------------------------
    # 단일 프레임용 (spatial features)
    # -----------------------------
    def preprocess_texture(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['texture'].preprocess(arr)

    def preprocess_edge(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['edge'].preprocess(arr)

    def preprocess_other(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['other'].preprocess(arr)

    def preprocess_selected(self, image: ImageLike, features: List[str]) -> Dict[str, torch.Tensor]:
        """
        지정된 spatial features만 처리. 디스크/메모리 로드는 1회.
        ※ temporal은 시퀀스 전용이므로 여기서 다루지 않는다.
        """
        if 'temporal' in features:
            raise ValueError("Use 'preprocess_temporal_paths' for temporal features (sequence from paths).")
        arr = _load_image_any(image)
        out: Dict[str, torch.Tensor] = {}
        for k in features:
            if k not in self.preprocessors:
                raise ValueError(f"Unknown feature type: {k}")
            out[k] = self.preprocessors[k].preprocess(arr)
        return out

    def preprocess_all(self, image: ImageLike) -> Dict[str, torch.Tensor]:
        """texture/edge/other 전부를 1회 로드로 처리 (temporal 제외)"""
        return self.preprocess_selected(image, ['texture', 'edge', 'other'])

    # -----------------------------
    # 시퀀스(프레임 경로 리스트) 전용
    # -----------------------------
    def preprocess_temporal_paths(self, frame_paths: List[str]) -> torch.Tensor:
        """
        프레임 '경로 리스트'를 받아 (Seq_len, Frames_per_step, C, H, W)를 반환.
        내부에서 앞/뒤 프레임까지 직접 로드하여 윈도우 구성.
        """
        proc: TemporalWindowPreprocessor = self.preprocessors['temporal']
        return proc.preprocess_paths(frame_paths)

    def preprocess_sequence_selected_paths(self, frame_paths: List[str], features: List[str]) -> Dict[str, torch.Tensor]:
        """
        시퀀스 입력에서 선택된 feature 처리.
        - 'temporal' → [Seq_len, F, C, H, W]
        - 나머지 spatial features → 모든 프레임을 독립 처리 후 [T, C, H, W]
        """
        out: Dict[str, torch.Tensor] = {}
        want_temporal = 'temporal' in features

        # temporal 먼저
        if want_temporal:
            out['temporal'] = self.preprocess_temporal_paths(frame_paths)

        # spatial들
        spatial = [k for k in features if k != 'temporal']
        if spatial:
            # 경로 리스트를 한 장씩 읽어 처리
            tensors_by_key: Dict[str, List[torch.Tensor]] = {k: [] for k in spatial}
            for p in frame_paths:
                arr = _load_image_any(p)
                for k in spatial:
                    proc = self.preprocessors[k]
                    tensors_by_key[k].append(proc.preprocess(arr))
            for k in spatial:
                out[k] = torch.stack(tensors_by_key[k], dim=0)  # [T, C, H, W]
        return out

    # -----------------------------
    # 배치 시퀀스 처리 (옵션)
    # -----------------------------
    def preprocess_batch_sequences(
        self,
        sequences: List[List[str]],
        features: List[str] = ('temporal',),
    ) -> Dict[str, torch.Tensor]:
        """
        sequences: 각 항목이 프레임 경로 리스트인 리스트
        반환:
          - 'temporal' in features: [B, Seq, F, C, H, W]
          - spatial features: [B, T, C, H, W]
        """
        bucket: Dict[str, List[torch.Tensor]] = {}
        for paths in sequences:
            out = self.preprocess_sequence_selected_paths(paths, list(features))
            for k, v in out.items():
                bucket.setdefault(k, []).append(v)

        stacked: Dict[str, torch.Tensor] = {}
        for k, vs in bucket.items():
            stacked[k] = torch.stack(vs, dim=0)
        return stacked