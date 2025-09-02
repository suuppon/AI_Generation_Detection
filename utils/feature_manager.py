# feature_manager.py
import os
import json
from typing import Dict, Any, Union, List
import torch
import numpy as np
from PIL import Image
import cv2

from utils import TexturePreprocessor, EdgePreprocessor, OtherPreprocessor

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
        # 채널/타입은 호출부에서 맞춘다고 가정
        return arr
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

class FeatureManager:
    """Feature 전처리 통합 관리자"""

    def __init__(self, config_path: str = None):
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
                'edge_method': 'canny',
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
                'normalize': True
            }
        }
        cfg = {**default_config, **self.config}
        self.preprocessors['texture'] = TexturePreprocessor(**cfg['texture'])
        self.preprocessors['edge']    = EdgePreprocessor(**cfg['edge'])
        self.preprocessors['other']   = OtherPreprocessor(**cfg['other'])

    # -------- 단일 feature (개별 호출 시엔 각각 1회 로드됨) --------
    def preprocess_texture(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['texture'].preprocess(arr)

    def preprocess_edge(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['edge'].preprocess(arr)

    def preprocess_other(self, image: ImageLike) -> torch.Tensor:
        arr = _load_image_any(image)
        return self.preprocessors['other'].preprocess(arr)

    # -------- 여러 feature를 한 번에: 이미지 1회 로드 공유(핵심) --------
    def preprocess_selected(self, image: ImageLike, features: List[str]) -> Dict[str, torch.Tensor]:
        """
        지정된 features만 처리. 디스크/메모리 로드는 단 1회.
        """
        arr = _load_image_any(image)
        out: Dict[str, torch.Tensor] = {}
        for k in features:
            if k not in self.preprocessors:
                raise ValueError(f"Unknown feature type: {k}")
            out[k] = self.preprocessors[k].preprocess(arr)
        return out

    def preprocess_all(self, image: ImageLike) -> Dict[str, torch.Tensor]:
        """texture/edge/other 전부를 1회 로드로 처리"""
        return self.preprocess_selected(image, ['texture', 'edge', 'other'])

    def preprocess_batch_all(self, images: List[ImageLike]) -> Dict[str, torch.Tensor]:
        """
        배치 처리. 각 이미지당 1회 로드 + 3 feature 계산 후 stack.
        실질 병목은 여기의 전처리 연산이므로 DataLoader의 num_workers/prefetch로 병렬화 권장.
        """
        results = {'texture': [], 'edge': [], 'other': []}
        for image in images:
            processed = self.preprocess_all(image)  # <-- 여기서도 각 이미지 1회 로드
            for k, t in processed.items():
                results[k].append(t)
        return {
            'texture': torch.stack(results['texture']),
            'edge':    torch.stack(results['edge']),
            'other':   torch.stack(results['other']),
        }

    def get_preprocessor(self, feature_type: str):
        if feature_type not in self.preprocessors:
            raise ValueError(f"Unknown feature type: {feature_type}")
        return self.preprocessors[feature_type]