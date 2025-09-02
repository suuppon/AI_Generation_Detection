# feature_manager.py
import os
import json
from typing import Dict, Any, Union, List
import torch
import numpy as np

from utils import TexturePreprocessor, EdgePreprocessor, OtherPreprocessor

class FeatureManager:
    """Feature 전처리 통합 관리자"""
    
    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 설정 파일 경로 (JSON)
        """
        self.preprocessors = {}
        self.config = self._load_config(config_path) if config_path else {}
        self._initialize_preprocessors()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _initialize_preprocessors(self):
        """전처리기 초기화"""
        # 기본 설정
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
        
        # 설정 파일의 값으로 업데이트
        config = {**default_config, **self.config}
        
        # 전처리기 생성
        self.preprocessors['texture'] = TexturePreprocessor(**config['texture'])
        self.preprocessors['edge'] = EdgePreprocessor(**config['edge'])
        self.preprocessors['other'] = OtherPreprocessor(**config['other'])
    
    def preprocess_texture(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Texture feature 전처리"""
        return self.preprocessors['texture'].preprocess(image)
    
    def preprocess_edge(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Edge feature 전처리"""
        return self.preprocessors['edge'].preprocess(image)
    
    def preprocess_other(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """기타 feature 전처리"""
        return self.preprocessors['other'].preprocess(image)
    
    def preprocess_all(self, image: Union[str, np.ndarray, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """모든 feature 전처리"""
        return {
            'texture': self.preprocess_texture(image),
            'edge': self.preprocess_edge(image),
            'other': self.preprocess_other(image)
        }
    
    def preprocess_batch_all(self, images: List[Union[str, np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """배치 이미지의 모든 feature 전처리"""
        results = {'texture': [], 'edge': [], 'other': []}
        
        for image in images:
            processed = self.preprocess_all(image)
            for feature_type, tensor in processed.items():
                results[feature_type].append(tensor)
        
        return {
            'texture': torch.stack(results['texture']),
            'edge': torch.stack(results['edge']),
            'other': torch.stack(results['other'])
        }
    
    def get_preprocessor(self, feature_type: str):
        """특정 feature 전처리기 반환"""
        if feature_type not in self.preprocessors:
            raise ValueError(f"Unknown feature type: {feature_type}")
        return self.preprocessors[feature_type]

# 사용 예제
if __name__ == "__main__":
    # Feature 매니저 생성
    manager = FeatureManager()
    
    # 개별 feature 전처리
    try:
        # Texture feature만 전처리
        texture_tensor = manager.preprocess_texture("cat_img.png")
        print(f"Texture tensor shape: {texture_tensor.shape}")
        
        # Edge feature만 전처리
        edge_tensor = manager.preprocess_edge("cat_img.png")
        print(f"Edge tensor shape: {edge_tensor.shape}")
        
        # 기타 feature만 전처리
        other_tensor = manager.preprocess_other("cat_img.png")
        print(f"Other tensor shape: {other_tensor.shape}")
        
        # 모든 feature 전처리
        all_features = manager.preprocess_all("cat_img.png")
        print(f"All features: {list(all_features.keys())}")
        
    except FileNotFoundError:
        print("cat_img.png 파일을 찾을 수 없습니다.")
        
        # 더미 이미지로 테스트
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        texture_tensor = manager.preprocess_texture(dummy_image)
        edge_tensor = manager.preprocess_edge(dummy_image)
        other_tensor = manager.preprocess_other(dummy_image)
        
        print(f"Texture tensor shape: {texture_tensor.shape}")
        print(f"Edge tensor shape: {edge_tensor.shape}")
        print(f"Other tensor shape: {other_tensor.shape}")
        
        # 배치 처리 테스트
        dummy_batch = [dummy_image] * 3
        batch_results = manager.preprocess_batch_all(dummy_batch)
        
        for feature_type, tensor in batch_results.items():
            print(f"{feature_type} batch shape: {tensor.shape}")