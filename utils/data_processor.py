# edge_preprocessor.py
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """기본 전처리 추상 클래스"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.torch_transform = self._create_torch_transform()
    
    def _create_torch_transform(self):
        """PyTorch transform 생성"""
        transform_list = []
        transform_list.append(transforms.Resize(self.target_size))
        transform_list.append(transforms.ToTensor())
        
        if self.normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            ))
        
        return transforms.Compose(transform_list)
    
    def load_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """이미지 로드 및 기본 전처리"""
        if isinstance(image, str):
            img = cv.imread(image)
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image.copy()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        elif isinstance(image, torch.Tensor):
            img = image.numpy()
            if len(img.shape) == 3:
                img = np.transpose(img, (1, 2, 0))
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return img
    
    @abstractmethod
    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Feature별 전처리 수행"""
        pass
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """배치 전처리"""
        processed_images = []
        for image in images:
            processed = self.preprocess(image)
            processed_images.append(processed)
        return torch.stack(processed_images)

class EdgePreprocessor(BasePreprocessor):
    """Edge feature 전처리 클래스"""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 canny_low: int = 50,
                 canny_high: int = 150,
                 sobel_kernel: int = 3,
                 laplacian_kernel: int = 3,
                 edge_method: str = 'canny',
                 edge_weight: float = 0.4,
                 edge_enhancement: bool = False,
                 normalize: bool = True):
        """
        Args:
            target_size: 출력 이미지 크기
            canny_low: Canny 하한 임계값
            canny_high: Canny 상한 임계값
            sobel_kernel: Sobel 커널 크기
            laplacian_kernel: Laplacian 커널 크기
            edge_method: 엣지 검출 방법 ('canny', 'sobel', 'laplacian')
            edge_weight: 엣지 feature 가중치
            edge_enhancement: 엣지 강화 여부
            normalize: 정규화 여부
        """
        super().__init__(target_size, normalize)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.sobel_kernel = sobel_kernel
        self.laplacian_kernel = laplacian_kernel
        self.edge_method = edge_method
        self.edge_weight = edge_weight
        self.edge_enhancement = edge_enhancement
    
    def apply_canny(self, image: np.ndarray) -> np.ndarray:
        """Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        return cv.Canny(gray, self.canny_low, self.canny_high)
    
    def apply_sobel(self, image: np.ndarray) -> np.ndarray:
        """Sobel edge detection"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=self.sobel_kernel)
        sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=self.sobel_kernel)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        return np.uint8(magnitude / magnitude.max() * 255)
    
    def apply_laplacian(self, image: np.ndarray) -> np.ndarray:
        """Laplacian edge detection"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        laplacian = cv.Laplacian(gray, cv.CV_64F, ksize=self.laplacian_kernel)
        return np.uint8(np.absolute(laplacian))
    
    def apply_edge_enhancement(self, image: np.ndarray) -> np.ndarray:
        """엣지 강화"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        return cv.filter2D(image, -1, kernel)
    
    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Edge feature 전처리"""
        img = self.load_image(image)
        result = img.copy()
        
        # 엣지 검출
        if self.edge_method == 'canny':
            edges = self.apply_canny(img)
        elif self.edge_method == 'sobel':
            edges = self.apply_sobel(img)
        elif self.edge_method == 'laplacian':
            edges = self.apply_laplacian(img)
        else:
            edges = self.apply_canny(img)
        
        # 엣지 강화
        if self.edge_enhancement:
            result = self.apply_edge_enhancement(result)
        
        # 엣지 정보 결합
        edges_3d = np.stack([edges, edges, edges], axis=2)
        result = cv.addWeighted(result, 1-self.edge_weight, edges_3d, self.edge_weight, 0)
        
        # PIL Image로 변환 후 PyTorch transform 적용
        from PIL import Image
        pil_image = Image.fromarray(result)
        return self.torch_transform(pil_image)

class OtherPreprocessor(BasePreprocessor):
    """기타 feature 전처리 클래스"""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 histogram_equalization: bool = False,
                 noise_reduction: bool = False,
                 blur: bool = False,
                 blur_kernel: int = 3,
                 contrast_enhancement: bool = False,
                 brightness_adjustment: float = 0.0,
                 noise_factor: float = 0.0,
                 normalize: bool = True):
        """
        Args:
            target_size: 출력 이미지 크기
            histogram_equalization: 히스토그램 평활화 여부
            noise_reduction: 노이즈 제거 여부
            blur: 블러 적용 여부
            blur_kernel: 블러 커널 크기
            contrast_enhancement: 대비 향상 여부
            brightness_adjustment: 밝기 조정 값 (-1~1)
            noise_factor: 노이즈 추가 강도 (0~1)
            normalize: 정규화 여부
        """
        super().__init__(target_size, normalize)
        self.histogram_equalization = histogram_equalization
        self.noise_reduction = noise_reduction
        self.blur = blur
        self.blur_kernel = blur_kernel
        self.contrast_enhancement = contrast_enhancement
        self.brightness_adjustment = brightness_adjustment
        self.noise_factor = noise_factor
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """히스토그램 평활화"""
        if len(image.shape) == 3:
            yuv = cv.cvtColor(image, cv.COLOR_RGB2YUV)
            yuv[:,:,0] = cv.equalizeHist(yuv[:,:,0])
            return cv.cvtColor(yuv, cv.COLOR_YUV2RGB)
        else:
            return cv.equalizeHist(image)
    
    def apply_noise_reduction(self, image: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        return cv.bilateralFilter(image, 9, 75, 75)
    
    def apply_noise_addition(self, image: np.ndarray) -> np.ndarray:
        """노이즈 추가"""
        noise = np.random.normal(0, self.noise_factor * 255, image.shape).astype(np.uint8)
        result = cv.add(image, noise)
        return np.clip(result, 0, 255)
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """블러 적용"""
        return cv.GaussianBlur(image, (self.blur_kernel, self.blur_kernel), 0)
    
    def apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """대비 향상"""
        if len(image.shape) == 3:
            lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            return cv.cvtColor(lab, cv.COLOR_LAB2RGB)
        else:
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            return clahe.apply(image)
    
    def apply_brightness_adjustment(self, image: np.ndarray) -> np.ndarray:
        """밝기 조정"""
        if self.brightness_adjustment != 0:
            return cv.convertScaleAbs(image, alpha=1, beta=self.brightness_adjustment * 255)
        return image.copy()
    
    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """기타 feature 전처리"""
        img = self.load_image(image)
        result = img.copy()
        
        # 순차적으로 전처리 적용
        if self.histogram_equalization:
            result = self.apply_histogram_equalization(result)
        
        if self.noise_reduction:
            result = self.apply_noise_reduction(result)
        
        if self.noise_factor > 0 and not self.noise_reduction:
            result = self.apply_noise_addition(result)
        
        if self.blur:
            result = self.apply_blur(result)
        
        if self.contrast_enhancement:
            result = self.apply_contrast_enhancement(result)
        
        result = self.apply_brightness_adjustment(result)
        
        # PIL Image로 변환 후 PyTorch transform 적용
        from PIL import Image
        pil_image = Image.fromarray(result)
        return self.torch_transform(pil_image)

class TexturePreprocessor(BasePreprocessor):
    """Texture feature 전처리 클래스"""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 gabor_angles: list = [0, 45, 90, 135],
                 gabor_freq: float = 0.1,
                 lbp_radius: int = 1,
                 lbp_points: int = 8,
                 use_gabor: bool = True,
                 use_lbp: bool = True,
                 texture_weight: float = 0.3,
                 normalize: bool = True):
        """
        Args:
            target_size: 출력 이미지 크기
            gabor_angles: Gabor 필터 각도들
            gabor_freq: Gabor 필터 주파수
            lbp_radius: LBP 반지름
            lbp_points: LBP 포인트 수
            use_gabor: Gabor 필터 사용 여부
            use_lbp: LBP 사용 여부
            texture_weight: texture feature 가중치
            normalize: 정규화 여부
        """
        super().__init__(target_size, normalize)
        self.gabor_angles = gabor_angles
        self.gabor_freq = gabor_freq
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.use_gabor = use_gabor
        self.use_lbp = use_lbp
        self.texture_weight = texture_weight
    
    def apply_gabor_filter(self, image: np.ndarray) -> np.ndarray:
        """Gabor 필터 적용"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gabor_responses = []
        for angle in self.gabor_angles:
            kernel = cv.getGaborKernel(
                ksize=(21, 21),
                sigma=5.0,
                theta=np.radians(angle),
                lambd=self.gabor_freq,
                gamma=0.5,
                psi=0
            )
            filtered = cv.filter2D(gray, cv.CV_8UC3, kernel)
            gabor_responses.append(filtered)
        
        result = np.mean(gabor_responses, axis=0).astype(np.uint8)
        return result
    
    def apply_lbp(self, image: np.ndarray) -> np.ndarray:
        """Local Binary Pattern 적용"""
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        height, width = gray.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(self.lbp_radius, height - self.lbp_radius):
            for j in range(self.lbp_radius, width - self.lbp_radius):
                center = gray[i, j]
                code = 0
                for k in range(self.lbp_points):
                    angle = 2 * np.pi * k / self.lbp_points
                    x = int(i + self.lbp_radius * np.cos(angle))
                    y = int(j + self.lbp_radius * np.sin(angle))
                    
                    x = max(0, min(x, height - 1))
                    y = max(0, min(y, width - 1))
                    
                    if gray[x, y] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        return lbp
    
    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Texture feature 전처리"""
        img = self.load_image(image)
        result = img.copy()
        
        # Texture features 적용
        if self.use_gabor:
            gabor_result = self.apply_gabor_filter(img)
            gabor_3d = np.stack([gabor_result, gabor_result, gabor_result], axis=2)
            result = cv.addWeighted(result, 1-self.texture_weight, gabor_3d, self.texture_weight, 0)
        
        if self.use_lbp:
            lbp_result = self.apply_lbp(img)
            lbp_3d = np.stack([lbp_result, lbp_result, lbp_result], axis=2)
            result = cv.addWeighted(result, 1-self.texture_weight, lbp_3d, self.texture_weight, 0)
        
        # PIL Image로 변환 후 PyTorch transform 적용
        from PIL import Image
        pil_image = Image.fromarray(result)
        return self.torch_transform(pil_image)

# 사용 예제
if __name__ == "__main__":
    # Edge 전처리기 생성
    edge_preprocessor = EdgePreprocessor(
        target_size=(224, 224),
        edge_method='canny',
        edge_weight=0.4,
        edge_enhancement=True
    )
    
    # 이미지 전처리 테스트
    try:
        edge_tensor = edge_preprocessor.preprocess("cat_img.png")
        print(f"Edge tensor shape: {edge_tensor.shape}")
        print(f"Edge tensor range: {edge_tensor.min():.3f} ~ {edge_tensor.max():.3f}")
    except FileNotFoundError:
        print("cat_img.png 파일을 찾을 수 없습니다.")
        
        # 더미 이미지로 테스트
        dummy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        edge_tensor = edge_preprocessor.preprocess(dummy_image)
        print(f"Edge tensor shape: {edge_tensor.shape}")