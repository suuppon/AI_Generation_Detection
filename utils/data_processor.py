# utils/data_processor.py
import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from typing import Union, Tuple, Optional, List, Dict
from abc import ABC, abstractmethod

# OpenCV 전역 최적화
try:
    cv.setNumThreads(0)  # 파이썬 GIL과의 경합을 줄이는 경우가 많음
    cv.setUseOptimized(True)
except Exception:
    pass


def _to_rgb_np(img: np.ndarray) -> np.ndarray:
    """BGR 가능성을 가진 ndarray를 RGB ndarray로 보장."""
    if img.ndim == 3 and img.shape[2] == 3:
        # 대부분의 ndarray는 OpenCV 경로(BGR)에서 옴
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


class BasePreprocessor(ABC):
    """기본 전처리 추상 클래스"""

    def __init__(self, target_size: Tuple[int, int] = (224, 224), normalize: bool = True):
        self.target_size = target_size
        self.normalize = normalize
        self.torch_transform = self._create_torch_transform()

    def _create_torch_transform(self):
        """PyTorch transform 생성 (Resize → ToTensor → Normalize)"""
        tfms = [transforms.Resize(self.target_size),
                transforms.ToTensor()]
        if self.normalize:
            tfms.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            )
        return transforms.Compose(tfms)

    def load_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> np.ndarray:
        """이미지 로드 및 기본 전처리: RGB uint8 ndarray(HWC)로 통일."""
        if isinstance(image, str):
            img = cv.imread(image, cv.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image: {image}")
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        elif isinstance(image, np.ndarray):
            img = image
            # 가능하면 복사 없이, 필요 시만 변환
            if img.ndim == 3 and img.shape[2] == 3:
                # 대개 BGR 이므로 RGB로 일괄 변환
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            else:
                img = img.copy()

        elif isinstance(image, torch.Tensor):
            npimg = image.detach().cpu().numpy()
            if npimg.ndim == 3:
                # CHW 또는 HWC를 모두 처리
                if npimg.shape[0] in (1, 3):  # CHW
                    npimg = np.transpose(npimg, (1, 2, 0))
                # float 범위 [0,1]일 수 있으니 0~255로 환산
                if npimg.dtype != np.uint8:
                    npimg = np.clip(npimg * (255.0 if npimg.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            img = npimg
            if img.ndim == 3 and img.shape[2] == 3:
                # 텐서에서 온 경우는 이미 RGB 확률이 높지만 안전하게 유지
                pass
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        return _ensure_uint8(img)

    @abstractmethod
    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Feature별 전처리 수행"""
        pass

    def preprocess_batch(self, images: List[Union[str, np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """배치 전처리 (리턴 그대로 torch.Tensor)"""
        return torch.stack([self.preprocess(im) for im in images])


class EdgePreprocessor(BasePreprocessor):
    """Edge feature 전처리 클래스"""

    # 엣지 강화 커널을 클래스 상수로 캐싱
    _SHARPEN_KERNEL = np.array(
        [[-1, -1, -1],
         [-1,  9, -1],
         [-1, -1, -1]],
        dtype=np.float32
    )

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
        super().__init__(target_size, normalize)
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.sobel_kernel = sobel_kernel
        self.laplacian_kernel = laplacian_kernel
        self.edge_method = edge_method
        self.edge_weight = edge_weight
        self.edge_enhancement = edge_enhancement

    @staticmethod
    def _to_gray(rgb_or_gray: np.ndarray) -> np.ndarray:
        return cv.cvtColor(rgb_or_gray, cv.COLOR_RGB2GRAY) if rgb_or_gray.ndim == 3 else rgb_or_gray

    def apply_canny(self, gray: np.ndarray) -> np.ndarray:
        """Canny edge detection (입력은 그레이스케일)"""
        return cv.Canny(gray, self.canny_low, self.canny_high)

    def apply_sobel(self, gray: np.ndarray) -> np.ndarray:
        """Sobel edge detection"""
        gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=self.sobel_kernel)
        gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=self.sobel_kernel)
        mag = cv.magnitude(gx, gy)
        # 0~255 정규화 후 uint8
        mag_u8 = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        return mag_u8

    def apply_laplacian(self, gray: np.ndarray) -> np.ndarray:
        """Laplacian edge detection (빠르고 안정적인 16S + convertScaleAbs)"""
        lap = cv.Laplacian(gray, cv.CV_16S, ksize=self.laplacian_kernel)
        return cv.convertScaleAbs(lap)

    def apply_edge_enhancement(self, rgb_img: np.ndarray) -> np.ndarray:
        return cv.filter2D(rgb_img, ddepth=-1, kernel=self._SHARPEN_KERNEL)

    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Edge feature 전처리"""
        from PIL import Image

        img = self.load_image(image)                  # RGB uint8 (H, W, 3) or gray
        gray = self._to_gray(img)                     # 1채널 uint8

        # 엣지 검출
        method = self.edge_method.lower()
        if method == 'canny':
            edges = self.apply_canny(gray)
        elif method == 'sobel':
            edges = self.apply_sobel(gray)
        elif method == 'laplacian':
            edges = self.apply_laplacian(gray)
        else:
            edges = self.apply_canny(gray)

        # 엣지 강화(선택)
        result = img if not self.edge_enhancement else self.apply_edge_enhancement(img)

        # 엣지 결합 (edges는 1채널 → 3채널로 브로드캐스트)
        edges_3d = np.repeat(edges[..., None], 3, axis=2)
        result = cv.addWeighted(result, 1.0 - self.edge_weight, edges_3d, self.edge_weight, 0.0)

        # torchvision 파이프라인 적용 (PIL로 변환)
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
        super().__init__(target_size, normalize)
        self.histogram_equalization = histogram_equalization
        self.noise_reduction = noise_reduction
        self.blur = blur
        self.blur_kernel = blur_kernel
        self.contrast_enhancement = contrast_enhancement
        self.brightness_adjustment = brightness_adjustment
        self.noise_factor = noise_factor

    @staticmethod
    def _eq_hist(rgb_or_gray: np.ndarray) -> np.ndarray:
        if rgb_or_gray.ndim == 3:
            yuv = cv.cvtColor(rgb_or_gray, cv.COLOR_RGB2YUV)
            yuv[..., 0] = cv.equalizeHist(yuv[..., 0])
            return cv.cvtColor(yuv, cv.COLOR_YUV2RGB)
        return cv.equalizeHist(rgb_or_gray)

    @staticmethod
    def _denoise(img: np.ndarray) -> np.ndarray:
        # bilateralFilter는 비용이 큼 → 유지하되 파라미터는 동일
        return cv.bilateralFilter(img, 9, 75, 75)

    def _add_noise(self, img: np.ndarray) -> np.ndarray:
        if self.noise_factor <= 0:
            return img
        # float32로 연산 후 클리핑 → uint8
        noise = np.random.normal(0.0, self.noise_factor * 255.0, img.shape).astype(np.float32)
        out = img.astype(np.float32) + noise
        return _ensure_uint8(out)

    def _blur(self, img: np.ndarray) -> np.ndarray:
        k = int(self.blur_kernel) | 1  # 홀수 보장
        return cv.GaussianBlur(img, (k, k), 0)

    @staticmethod
    def _contrast(rgb_or_gray: np.ndarray) -> np.ndarray:
        if rgb_or_gray.ndim == 3:
            lab = cv.cvtColor(rgb_or_gray, cv.COLOR_RGB2LAB)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[..., 0] = clahe.apply(lab[..., 0])
            return cv.cvtColor(lab, cv.COLOR_LAB2RGB)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(rgb_or_gray)

    def _adjust_brightness(self, img: np.ndarray) -> np.ndarray:
        b = float(self.brightness_adjustment)
        if abs(b) < 1e-8:
            return img
        # float으로 더한 뒤 안전하게 반환
        out = img.astype(np.float32) + (b * 255.0)
        return _ensure_uint8(out)

    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        from PIL import Image

        img = self.load_image(image)
        result = img

        if self.histogram_equalization:
            result = self._eq_hist(result)

        if self.noise_reduction:
            result = self._denoise(result)

        # 노이즈는 노이즈 제거 사용 중이 아닐 때만 (원래 로직 유지)
        if self.noise_factor > 0 and not self.noise_reduction:
            result = self._add_noise(result)

        if self.blur:
            result = self._blur(result)

        if self.contrast_enhancement:
            result = self._contrast(result)

        result = self._adjust_brightness(result)

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
        super().__init__(target_size, normalize)
        self.gabor_angles = gabor_angles
        self.gabor_freq = gabor_freq
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.use_gabor = use_gabor
        self.use_lbp = use_lbp
        self.texture_weight = texture_weight

        # Gabor 커널 미리 생성(속도 ↑)
        self._gabor_kernels = None
        if self.use_gabor:
            self._gabor_kernels = [
                cv.getGaborKernel(
                    ksize=(21, 21),
                    sigma=5.0,
                    theta=np.radians(ang),
                    lambd=self.gabor_freq,  # 기존 의미 유지
                    gamma=0.5,
                    psi=0
                ).astype(np.float32)
                for ang in self.gabor_angles
            ]

    @staticmethod
    def _to_gray(rgb_or_gray: np.ndarray) -> np.ndarray:
        return cv.cvtColor(rgb_or_gray, cv.COLOR_RGB2GRAY) if rgb_or_gray.ndim == 3 else rgb_or_gray

    def apply_gabor_filter(self, gray: np.ndarray) -> np.ndarray:
        # float32로 한 번만 캐스팅
        g = gray.astype(np.float32)
        acc = None
        for ker in self._gabor_kernels:
            filtered = cv.filter2D(g, cv.CV_32F, ker)
            acc = filtered if acc is None else (acc + filtered)
        acc /= float(len(self._gabor_kernels))
        acc = cv.normalize(acc, None, 0, 255, cv.NORM_MINMAX)
        return acc.astype(np.uint8)

    def apply_lbp(self, gray: np.ndarray) -> np.ndarray:
        """
        LBP (원래 로직과 동일한 근접 픽셀 샘플링/비트코딩) 벡터화 버전.
        - 8방향 이웃을 radius로 시프트해서 비교 후 비트 합성
        """
        r = int(self.lbp_radius)
        h, w = gray.shape
        if h < 2 * r + 1 or w < 2 * r + 1:
            # 너무 작으면 원본대로 0으로 채운 결과 반환
            return np.zeros_like(gray, dtype=np.uint8)

        center = gray
        lbp = np.zeros_like(gray, dtype=np.uint8)

        # 8개 방향의 정수 좌표 샘플링
        # (이전 구현은 float 좌표를 int()로 캐스팅 → 최근접 픽셀 취득이었으므로, 동일하게 정수 오프셋 기반 시프트로 구현)
        offsets = []
        for k in range(self.lbp_points):
            angle = 2 * np.pi * k / self.lbp_points
            dx = int(round(r * np.cos(angle)))
            dy = int(round(r * np.sin(angle)))
            offsets.append((dx, dy))

        # 가장자리 처리를 위해 복제 패딩
        pad = r
        padded = cv.copyMakeBorder(center, pad, pad, pad, pad, borderType=cv.BORDER_REPLICATE)

        # 시프트하여 비교
        for bit, (dx, dy) in enumerate(offsets):
            xs = pad + dy  # 행 방향(위/아래) 이동: y
            ys = pad + dx  # 열 방향(좌/우) 이동: x
            nbr = padded[xs:xs + h, ys:ys + w]
            lbp |= ((nbr >= center).astype(np.uint8) << bit)

        # 테두리는 원래 for-loop 구현도 중심 인덱스 범위 밖에는 0으로 남겼음 → 유지
        if r > 0:
            lbp[:r, :] = 0
            lbp[-r:, :] = 0
            lbp[:, :r] = 0
            lbp[:, -r:] = 0

        return lbp

    def preprocess(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        from PIL import Image

        img = self.load_image(image)
        gray = self._to_gray(img)
        result = img

        if self.use_gabor:
            gabor = self.apply_gabor_filter(gray)
            gabor_3d = np.repeat(gabor[..., None], 3, axis=2)
            result = cv.addWeighted(result, 1.0 - self.texture_weight, gabor_3d, self.texture_weight, 0.0)

        if self.use_lbp:
            lbp = self.apply_lbp(gray)
            lbp_3d = np.repeat(lbp[..., None], 3, axis=2)
            result = cv.addWeighted(result, 1.0 - self.texture_weight, lbp_3d, self.texture_weight, 0.0)

        pil_image = Image.fromarray(result)
        return self.torch_transform(pil_image)

class TemporalWindowPreprocessor(BasePreprocessor):
    """
    입력 형태에 따라 자동 동작하는 시퀀스 전처리기.
      - List[str] (프레임 경로 리스트)          -> [Seq_len, Frames_per_step, C, H, W]
      - List[np.ndarray | torch.Tensor]        -> [Seq_len, Frames_per_step, C, H, W]
      - 단일 프레임(str/ndarray/tensor)        -> [1,      Frames_per_step, C, H, W]
    pad:
      - 'edge' : 경계를 가장 가까운 프레임으로 복제
      - 'zero' : 0으로 패딩
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        frames_per_step: int = 3,
        stride: int = 1,
        pad: str = "edge",     # 'edge' | 'zero'
        normalize: bool = True
    ):
        super().__init__(target_size, normalize)
        assert frames_per_step >= 1 and frames_per_step % 2 == 1, \
            "frames_per_step는 1 이상의 홀수여야 합니다. (예: 3,5,7)"
        self.frames_per_step = int(frames_per_step)
        self.stride = max(1, int(stride))
        self.pad = pad

    # === 핵심: preprocess가 시퀀스를 통째로 만들어서 반환 ===
    def preprocess(
        self,
        image: Union[
            str, np.ndarray, torch.Tensor,
            List[Union[str, np.ndarray, torch.Tensor]]
        ]
    ) -> torch.Tensor:
        """
        리스트가 오면 시퀀스로 간주하여 [Seq_len, F, C, H, W] 반환.
        단일 프레임이면 그 프레임을 중심으로 한 1-step 시퀀스 [1, F, C, H, W] 반환.
        """
        if isinstance(image, (list, tuple)):
            if len(image) == 0:
                raise ValueError("Empty sequence passed to TemporalWindowPreprocessor.preprocess()")
            if isinstance(image[0], str):
                return self._preprocess_paths(list(image))
            else:
                return self._preprocess_frames(list(image))
        # 단일 프레임
        return self._preprocess_frames([image])

    # === 배치 지원: 항목별 시퀀스를 제로패딩으로 맞춰 [B, S_max, F, C, H, W] ===
    def preprocess_batch(
        self,
        items: List[
            Union[
                str, np.ndarray, torch.Tensor,
                List[Union[str, np.ndarray, torch.Tensor]]
            ]
        ]
    ) -> torch.Tensor:
        seq_list: List[torch.Tensor] = [self.preprocess(x) for x in items]  # 각자 [S_i,F,C,H,W]
        S_max = max(t.shape[0] for t in seq_list)
        B = len(seq_list)
        F, C, H, W = seq_list[0].shape[1:]
        out = seq_list[0].new_zeros((B, S_max, F, C, H, W))
        for i, t in enumerate(seq_list):
            S = t.shape[0]
            out[i, :S] = t
        return out

    # === 내부 유틸 ===
    def _load_rgb_uint8_from_path(self, path: str) -> np.ndarray:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

    def _to_tensor(self, arr_or_tensor: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """RGB uint8 ndarray(혹은 tensor)를 [C,H,W]로 변환하고 Resize/Normalize."""
        from PIL import Image
        if torch.is_tensor(arr_or_tensor):
            x = arr_or_tensor.detach().cpu()
            if x.ndim == 3 and x.shape[0] in (1, 3):        # CHW
                chw = x
            elif x.ndim == 3 and x.shape[2] in (1, 3):      # HWC
                chw = x.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")

            if chw.dtype != torch.uint8:
                y = chw
                y = (y * 255.0).clamp(0, 255) if y.max() <= 1.0 else y.clamp(0, 255)
                y = y.to(torch.uint8)
                hwc = y.permute(1, 2, 0).numpy()
                return self.torch_transform(Image.fromarray(hwc))
            else:
                hwc = chw.permute(1, 2, 0).numpy()
                return self.torch_transform(Image.fromarray(hwc))

        # ndarray
        arr = np.asarray(arr_or_tensor)
        if arr.ndim == 2:  # grayscale → RGB
            arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported ndarray shape: {arr.shape}")
        return self.torch_transform(Image.fromarray(_ensure_uint8(_to_rgb_np(arr))))

    def _get_index(self, i: int, n: int) -> int:
        if 0 <= i < n:
            return i
        if self.pad == "edge":
            return max(0, min(i, n - 1))
        return -1  # zero pad

    # === 경로 리스트 → 시퀀스 ===
    def _preprocess_paths(self, frame_paths: List[str]) -> torch.Tensor:
        """
        frame_paths: 정렬된 경로 리스트
        return: [Seq_len, Frames_per_step, C, H, W]
        """
        from PIL import Image
        n = len(frame_paths)
        half = self.frames_per_step // 2
        seq_list: List[torch.Tensor] = []
        cache: Dict[int, torch.Tensor] = {}

        def load_as_tensor(idx: int) -> torch.Tensor:
            if idx in cache:
                return cache[idx]
            if idx < 0 or idx >= n:
                if self.pad == "zero":
                    t0 = next(iter(cache.values())) if cache else self.torch_transform(
                        Image.fromarray(self._load_rgb_uint8_from_path(frame_paths[0]))
                    )
                    return torch.zeros_like(t0)
            arr = self._load_rgb_uint8_from_path(frame_paths[idx])
            t = self.torch_transform(Image.fromarray(arr))  # [C,H,W]
            cache[idx] = t
            return t

        i = 0
        while i < n:
            window: List[torch.Tensor] = []
            for off in range(-half, half + 1):
                j = i + off
                jj = self._get_index(j, n)
                if jj == -1:  # zero pad
                    t = load_as_tensor(0)
                    window.append(torch.zeros_like(t))
                else:
                    window.append(load_as_tensor(jj))
            seq_list.append(torch.stack(window, dim=0))  # [F,C,H,W]
            i += self.stride

        return torch.stack(seq_list, dim=0)  # [Seq_len,F,C,H,W]

    # === ndarray/tensor 리스트 → 시퀀스 ===
    def _preprocess_frames(self, frames: List[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
        """
        frames: ndarray/tensor 리스트 (정렬된 순서)
        return: [Seq_len, Frames_per_step, C, H, W]
        """
        n = len(frames)
        half = self.frames_per_step // 2

        cache: Dict[int, torch.Tensor] = {}

        def get_tensor(idx: int) -> torch.Tensor:
            if idx in cache:
                return cache[idx]
            t = self._to_tensor(frames[idx])  # [C,H,W]
            cache[idx] = t
            return t

        template = get_tensor(0)
        seq_list: List[torch.Tensor] = []

        i = 0
        while i < n:
            window: List[torch.Tensor] = []
            for off in range(-half, half + 1):
                j = i + off
                if 0 <= j < n:
                    window.append(get_tensor(j))
                else:
                    if self.pad == "edge":
                        jj = 0 if j < 0 else n - 1
                        window.append(get_tensor(jj))
                    else:
                        window.append(torch.zeros_like(template))
            seq_list.append(torch.stack(window, dim=0))  # [F,C,H,W]
            i += self.stride

        return torch.stack(seq_list, dim=0)  # [Seq_len,F,C,H,W]