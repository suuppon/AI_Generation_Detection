import os
import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

ImageFile.LOAD_TRUNCATED_IMAGES = True

SAVE_COUNT = 0
MAX_SAVE = 20

def save_with_original(tensor, pil_img, path, mode="texture"):
    """
    tensor: transformed tensor [C,H,W]
    pil_img: 원본 PIL.Image
    path: 원래 파일 경로
    mode: 변환 모드 (texture, edge, sharpen 등)
    """
    global SAVE_COUNT
    if SAVE_COUNT >= MAX_SAVE:
        return tensor

    # Tensor → PIL 변환
    transformed = transforms.ToPILImage()(tensor.cpu().clone().detach())

    # 크기 맞추기 (원본/변환 이미지 크기 동일화)
    W, H = pil_img.size
    transformed = transformed.resize((W, H))

    # 두 이미지 붙이기 (좌우)
    combined = Image.new("RGB", (W * 2, H))
    combined.paste(pil_img, (0, 0))
    combined.paste(transformed, (W, 0))

    # 저장 디렉토리
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join("samples", base)
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"{mode}_comparison.png")
    combined.save(save_path)

    print(f"✅ 샘플 저장: {save_path}")
    SAVE_COUNT += 1
    return tensor

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # ✅ (image, label, path)로 반환
        return img, target, path

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')

def texture_transform(img: Image.Image):
    """간단한 텍스처 부각 (Sobel magnitude)"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    mag = mag.astype(np.uint8)
    # 다시 PIL.Image로 변환해서 pipeline 이어가기
    mag_rgb = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(mag_rgb)

def edge_transform(img: Image.Image):
    """간단한 엣지 부각 (Canny edge)"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    # [H,W] → [H,W,3] (흑백을 3채널로 확장해서 파이프라인 호환)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def sharpen_transform(img: Image.Image):
    arr = np.array(img)
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(sharp)

# def lbp_transform(img: Image.Image, P=8, R=1):
#     gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
#     lbp = local_binary_pattern(gray, P, R, method="uniform")
#     lbp = (lbp / lbp.max() * 255).astype(np.uint8)
#     lbp_rgb = cv2.cvtColor(lbp, cv2.COLOR_GRAY2RGB)
#     return Image.fromarray(lbp_rgb)

def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))

    if opt.transform_mode == 'texture':
        additional_transform = texture_transform
    elif opt.transform_mode == 'edge':
        additional_transform = edge_transform
    elif opt.transform_mode == 'sharpen':
        additional_transform = sharpen_transform
    # elif opt.transform_mode == 'lbp':
    #     additional_transform = lbp_transform
    else:
        raise NotImplementedError(f'Unknown transform_mode: {opt.transform_mode}')
    
    dset = ImageFolderWithPaths(root,
                                transform=transforms.Compose([
                                    rz_func,
                                    crop_func,
                                    flip_func,
                                    additional_transform,
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],
                                                         std=[0.229,0.224,0.225])
                                ]))
    dset.transform_mode = opt.transform_mode
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
