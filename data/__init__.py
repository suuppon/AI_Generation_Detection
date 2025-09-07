# data/__init__.py
import os
import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import (
    ProcessorBuilder,
    MultiProcVideoDataset,
    collate_concat,
    collate_dict,
)

def _pick_names(classes):
    """
    dataroot 바로 아래에 다음 중 하나 조합이 있으면 사용:
      - ('0_real', '1_fake')  또는  ('real', 'fake')
    """
    s = set(classes)
    if "0_real" in s and "1_fake" in s:
        return "0_real", "1_fake"
    if "real" in s and "fake" in s:
        return "real", "fake"
    raise RuntimeError(
        f"Expected 'real' & 'fake' (or '0_real' & '1_fake') under dataroot, but found: {sorted(list(classes))}"
    )

def get_dataset(opt,mode):
    """
    dataroot/
      real/ vid*/frame.png ...
      fake/ vid*/frame.png ...
    """
    if mode == "val":
        dataroot = opt.val_dataroot
    else:
        dataroot = opt.dataroot

    classes = os.listdir(dataroot)
    real_name, fake_name = _pick_names(classes)

    # Processor / 옵션 적용
    features = getattr(opt, "features", ["edge", "texture"])
    proc_config = getattr(opt, "proc_config", None)
    proc_mode = getattr(opt, "proc_mode", "dict")   # dict 권장
    max_videos_per_class = getattr(opt, "max_videos_per_class", None)
    sort_frames_numeric = getattr(opt, "sort_frames_numeric", True)
    frame_max    = getattr(opt, "frame_max", None)
    frame_stride = getattr(opt, "frame_stride", 1)
    frame_start  = getattr(opt, "frame_start", 0)

    builder = ProcessorBuilder(config_path=proc_config, features=features)

    dset = MultiProcVideoDataset(
        fake_dir=os.path.join(opt.dataroot, fake_name),
        real_dir=os.path.join(opt.dataroot, real_name),
        processor=builder,
        mode=proc_mode,
        max_videos_per_class=max_videos_per_class,
        sort_frames_numeric=sort_frames_numeric,
        frame_max=frame_max,
        frame_stride=frame_stride,
        frame_start=frame_start,
    )
    dset._is_multi_proc = True
    dset._proc_mode = proc_mode
    return dset

def get_bal_sampler(dataset):
    if not getattr(dataset, "_is_multi_proc", False):
        raise TypeError("get_bal_sampler() supports only MultiProcVideoDataset.")
    targets = [lbl for _, lbl in dataset.index]  # (vdir,label)
    ratio = np.bincount(np.asarray(targets, dtype=np.int64))
    if (ratio <= 0).any():
        raise ValueError(f"Invalid class distribution: {ratio.tolist()}")
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[torch.tensor(targets, dtype=torch.long)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

def create_dataloader(opt,mode):
    dataset = get_dataset(opt,mode)
    shuffle = not getattr(opt, "serial_batches", False) if (getattr(opt, "isTrain", True) and not getattr(opt, "class_bal", False)) else False
    sampler = get_bal_sampler(dataset) if getattr(opt, "class_bal", False) else None
    proc_mode = getattr(dataset, "_proc_mode", "dict")
    collate_fn = collate_dict if (proc_mode == "dict") else collate_concat

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=getattr(opt, "batch_size", 1),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=int(getattr(opt, "num_threads", 4)),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )