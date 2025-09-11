# test.py
import sys
import time
import os
import csv
import random
import numpy as np
import torch

from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet

# 🔥 Grad-CAM 유틸
from xai_gradcam import save_cams_for_folder
# (선택) 전처리 유틸을 쓰고 있다면 유지. 미사용이면 삭제해도 무방.
try:
    from data.datasets import texture_transform, edge_transform, sharpen_transform  # noqa: F401
except Exception:
    pass

# ==========================
# 옵션 파싱
# ==========================
opt = TestOptions().parse(print_options=False)
print(f"Model_path {opt.model_path}")

# transform_mode가 옵션에 없을 수 있으니 안전 처리
TRANSFORM_MODE = getattr(opt, "transform_mode", "default") or "default"

# ==========================
# 공통 설정
# ==========================
BASE_DATAROOT     = "/data/Deepfake_train_val/test"
NO_RESIZE_DEFAULT = False
NO_CROP_DEFAULT   = True

# Grad-CAM 설정
DO_GRADCAM    = True
PER_CLASS_MAX = 2000   # 각 클래스당 저장할 최대 개수 (0_fake / 0_real 등)
TOTAL_MAX     = 5000   # 클래스 폴더가 없을 때 총 저장할 최대 개수
GRADCAM_DIR   = "./runs/gradcam"
SHUFFLE_SEED  = 0    # 무작위 셔플 시드(재현 원하면 고정, 매회 랜덤하려면 None)

# 리포트 설정
REPORT_DIR   = "./runs/reports"
THRESHOLD    = 0.5            # y_pred >= THRESHOLD → fake(1)로 판정
PREFIX       = "deepfake_test"
TIMESTR      = time.strftime("%Y%m%d_%H%M%S", time.localtime())

# 하위 폴더(변환 모드) 보장
REPORT_SUBDIR = os.path.join(REPORT_DIR, TRANSFORM_MODE)
os.makedirs(REPORT_SUBDIR, exist_ok=True)
CSV_PATH = os.path.join(REPORT_SUBDIR, f"{PREFIX}_{TIMESTR}_report.csv")

# ==========================
# 재현성
# ==========================
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

# ==========================
# 메트릭/리포트 유틸
# ==========================
from sklearn.metrics import confusion_matrix

def _safe_div(a, b):
    return float(a) / float(b) if b else 0.0

def compute_metrics_from_probs(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5):
    """
    이진 분류 (0=real, 1=fake) 가정.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_prob = np.asarray(y_pred_prob).astype(float)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # labels=[0,1] 고정
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    total = tn + fp + fn + tp
    acc   = _safe_div(tn + tp, total)

    precision   = _safe_div(tp, tp + fp)          # PPV
    recall      = _safe_div(tp, tp + fn)          # TPR
    f1          = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    specificity = _safe_div(tn, tn + fp)          # TNR
    fpr         = _safe_div(fp, fp + tn)          # False Positive Rate
    fnr         = _safe_div(fn, fn + tp)          # False Negative Rate

    # 클래스별 정확도 (참고용)
    real_acc = _safe_div(tn, tn + fp) if (tn + fp) > 0 else 0.0
    fake_acc = _safe_div(tp, tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "total": total, "acc": acc,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": specificity, "fpr": fpr, "fnr": fnr,
        "real_acc": real_acc, "fake_acc": fake_acc
    }

def save_report_csv(rows, csv_path):
    """
    rows: 리스트[{
      'subset': str,
      'acc': float, 'ap': float,
      'precision': float, 'recall': float, 'f1': float,
      'specificity': float, 'fpr': float, 'fnr': float,
      'real_acc': float, 'fake_acc': float,
      'tn': int, 'fp': int, 'fn': int, 'tp': int, 'total': int
    }]
    """
    # 상위 디렉터리까지 보장
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 집계
    total_tn = sum(r["tn"] for r in rows)
    total_fp = sum(r["fp"] for r in rows)
    total_fn = sum(r["fn"] for r in rows)
    total_tp = sum(r["tp"] for r in rows)
    total_n  = sum(r["total"] for r in rows)

    # 가중 평균(Acc) / 단순 평균(AP)
    mean_acc = np.average([r["acc"] for r in rows], weights=[r["total"] for r in rows]) if total_n > 0 else 0.0
    mean_ap  = float(np.mean([r["ap"] for r in rows])) if rows else 0.0

    g_precision   = _safe_div(total_tp, total_tp + total_fp)
    g_recall      = _safe_div(total_tp, total_tp + total_fn)
    g_f1          = _safe_div(2 * g_precision * g_recall, g_precision + g_recall) if (g_precision + g_recall) > 0 else 0.0
    g_specificity = _safe_div(total_tn, total_tn + total_fp)
    g_fpr         = _safe_div(total_fp, total_fp + total_tn)
    g_fnr         = _safe_div(total_fn, total_fn + total_tp)
    g_real_acc    = _safe_div(total_tn, total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    g_fake_acc    = _safe_div(total_tp, total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Subset","Acc(%)","AP(%)","Precision","Recall","F1",
            "Specificity","FPR","FNR","Real_Acc","Fake_Acc",
            "TN","FP","FN","TP","Total","Threshold"
        ])
        for r in rows:
            w.writerow([
                r["subset"],
                f"{r['acc']*100:.2f}", f"{r['ap']*100:.2f}",
                f"{r['precision']:.6f}", f"{r['recall']:.6f}", f"{r['f1']:.6f}",
                f"{r['specificity']:.6f}", f"{r['fpr']:.6f}", f"{r['fnr']:.6f}",
                f"{r['real_acc']:.6f}", f"{r['fake_acc']:.6f}",
                r["tn"], r["fp"], r["fn"], r["tp"], r["total"],
                r.get("threshold", THRESHOLD),
            ])
        # 글로벌/평균 요약 행
        w.writerow([
            "MEAN/GLOBAL",
            f"{mean_acc*100:.2f}", f"{mean_ap*100:.2f}",
            f"{g_precision:.6f}", f"{g_recall:.6f}", f"{g_f1:.6f}",
            f"{g_specificity:.6f}", f"{g_fpr:.6f}", f"{g_fnr:.6f}",
            f"{g_real_acc:.6f}", f"{g_fake_acc:.6f}",
            total_tn, total_fp, total_fn, total_tp, total_n,
            THRESHOLD
        ])

    print(f"[report] CSV 저장 완료 → {csv_path}")
    print(f"[report] Mean Acc: {mean_acc*100:.2f}%, Mean AP: {mean_ap*100:.2f}%")

# ==========================
# 모델 로딩
# ==========================
model = resnet50(num_classes=1)

# --- robust state_dict loader ---
ckpt = torch.load(opt.model_path, map_location="cpu")
if isinstance(ckpt, dict):
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    else:
        sd = ckpt
else:
    sd = ckpt

new_sd = {}
for k, v in sd.items():
    nk = k[7:] if k.startswith("module.") else k
    new_sd[nk] = v

# fc → fc1 키 스왑 대응
if any(k.startswith("fc.") for k in new_sd.keys()) and not any(k.startswith("fc1.") for k in new_sd.keys()):
    tmp = {}
    for k, v in new_sd.items():
        if k.startswith("fc."):
            tmp["fc1." + k[len("fc."):]] = v
        else:
            tmp[k] = v
    new_sd = tmp

missing, unexpected = model.load_state_dict(new_sd, strict=False)
if missing or unexpected:
    print("[load_state_dict] warning")
    if missing:
        print("  missing:", missing)
    if unexpected:
        print("  unexpected:", unexpected)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ==========================
# 테스트 루프
# ==========================
if not os.path.isdir(BASE_DATAROOT):
    raise FileNotFoundError(f"[test] dataroot not found: {BASE_DATAROOT}")

subsets = sorted([d for d in os.listdir(BASE_DATAROOT)
                  if os.path.isdir(os.path.join(BASE_DATAROOT, d))])

printSet(os.path.basename(BASE_DATAROOT))
print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

rows = []  # detailed rows for CSV

for v_id, val in enumerate(subsets):
    opt.dataroot  = os.path.join(BASE_DATAROOT, val)
    opt.classes   = ''
    opt.no_resize = NO_RESIZE_DEFAULT
    opt.no_crop   = NO_CROP_DEFAULT

    # 평가 (validate는 acc, ap, r_acc, f_acc, y_true, y_pred 반환)
    acc, ap, r_acc, f_acc, y_true, y_pred = validate(model, opt)
    print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc * 100, ap * 100))

    # confusion 기반 상세 메트릭 계산
    m = compute_metrics_from_probs(y_true=np.array(y_true),
                                   y_pred_prob=np.array(y_pred),
                                   threshold=THRESHOLD)
    # validate에서 구한 acc/ap 사용 (일관성 유지)
    m["acc"]    = acc
    m["ap"]     = ap
    m["subset"] = val
    m["threshold"] = THRESHOLD
    rows.append(m)

    # ===== Grad-CAM 저장 (클래스별) =====
    if DO_GRADCAM:
        class_dirs = []
        for d in sorted(os.listdir(opt.dataroot)):
            full = os.path.join(opt.dataroot, d)
            if os.path.isdir(full):
                class_dirs.append((d, full))

        if not class_dirs:
            # 클래스 디렉토리가 없는 구조면 해당 폴더에서 TOTAL_MAX 랜덤 저장
            out_dir_root = os.path.join(
                GRADCAM_DIR, os.path.basename(BASE_DATAROOT), val, TRANSFORM_MODE
            )
            os.makedirs(out_dir_root, exist_ok=True)
            try:
                save_cams_for_folder(
                    model=model,
                    folder=opt.dataroot,
                    out_dir=out_dir_root,
                    no_resize=opt.no_resize,
                    no_crop=opt.no_crop,
                    max_images=TOTAL_MAX,
                    shuffle_seed=SHUFFLE_SEED,
                    opt=opt,
                )
            except Exception as e:
                print(f"[xai] Grad-CAM 실패 ({val}): {e}")
        else:
            # 각 클래스 폴더에서 PER_CLASS_MAX 랜덤 저장
            for cls_name, cls_path in class_dirs:
                out_dir_cls = os.path.join(
                    GRADCAM_DIR, os.path.basename(BASE_DATAROOT), val, TRANSFORM_MODE, cls_name
                )
                os.makedirs(out_dir_cls, exist_ok=True)
                try:
                    save_cams_for_folder(
                        model=model,
                        folder=cls_path,          # ex) /test/human/0_fake
                        out_dir=out_dir_cls,      # ex) ./runs/gradcam/test/human/<mode>/0_fake
                        no_resize=opt.no_resize,
                        no_crop=opt.no_crop,
                        max_images=PER_CLASS_MAX,
                        shuffle_seed=SHUFFLE_SEED,
                        opt=opt,
                    )
                except Exception as e:
                    print(f"[xai] Grad-CAM 실패 ({val}/{cls_name}): {e}")

# ==========================
# 리포트 저장(CSV)
# ==========================
save_report_csv(rows, CSV_PATH)

# 콘솔 요약
mean_acc = np.average([r["acc"] for r in rows], weights=[r["total"] for r in rows]) if rows else 0.0
mean_ap  = np.mean([r["ap"] for r in rows]) if rows else 0.0
print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(len(subsets), 'Mean', mean_acc * 100, mean_ap * 100))
print('*' * 25)