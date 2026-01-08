import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
VAL_LABELS = ROOT / "Cracks-main" / "output" / "dataset" / "labels" / "val"
PRED_LABELS = ROOT / "Cracks-main" / "output" / "yolo" / "train_preds" / "labels"
OUT_DIR = ROOT / "Cracks-main" / "output" / "presentation"


def yolo_to_xyxy_norm(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def iou_box(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_yolo_labels(path: Path) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            parts = line.strip().split()
            # Expect: class cx cy w h (conf optional not saved in our run)
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = map(float, (cx, cy, w, h))
            boxes.append(yolo_to_xyxy_norm(cx, cy, w, h))
    except Exception:
        pass
    return boxes


def greedy_match_iou(gt: List[Tuple[float, float, float, float]],
                     pr: List[Tuple[float, float, float, float]],
                     iou_thr: float = 0.5) -> Tuple[int, int, int]:
    """Return TP, FP, FN counts using greedy IoU matching."""
    used_pred = set()
    tp = 0
    for gi, g in enumerate(gt):
        best_iou, best_pi = 0.0, -1
        for pi, p in enumerate(pr):
            if pi in used_pred:
                continue
            iou = iou_box(g, p)
            if iou > best_iou:
                best_iou, best_pi = iou, pi
        if best_iou >= iou_thr and best_pi >= 0:
            tp += 1
            used_pred.add(best_pi)
    fp = len(pr) - len(used_pred)
    fn = len(gt) - tp
    return tp, fp, fn


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stems = {p.stem for p in VAL_LABELS.glob('*.txt')}
    image_level = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    total_tp = total_fp = total_fn = 0

    for stem in sorted(stems):
        gt_file = VAL_LABELS / f"{stem}.txt"
        pr_file = PRED_LABELS / f"{stem}.txt"
        gt = load_yolo_labels(gt_file)
        pr = load_yolo_labels(pr_file)

        tp, fp, fn = greedy_match_iou(gt, pr, iou_thr=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Image-level confusion
        gt_pos = len(gt) > 0
        pr_pos = len(pr) > 0
        if gt_pos and pr_pos and tp > 0:
            image_level["TP"] += 1
        elif (not gt_pos) and pr_pos:
            image_level["FP"] += 1
        elif gt_pos and (not pr_pos):
            image_level["FN"] += 1
        else:
            image_level["TN"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "object_level": {"TP": total_tp, "FP": total_fp, "FN": total_fn},
        "image_level": image_level,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # Save metrics JSON
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Plot confusion matrix (image-level)
    cm = [
        [image_level["TP"], image_level["FP"]],
        [image_level["FN"], image_level["TN"]],
    ]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Matrice de confusion (image)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_xticks([0, 1], ["Pos.", "Neg."])
    ax.set_yticks([0, 1], ["Pos.", "Neg."])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    print("✓ Metrics saved:", OUT_DIR / "metrics.json")
    print("✓ Confusion matrix saved:", OUT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
