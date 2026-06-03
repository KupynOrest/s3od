#!/usr/bin/env python3
import argparse
from pathlib import Path
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from synth_sod.model_training.metrics import EvaluationMetrics
from s3od import BackgroundRemoval


DATASET_GROUPS = {
    "dis": ["DIS-TE1", "DIS-TE2", "DIS-TE3", "DIS-TE4"],
    "sod": ["DUTS-TE", "DUT-OMRON", "HRSOD-TE", "UHRSD-TE", "DAVIS-S"],
    "cod": ["CAMO-TE", "NC4K", "COD10K-TE"],
}
DATASET_GROUPS["all"] = DATASET_GROUPS["dis"] + DATASET_GROUPS["sod"] + DATASET_GROUPS["cod"]

METRIC_HEADERS = ["MAE↓", "maxF↑", "avgF↑", "wF↑", "Sm↑", "Em↑"]
METRIC_KEYS    = ["MAE", "MaxF", "AvgF", "wF", "Sm", "Em"]


def resolve_datasets(spec: str) -> list[str]:
    if spec in DATASET_GROUPS:
        return DATASET_GROUPS[spec]
    return [d.strip() for d in spec.split(",")]


def find_gt_mask(image_path: Path, dataset_dir: Path) -> Optional[Path]:
    masks_dir = dataset_dir / "masks"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = masks_dir / (image_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def select_oracle_mask(all_masks: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    gt_bool = gt_mask > 0.5
    best_iou, best_idx = -1.0, 0
    for i, mask in enumerate(all_masks):
        mask_bool = mask > 0.5
        intersection = np.logical_and(mask_bool, gt_bool).sum()
        union = np.logical_or(mask_bool, gt_bool).sum()
        iou = float(intersection) / float(union) if union > 0 else 1.0
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return all_masks[best_idx]


def evaluate_dataset(
    dataset_dir: Path,
    predictor: BackgroundRemoval,
    device: str,
    oracle: bool,
) -> dict:
    image_paths = sorted(glob(str(dataset_dir / "images" / "*")))
    metrics = EvaluationMetrics(device=device)
    oracle_metrics = EvaluationMetrics(device=device) if oracle else None

    for path_str in tqdm(image_paths, desc=dataset_dir.name, leave=False):
        image_path = Path(path_str)
        gt_path = find_gt_mask(image_path, dataset_dir)
        if gt_path is None:
            continue

        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        result = predictor.remove_background(image)

        gt_mask = (cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 128).astype(np.float32)
        gt_tensor = torch.from_numpy(gt_mask).to(device)

        metrics.step(
            torch.from_numpy(result.predicted_mask.astype(np.float32)).to(device),
            gt_tensor.clone(),
        )

        if oracle_metrics is not None:
            oracle_soft = select_oracle_mask(result.all_masks, gt_mask)
            oracle_metrics.step(
                torch.from_numpy(oracle_soft.astype(np.float32)).to(device),
                gt_tensor.clone(),
            )

    return {
        "pred": metrics.compute_metrics(),
        **({"oracle": oracle_metrics.compute_metrics()} if oracle_metrics else {}),
    }


def print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    col_w, name_w = 10, 12
    sep = "-" * (name_w + col_w * len(METRIC_HEADERS))
    print(f"\n{title}")
    print(sep)
    print(f"{'Dataset':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in METRIC_HEADERS))
    print(sep)
    for name, m in rows:
        vals = "".join(f"{m.get(k, float('nan')):>{col_w}.4f}" for k in METRIC_KEYS)
        print(f"{name:<{name_w}}{vals}")
    if len(rows) > 1:
        print(sep)
        mean = {k: float(np.mean([m.get(k, float('nan')) for _, m in rows])) for k in METRIC_KEYS}
        vals = "".join(f"{mean[k]:>{col_w}.4f}" for k in METRIC_KEYS)
        print(f"{'Mean':<{name_w}}{vals}")
    print(sep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official S3OD evaluation on SOD/DIS benchmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_id", default="okupyn/s3od")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--datasets", default="dis",
                        help="'dis', 'sod', 'cod', 'all', or comma-separated dataset names")
    parser.add_argument("--oracle", action="store_true",
                        help="Also report GT-oracle upper-bound metrics")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    datasets = resolve_datasets(args.datasets)

    predictor = BackgroundRemoval(
        model_id=args.model_id, image_size=args.image_size, device=device
    )
    print(f"Model    : {args.model_id}")
    print(f"Device   : {device}  |  Image size: {args.image_size}")
    print(f"Datasets : {', '.join(datasets)}")

    pred_rows: list[tuple[str, dict]] = []
    oracle_rows: list[tuple[str, dict]] = []

    for dataset in datasets:
        dataset_dir = data_dir / dataset
        if not dataset_dir.exists():
            print(f"  [skip] {dataset_dir} not found")
            continue
        results = evaluate_dataset(dataset_dir, predictor, device, args.oracle)
        pred_rows.append((dataset, results["pred"]))
        if "oracle" in results:
            oracle_rows.append((dataset, results["oracle"]))
        print_table(dataset, [(dataset, results["pred"])])
        if "oracle" in results:
            print_table(f"{dataset} (oracle)", [(dataset, results["oracle"])])

    if len(pred_rows) > 1:
        print_table("Summary — Model metrics (IoU-head selection)", pred_rows)
        if oracle_rows:
            print_table("Summary — Oracle metrics (GT-IoU upper bound)", oracle_rows)


if __name__ == "__main__":
    main()
