from collections import defaultdict
import datetime
import numpy as np
import json
import os
import cv2
import torch
from tqdm import tqdm
from fire import Fire
from typing import Dict, List, Tuple

from model_training.predictor import SODPredictor
from model_training.metrics import EvaluationMetrics


def eval_sample(predictor: SODPredictor, image: np.ndarray, gt_mask: np.ndarray,
               metric_counter: EvaluationMetrics) -> float:
    """Evaluate single image with flips and consistency"""
    result_orig = predictor.predict(image)
    soft_mask_orig = result_orig.soft_mask
    
    flipped_image = cv2.flip(image, 1)
    result_flipped = predictor.predict(flipped_image)
    soft_mask_flipped = cv2.flip(result_flipped.soft_mask, 1)

    soft_mask_orig = torch.tensor(soft_mask_orig).cuda()
    soft_mask_flipped = torch.tensor(soft_mask_flipped).cuda()
    gt_mask = torch.tensor(gt_mask).cuda()

    # Evaluate original prediction
    metric_counter.step(soft_mask_orig, gt_mask)
    s_orig = metric_counter.compute_metrics()['Sm']
    metric_counter.reset()
    
    # Evaluate flipped prediction
    metric_counter.step(soft_mask_flipped, gt_mask)
    s_flip = metric_counter.compute_metrics()['Sm']
    metric_counter.reset()
    
    # Evaluate consistency
    metric_counter.step(soft_mask_orig, soft_mask_flipped)
    s_cons = metric_counter.compute_metrics()['Sm']
    metric_counter.reset()
    
    if np.isnan(s_cons):
        s_cons = (s_orig + s_flip) / 2
    return (s_orig + s_flip) * s_cons / 2


def eval_category(predictor: SODPredictor, category_images: List[str], metric_counter: EvaluationMetrics) -> Tuple[
    float, List[float]]:
    """Evaluate category and return mean score and per-sample scores"""
    scores = []
    for image_path in tqdm(category_images, desc="Processing images", leave=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt_mask = cv2.imread(image_path.replace('images', 'masks').replace('.jpg', '.png'),
                             cv2.IMREAD_GRAYSCALE) / 255.0
        score = eval_sample(predictor, image, gt_mask, metric_counter)
        if np.isnan(score):
            print(f"NaN score for {image_path}")
            continue
        scores.append(eval_sample(predictor, image, gt_mask, metric_counter))
    return np.mean(scores), scores


def calculate_new_samples(category_scores: Dict[str, float], min_samples: int = 10, max_samples: int = 50,
                          high_threshold: float = 0.95, low_threshold: float = 0.8) -> Dict[str, int]:
    """
    Calculate number of new samples with aggressive scaling for low-performing categories.

    Args:
        category_scores: Dictionary of category scores
        min_samples: Minimum number of samples to generate
        max_samples: Maximum number of samples to generate
        high_threshold: Score threshold above which categories get minimal samples
        low_threshold: Score threshold below which categories get scaled aggressively
    """
    scores = np.array(list(category_scores.values()))

    difficulties = np.zeros_like(scores)

    for i, score in enumerate(scores):
        if score >= high_threshold:
            difficulties[i] = 0.1
        elif score <= low_threshold:
            difficulties[i] = 0.7 + 0.3 * (low_threshold - score) / low_threshold
        else:
            difficulties[i] = 0.1 + 0.6 * (high_threshold - score) / (high_threshold - low_threshold)

    scaled = 1 / (1 + np.exp(-8 * (difficulties - 0.5)))
    samples = min_samples + (max_samples - min_samples) * scaled

    return {cat: int(round(n)) for cat, n in zip(category_scores.keys(), samples)}


def analyze_stability(scores: Dict[str, float], n_categories: int = 15) -> Tuple[List[str], List[str]]:
    """Return most and least stable categories"""
    sorted_cats = sorted(scores.items(), key=lambda x: x[1])
    return (
        [cat for cat, _ in sorted_cats[:n_categories]],
        [cat for cat, _ in sorted_cats[-n_categories:]]
    )


def save_results(results: dict, output_dir: str, prefix: str = ""):
    """Save evaluation results to JSON file with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_eval_results_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    # Convert any numpy values to native Python types
    clean_results = {
        'category_scores': {k: float(v) for k, v in results['category_scores'].items()},
        'new_samples': results['new_samples'],
        'category_sample_scores': {k: [float(s) for s in v] for k, v in results['category_sample_scores'].items()},
        'stable_categories': results['stable_categories'],
        'unstable_categories': results['unstable_categories']
    }

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(clean_results, f, indent=4)

    print(f"Results saved to: {output_path}")
    return output_path


def main(input_dir: str, model_path: str, img_size: int = 1024, min_samples: int = 20, max_samples: int = 100, max_val_samples: int = 10, output_dir: str = "results"):
    predictor = SODPredictor(model_path, img_size, device='cuda')
    metric_counter = EvaluationMetrics(device='cuda', sm_only=True)

    # Check if data_splits.json exists, otherwise use all images
    splits_file = os.path.join(input_dir, 'data_splits.json')
    if os.path.exists(splits_file):
        with open(splits_file, 'r') as f:
            splits = json.load(f)
        image_files = splits['val']
    else:
        # Use all images from images folder
        images_dir = os.path.join(input_dir, 'images')
        image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]

    # Group by category
    categories = defaultdict(list)
    for image_file in image_files:
        category = image_file.rsplit('_', 1)[0]
        categories[category].append(os.path.join(input_dir, 'images', image_file))

    # Evaluate categories
    category_scores = {}
    category_sample_scores = {}
    for category, images in categories.items():
        # Limit number of images per category if specified
        eval_images = images[:max_val_samples] if max_val_samples else images
        mean_score, sample_scores = eval_category(predictor, eval_images, metric_counter)
        category_scores[category] = mean_score
        category_sample_scores[category] = sample_scores

    # Calculate new samples needed
    new_samples = calculate_new_samples(category_scores, min_samples, max_samples)

    # Analyze stability
    unstable_cats, stable_cats = analyze_stability(category_scores)

    print("\nResults:")
    print("\nMost Stable Categories:")
    for cat in stable_cats:
        print(f"{cat}: score={category_scores[cat]:.4f}, new_samples={new_samples[cat]}")

    print("\nLeast Stable Categories:")
    for cat in unstable_cats:
        print(f"{cat}: score={category_scores[cat]:.4f}, new_samples={new_samples[cat]}")

    results = {
        'category_scores': category_scores,
        'new_samples': new_samples,
        'category_sample_scores': category_sample_scores,
        'stable_categories': stable_cats,
        'unstable_categories': unstable_cats
    }
    save_results(results, output_dir)


if __name__ == "__main__":
    Fire(main)
