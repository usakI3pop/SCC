"""Evaluation script for Zero-Shot Clean Accuracy on CIFAR-10.

Paper Benchmark (Table 1):
  CLIP ViT-B/32 Clean Accuracy on CIFAR-10 = 85.12%
Acceptance Target for Phase 1: ~85.1% (within +/-1.5%)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import CLIPWrapper, set_seed


# Standard CIFAR-10 class taxonomy
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_dataloader(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_samples: int | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, int]:
    """Prepares CIFAR-10 test DataLoader with CLIP spatial pre-processing in [0, 1]."""
    # Raw pixel preprocessing: bicubic resize to 224, center crop, to [0, 1] tensor
    # Mean/std normalization is done INSIDE CLIPWrapper
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Output is float tensor in [0.0, 1.0]
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    total_samples = len(dataset)
    if num_samples is not None and num_samples < total_samples:
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
        total_samples = num_samples

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, total_samples


def evaluate_clean_accuracy(
    model: CLIPWrapper,
    dataloader: DataLoader,
    class_names: list[str] = CIFAR10_CLASSES,
    device: torch.device | None = None,
) -> float:
    """Evaluates zero-shot clean top-1 classification accuracy."""
    if device is None:
        device = model.device

    # Precompute normalized text features once and cache
    text_features = model.encode_text_classes(class_names, template="a photo of a {class}")

    correct = 0
    total = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating CIFAR-10 Clean", unit="batch")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Extract normalized visual features from raw [0, 1] images
            image_features = model.encode_image(images, normalize=True)

            # Compute logits using model's own logit_scale
            logits = model.compute_logits(image_features, text_features)

            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            current_acc = 100.0 * correct / total
            pbar.set_postfix({"Acc": f"{current_acc:.2f}%", "N": total})

    elapsed = time.perf_counter() - start_time
    final_acc = 100.0 * correct / total

    print("\n" + "=" * 60)
    print("PHASE 1 ACCEPTANCE RESULTS: CIFAR-10 CLEAN ZERO-SHOT")
    print("=" * 60)
    print(f"Total Evaluated Samples : {total}")
    print(f"Correct Predictions     : {correct}")
    print(f"Clean Top-1 Accuracy    : {final_acc:.2f}%")
    print(f"Paper Reference Target  : 85.12%")
    print(f"Delta to Paper          : {final_acc - 85.12:+.2f}%")
    print(f"Elapsed Time            : {elapsed:.2f} s ({total / elapsed:.1f} img/s)")
    print("=" * 60)

    return final_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate Clean Zero-Shot CLIP ViT-B/32 on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 dataset directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of test samples to evaluate (default: all 10,000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Initializing frozen open_clip ViT-B/32 (OpenAI weights)...")
    model = CLIPWrapper(model_name="ViT-B-32", pretrained="openai")
    print(f"Device: {model.device} | Logit scale: {model.logit_scale.item():.2f}")

    print(f"Loading CIFAR-10 test set from '{args.data_dir}'...")
    dataloader, count = get_cifar10_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )
    print(f"Loaded {count} test images.")

    acc = evaluate_clean_accuracy(model, dataloader, CIFAR10_CLASSES)

    # Acceptance criterion: ~85.1%
    if abs(acc - 85.12) <= 1.5:
        print(">>> Phase 1 Acceptance Criterion: PASSED! <<<")
    else:
        print(f">>> Phase 1 Acceptance Criterion: Check delta ({acc:.2f}% vs 85.12%) <<<")


if __name__ == "__main__":
    main()
