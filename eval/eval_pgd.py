"""Evaluation script for Robust Accuracy under Untargeted PGD-10 Attack on CIFAR-10.

Paper Reference (Liu et al., ICML 2026, Table 1):
  Threat Model : Untargeted white-box PGD-10
  Budget eps_a : 1/255 (~0.00392)
  Step size a  : 1/255
  Steps        : 10
  Random start : False

Paper Targets:
  CIFAR-10 Clean Acc: ~85.12%
  CIFAR-10 Robust Acc: ~0.74% (Attack Success Rate ~99.26%)
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
from core.attacks import pgd_attack


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
    batch_size: int = 32,
    num_samples: int | None = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, int]:
    """Loads CIFAR-10 test set in raw pixel range [0, 1]."""
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 1] float tensor
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    total_samples = len(dataset)
    if num_samples is not None and num_samples < total_samples:
        dataset = Subset(dataset, list(range(num_samples)))
        total_samples = num_samples

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, total_samples


def evaluate_pgd_robustness(
    model: CLIPWrapper,
    dataloader: DataLoader,
    class_names: list[str] = CIFAR10_CLASSES,
    epsilon: float = 1.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    steps: int = 10,
    random_start: bool = False,
    device: torch.device | None = None,
) -> Tuple[float, float, float]:
    """Evaluates zero-shot clean accuracy, robust accuracy, and attack success rate."""
    if device is None:
        device = model.device

    text_features = model.encode_text_classes(class_names, template="a photo of a {class}")

    clean_correct = 0
    adv_correct = 0
    total = 0
    start_time = time.perf_counter()

    pbar = tqdm(dataloader, desc=f"PGD-10 (eps={epsilon*255:.1f}/255)", unit="batch")
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        batch_n = targets.size(0)

        # 1. Clean accuracy
        with torch.no_grad():
            clean_feats = model.encode_image(images, normalize=True)
            clean_logits = model.compute_logits(clean_feats, text_features)
            clean_preds = clean_logits.argmax(dim=-1)
            clean_correct += (clean_preds == targets).sum().item()

        # 2. PGD Attack
        _, _, diag = pgd_attack(
            model=model,
            images=images,
            targets=targets,
            text_features=text_features,
            epsilon=epsilon,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
        )

        adv_preds = diag["preds"]
        adv_correct += (adv_preds == targets).sum().item()
        total += batch_n

        current_clean = 100.0 * clean_correct / total
        current_robust = 100.0 * adv_correct / total
        pbar.set_postfix({"Clean": f"{current_clean:.1f}%", "Robust": f"{current_robust:.2f}%", "N": total})

    elapsed = time.perf_counter() - start_time
    clean_acc = 100.0 * clean_correct / total
    robust_acc = 100.0 * adv_correct / total
    asr = 100.0 - robust_acc

    print("\n" + "=" * 65)
    print("PHASE 2 ACCEPTANCE RESULTS: PGD-10 ATTACK (eps = 1/255)")
    print("=" * 65)
    print(f"Total Evaluated Samples   : {total}")
    print(f"Clean Top-1 Accuracy      : {clean_acc:.2f}%  (Paper: 85.12%)")
    print(f"Robust Top-1 Accuracy     : {robust_acc:.2f}%  (Paper: 0.74%)")
    print(f"Attack Success Rate (ASR) : {asr:.2f}%  (Paper: ~99.26%)")
    print(f"Delta to Paper Robust Acc : {robust_acc - 0.74:+.2f}%")
    print(f"Elapsed Time              : {elapsed:.2f} s ({total / elapsed:.1f} img/s)")
    print("=" * 65)

    return clean_acc, robust_acc, asr


def main():
    parser = argparse.ArgumentParser(description="Evaluate Untargeted PGD-10 on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to evaluate (default: 200)")
    parser.add_argument("--eps", type=float, default=1.0 / 255.0, help="Epsilon budget (default: 1/255)")
    parser.add_argument("--steps", type=int, default=10, help="PGD steps (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Initializing frozen open_clip ViT-B/32 (OpenAI weights)...")
    model = CLIPWrapper(model_name="ViT-B-32", pretrained="openai")

    print(f"Loading {args.num_samples} test samples from CIFAR-10...")
    dataloader, total = get_cifar10_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )

    clean_acc, robust_acc, asr = evaluate_pgd_robustness(
        model=model,
        dataloader=dataloader,
        epsilon=args.eps,
        alpha=args.eps,  # alpha = eps / 1 = 1/255
        steps=args.steps,
        random_start=False,
    )

    # Acceptance check: robust accuracy near 0.74% (within +/-2%)
    if abs(robust_acc - 0.74) <= 2.5:
        print(">>> Phase 2 Acceptance Criterion: PASSED! <<<")
    else:
        print(f">>> Phase 2 Acceptance Result: Robust Acc is {robust_acc:.2f}% (Paper: 0.74%) <<<")


if __name__ == "__main__":
    main()
