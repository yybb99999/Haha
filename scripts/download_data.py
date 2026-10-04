"""Download official torchvision datasets into each experiment's data root."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["utility", "attack"], required=True)
    args = parser.parse_args()
    from torchvision.datasets import CIFAR10, FashionMNIST
    data_root = ROOT / args.part / "data"
    datasets = [CIFAR10, FashionMNIST] if args.part == "utility" else [CIFAR10]
    for dataset in datasets:
        for train in (True, False):
            data = dataset(root=str(data_root), train=train, download=True)
            print(dataset.__name__, "train" if train else "test", len(data), data_root)


if __name__ == "__main__":
    main()
