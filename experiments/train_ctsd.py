import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.datasets.ctsd import CTSDSampleDataset, load_ctsd_data
from src.datasets.preprocessing import (
    get_train_transform,
    get_eval_transform,
)
from src.models.baseline import TrafficSignCNN
from src.training.train import train_model
from src.utils.seed import set_seed

BATCH_SIZE = 64
NUM_WORKERS = 0
EPOCHS = 10
LEARNING_RATE = 0.001
SEED = 42

NUM_CLASSES = 58

CHECKPOINT_PATH = "experiments/checkpoints/ctsd_baseline.pth"
HISTORY_PATH = "experiments/results/ctsd_baseline_history.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on CTSD."
    )

    parser.add_argument(
        "--ctsd-root",
        required=True,
        help="Path to the CTSD dataset directory.",
    )

    return parser.parse_args()


def create_ctsd_split(data, seed=42):
    """
    Create deterministic train/validation/test splits for CTSD.

    CTSD is highly imbalanced, with some classes containing
    only a few images. Therefore, a standard stratified split
    cannot be applied to all three subsets.

    Target proportions for sufficiently large classes:
        ~60% train
        ~20% validation
        ~20% test

    Very small classes are handled using integer constraints.
    """

    train_parts = []
    validation_parts = []
    test_parts = []

    for category, group in data.groupby("category"):
        group = group.sample(
            frac=1,
            random_state=seed + int(category),
        ).reset_index(drop=True)

        n = len(group)

        if n == 1:
            n_train = 1
            n_validation = 0
            n_test = 0

        elif n == 2:
            n_train = 1
            n_validation = 1
            n_test = 0

        elif n == 3:
            n_train = 1
            n_validation = 1
            n_test = 1

        elif n == 4:
            n_train = 2
            n_validation = 1
            n_test = 1

        else:
            n_test = max(1, round(0.20 * n))
            n_validation = max(1, round(0.20 * n))
            n_train = n - n_validation - n_test

        train_parts.append(
            group.iloc[:n_train]
        )

        validation_parts.append(
            group.iloc[
                n_train:n_train + n_validation
            ]
        )

        test_parts.append(
            group.iloc[
                n_train + n_validation:
            ]
        )

    train_data = (
        pd.concat(train_parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    validation_data = (
        pd.concat(validation_parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    test_data = (
        pd.concat(test_parts)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return train_data, validation_data, test_data


def main():
    args = parse_arguments()
    ctsd_root = args.ctsd_root

    # -------------------------
    # Reproducibility
    # -------------------------

    set_seed(SEED)

    # -------------------------
    # Device
    # -------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------
    # Dataset path
    # -------------------------

    print("\nCTSD root:", ctsd_root)

    if not os.path.isdir(ctsd_root):
        raise FileNotFoundError(
            f"CTSD dataset directory not found: {ctsd_root}"
        )

    # -------------------------
    # Load CTSD
    # -------------------------

    print("\nLoading CTSD dataset...")

    data, image_dir = load_ctsd_data(ctsd_root)

    print("Unique images:", len(data))
    print(
        "Number of classes:",
        data["category"].nunique(),
    )

    # -------------------------
    # Create split
    # -------------------------

    train_data, validation_data, test_data = (
        create_ctsd_split(
            data=data,
            seed=SEED,
        )
    )

    print("\nDataset split:")
    print("Train samples:", len(train_data))
    print("Validation samples:", len(validation_data))
    print("Test samples:", len(test_data))

    # -------------------------
    # Check split
    # -------------------------

    print(
        "Train classes:",
        train_data["category"].nunique(),
    )

    print(
        "Validation classes:",
        validation_data["category"].nunique(),
    )

    print(
        "Test classes:",
        test_data["category"].nunique(),
    )

    # -------------------------
    # Datasets
    # -------------------------

    train_dataset = CTSDSampleDataset(
        data=train_data,
        image_dir=image_dir,
        transform=get_train_transform(),
    )

    validation_dataset = CTSDSampleDataset(
        data=validation_data,
        image_dir=image_dir,
        transform=get_eval_transform(),
    )

    # -------------------------
    # DataLoaders
    # -------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nTrain batches:", len(train_loader))
    print(
        "Validation batches:",
        len(validation_loader),
    )

    # -------------------------
    # Model
    # -------------------------

    model = TrafficSignCNN(
        num_classes=NUM_CLASSES
    )

    print("\nModel created.")
    print("Number of classes:", NUM_CLASSES)

    # -------------------------
    # Output directories
    # -------------------------

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(HISTORY_PATH),
        exist_ok=True,
    )

    # -------------------------
    # Training
    # -------------------------

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # -------------------------
    # Save training history
    # -------------------------

    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=4)

    # -------------------------
    # Results
    # -------------------------

    print(
        "\nTraining history saved to:",
        HISTORY_PATH,
    )

    print("\nTraining completed.")

    print(
        "Best validation accuracy:",
        max(history["validation_accuracy"]),
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()