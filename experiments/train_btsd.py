import argparse
import json
import os

import torch

from src.datasets.btsd import create_btsd_dataloaders
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

CHECKPOINT_PATH = "experiments/checkpoints/btsd_baseline.pth"
HISTORY_PATH = "experiments/results/btsd_baseline_history.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on BTSD."
    )

    parser.add_argument(
        "--btsd-root",
        required=True,
        help="Path to the BTSD dataset directory.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    btsd_root = args.btsd_root

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nBTSD root:", btsd_root)

    if not os.path.isdir(btsd_root):
        raise FileNotFoundError(
            f"BTSD dataset directory not found: {btsd_root}"
        )

    print("\nLoading BTSD dataset...")

    train_loader, validation_loader, test_loader = (
        create_btsd_dataloaders(
            root_dir=btsd_root,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transform=get_train_transform(),
            eval_transform=get_eval_transform(),
        )
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))

    model = TrafficSignCNN(num_classes=62)

    print("\nModel created.")
    print("Number of classes:", 62)

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True,
    )

    os.makedirs(
        os.path.dirname(HISTORY_PATH),
        exist_ok=True,
    )

    history = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        checkpoint_path=CHECKPOINT_PATH,
    )

    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=4)


    print("\nTraining history saved to:", HISTORY_PATH)

    print("Training completed.")

    print(
        "Best validation accuracy:",
        max(history["validation_accuracy"]),
    )

    print("Checkpoint:", CHECKPOINT_PATH)


if __name__ == "__main__":
    main()