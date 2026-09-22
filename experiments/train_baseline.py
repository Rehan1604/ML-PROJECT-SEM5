import argparse
import os

import torch

from src.datasets.gtsrb import create_gtsrb_dataloaders
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

CHECKPOINT_PATH = "experiments/checkpoints/gtsrb_baseline.pth"
HISTORY_PATH = "experiments/results/gtsrb_baseline_history.json"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN on GTSRB."
    )

    parser.add_argument(
        "--gtsrb-root",
        required=True,
        help="Path to the GTSRB dataset directory.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    gtsrb_root = args.gtsrb_root

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nGTSRB root:", gtsrb_root)

    if not os.path.isdir(gtsrb_root):
        raise FileNotFoundError(
            f"GTSRB dataset directory not found: {gtsrb_root}"
        )

    print("\nLoading GTSRB dataset...")

    train_loader, validation_loader, test_loader = (
        create_gtsrb_dataloaders(
            root_dir=gtsrb_root,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_transform=get_train_transform(),
            eval_transform=get_eval_transform(),
        )
    )

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Test batches:", len(test_loader))

    model = TrafficSignCNN(
        num_classes=43
    )

    print("\nModel created.")

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
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
    
    os.makedirs(
        os.path.dirname(HISTORY_PATH),
        exist_ok=True,
    )

    import json

    with open(HISTORY_PATH, "w") as file:
        json.dump(history, file, indent=4)

    print("Training history saved to:", HISTORY_PATH)

    print("\nTraining completed.")

    print(
        "Best validation accuracy:",
        max(history["validation_accuracy"]),
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
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