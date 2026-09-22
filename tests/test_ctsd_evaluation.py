import argparse
import csv
import json
import os

import torch
from torch.utils.data import DataLoader

from src.datasets.ctsd import CTSDSampleDataset, load_ctsd_data
from src.datasets.preprocessing import get_eval_transform
from src.models.baseline import TrafficSignCNN
from src.training.evaluate import evaluate_model
from src.training.checkpoint import load_checkpoint
from src.utils.seed import set_seed

import pandas as pd




BATCH_SIZE = 64
NUM_WORKERS = 0
SEED = 42
NUM_CLASSES = 58

CHECKPOINT_PATH = (
    "experiments/checkpoints/ctsd_baseline.pth"
)

METRICS_PATH = (
    "experiments/results/ctsd_baseline.json"
)

CONFUSION_MATRIX_PATH = (
    "experiments/results/"
    "ctsd_baseline_confusion_matrix.csv"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate the baseline CNN on CTSD test set."
    )

    parser.add_argument(
        "--ctsd-root",
        required=True,
        help="Path to the CTSD dataset directory.",
    )

    return parser.parse_args()


def create_ctsd_test_split(data, seed=42):

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
            n_test = max(
                1,
                round(0.20 * n),
            )

            n_validation = max(
                1,
                round(0.20 * n),
            )

            n_train = (
                n
                - n_validation
                - n_test
            )

        test_parts.append(
            group.iloc[
                n_train + n_validation:
            ]
        )

    test_data = (
        pd.concat(test_parts)
        .sample(
            frac=1,
            random_state=seed,
        )
        .reset_index(drop=True)
    )

    return test_data


def main():

    args = parse_arguments()

    ctsd_root = args.ctsd_root


    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print("\nLoading CTSD dataset...")

    data, image_dir = load_ctsd_data(
        ctsd_root
    )

    print(
        "Unique images:",
        len(data),
    )

    print(
        "Number of classes:",
        data["category"].nunique(),
    )

    test_data = create_ctsd_test_split(
        data=data,
        seed=SEED,
    )

    print(
        "\nCTSD test samples:",
        len(test_data),
    )

    print(
        "Test classes:",
        test_data["category"].nunique(),
    )

    test_dataset = CTSDSampleDataset(
        data=test_data,
        image_dir=image_dir,
        transform=get_eval_transform(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(
        "Test batches:",
        len(test_loader),
    )

    model = TrafficSignCNN(
        num_classes=NUM_CLASSES
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )


    print("\nLoading checkpoint...")

    training_epoch, validation_accuracy = (
        load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=CHECKPOINT_PATH,
            device=device,
        )
    )

    model.to(device)

    print("Checkpoint loaded.")

    print(
        "Training epoch:",
        training_epoch,
    )

    print(
        "Best validation accuracy:",
        validation_accuracy,
    )

    print(
        "\nEvaluating on CTSD test set..."
    )

    (
        accuracy,
        f1,
        confusion,
        all_labels,
        all_predictions,
    ) = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    print(
        "\n===== CTSD BASELINE TEST RESULTS ====="
    )

    print(
        f"Test Accuracy: "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro F1: {f1:.4f}"
    )

    print(
        "Confusion Matrix Shape:",
        confusion.shape,
    )

    print(
        "Number of Test Samples:",
        len(all_labels),
    )

    os.makedirs(
        "experiments/results",
        exist_ok=True,
    )

    metrics = {
        "dataset": "CTSD",
        "model": "TrafficSignCNN",
        "num_classes": NUM_CLASSES,
        "test_samples": len(all_labels),
        "training_epoch": training_epoch,
        "best_validation_accuracy": validation_accuracy,
        "test_accuracy": accuracy,
        "macro_f1": f1,
    }

    with open(
        METRICS_PATH,
        "w",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    with open(
        CONFUSION_MATRIX_PATH,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "true_class"
            ]
            + list(
                range(NUM_CLASSES)
            )
        )

        for class_index, row in enumerate(
            confusion
        ):

            writer.writerow(
                [class_index]
                + row.tolist()
            )

    print("\nResults saved:")

    print(
        "Metrics:",
        METRICS_PATH,
    )

    print(
        "Confusion Matrix:",
        CONFUSION_MATRIX_PATH,
    )


if __name__ == "__main__":
    main()