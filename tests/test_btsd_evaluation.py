import csv
import json
import os

import torch

from src.datasets.btsd import create_btsd_dataloaders
from src.datasets.preprocessing import (
    get_train_transform,
    get_eval_transform,
)
from src.models.baseline import TrafficSignCNN
from src.training.evaluate import evaluate_model


BTSD_ROOT = r"C:\Users\Meet Rathod\Desktop\ML Lab Project datasets\BTSD"
CHECKPOINT_PATH = "experiments/checkpoints/btsd_baseline.pth"

RESULTS_DIR = "experiments/results"
RESULTS_JSON = os.path.join(
    RESULTS_DIR,
    "btsd_baseline.json",
)
CONFUSION_CSV = os.path.join(
    RESULTS_DIR,
    "btsd_baseline_confusion_matrix.csv",
)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nLoading BTSD test set...")

    _, _, test_loader = create_btsd_dataloaders(
        root_dir=BTSD_ROOT,
        batch_size=64,
        num_workers=0,
        train_transform=get_train_transform(),
        eval_transform=get_eval_transform(),
    )

    print("Test batches:", len(test_loader))

    model = TrafficSignCNN(num_classes=62)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("\nCheckpoint loaded.")
    print("Training epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy:",
        checkpoint["validation_accuracy"],
    )

    print("\nEvaluating on BTSD test set...")

    accuracy, f1, confusion, labels, predictions = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    results = {
        "dataset": "BTSD",
        "model": "TrafficSignCNN",
        "experiment": "clean_baseline",
        "test_samples": len(labels),
        "num_classes": 62,
        "training_epochs": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint[
            "validation_accuracy"
        ],
        "test_accuracy": accuracy,
        "macro_f1": f1,
    }

    with open(RESULTS_JSON, "w") as file:
        json.dump(results, file, indent=4)


    with open(CONFUSION_CSV, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow(
            ["true_class"] + list(range(62))
        )

        for class_index, row in enumerate(confusion):
            writer.writerow(
                [class_index] + row.tolist()
            )


    print("\n===== BTSD BASELINE TEST RESULTS =====")
    print(
        f"Test Accuracy: {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )
    print(f"Macro F1:      {f1:.4f}")
    print(f"Confusion Matrix Shape: {confusion.shape}")
    print(f"Number of Test Samples: {len(labels)}")

    print("\nResults saved:")
    print("Metrics:", RESULTS_JSON)
    print("Confusion Matrix:", CONFUSION_CSV)

    print("======================================")


if __name__ == "__main__":
    main()