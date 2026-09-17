from pathlib import Path

from src.datasets.btsd import create_btsd_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive (8)"


def test_btsd_split():
    train_dataset, validation_dataset, test_dataset = create_btsd_datasets(
        root_dir=DATA_DIR,
        validation_size=0.2,
        random_state=42,
    )

    train_paths = {str(path) for path, _ in train_dataset.samples}
    validation_paths = {str(path) for path, _ in validation_dataset.samples}
    test_paths = {str(path) for path, _ in test_dataset.samples}

    train_classes = {label for _, label in train_dataset.samples}
    validation_classes = {label for _, label in validation_dataset.samples}
    test_classes = {label for _, label in test_dataset.samples}

    expected_classes = set(range(62))

    assert train_classes == expected_classes
    assert validation_classes == expected_classes
    assert test_classes.issubset(expected_classes)

    assert train_paths.isdisjoint(validation_paths)
    assert train_paths.isdisjoint(test_paths)
    assert validation_paths.isdisjoint(test_paths)

    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Test samples:", len(test_dataset))
    print("Train classes:", len(train_classes))
    print("Validation classes:", len(validation_classes))
    print("Test classes:", len(test_classes))
    print("No path overlap between splits.")
    print("BTSD split validation passed.")


if __name__ == "__main__":
    test_btsd_split()