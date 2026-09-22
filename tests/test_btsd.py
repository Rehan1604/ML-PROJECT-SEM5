from pathlib import Path

from src.datasets.btsd import (
    NUM_CLASSES,
    create_btsd_dataloaders,
    load_btsd_samples,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive (8)"


def test_btsd_structure():
    train_samples, test_samples = load_btsd_samples(DATA_DIR)

    assert len(train_samples) > 0
    assert len(test_samples) > 0

    train_classes = {label for _, label in train_samples}
    test_classes = {label for _, label in test_samples}

    assert train_classes == set(range(NUM_CLASSES))

    assert test_classes.issubset(set(range(NUM_CLASSES)))


def test_btsd_dataloader():
    train_loader, validation_loader, test_loader = create_btsd_dataloaders(
        root_dir=DATA_DIR,
        batch_size=64,
        num_workers=0,
        validation_size=0.2,
        random_state=42,
    )

    assert len(train_loader.dataset) > 0
    assert len(validation_loader.dataset) > 0
    assert len(test_loader.dataset) > 0

    images, labels = next(iter(train_loader))

    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)

    assert images.min().item() >= -1.0
    assert images.max().item() <= 1.0

    assert labels.min().item() >= 0
    assert labels.max().item() < NUM_CLASSES


if __name__ == "__main__":
    test_btsd_structure()
    test_btsd_dataloader()
    print("All BTSD tests passed.")
