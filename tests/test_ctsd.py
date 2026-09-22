from pathlib import Path

from src.datasets.ctsd import (
    create_ctsd_dataloaders,
    create_ctsd_datasets,
    load_ctsd_annotations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive (9)"


def test_ctsd_annotations():
    data = load_ctsd_annotations(DATA_DIR / "annotations.csv")

    assert len(data) == 5998
    assert data["file_name"].is_unique

    expected_columns = {
        "file_name",
        "width",
        "height",
        "x1",
        "y1",
        "x2",
        "y2",
        "category",
    }

    assert expected_columns.issubset(data.columns)
    assert data["category"].notna().all()
    assert (data["category"] >= 0).all()

    image_dir = DATA_DIR / "images"

    for file_name in data["file_name"].head(10):
        assert (image_dir / file_name).exists()


def test_ctsd_split():
    train_dataset, validation_dataset = create_ctsd_datasets(
        root_dir=DATA_DIR,
        validation_size=0.2,
        random_state=42,
    )

    train_paths = set(train_dataset.data["file_name"])
    validation_paths = set(validation_dataset.data["file_name"])

    train_classes = set(train_dataset.data["category"])
    validation_classes = set(validation_dataset.data["category"])

    all_classes = train_classes | validation_classes

    expected_classes = set(range(58))

    
    assert all_classes == expected_classes

    
    assert train_paths.isdisjoint(validation_paths)

    
    assert train_classes.issubset(expected_classes)
    assert validation_classes.issubset(expected_classes)

    
    assert len(train_dataset) + len(validation_dataset) == 5998

    assert len(train_dataset) == 4798
    assert len(validation_dataset) == 1200


def test_ctsd_dataloader():
    train_loader, validation_loader = create_ctsd_dataloaders(
        root_dir=DATA_DIR,
        batch_size=64,
        num_workers=0,
        validation_size=0.2,
        random_state=42,
    )

    images, labels, metadata = next(iter(train_loader))

    assert images.shape == (64, 3, 32, 32)
    assert labels.shape == (64,)

    assert images.min().item() >= -1.0
    assert images.max().item() <= 1.0

    assert labels.min().item() >= 0
    assert labels.max().item() < 58

    assert "file_name" in metadata
    assert "x1" in metadata
    assert "y1" in metadata
    assert "x2" in metadata
    assert "y2" in metadata

    assert len(train_loader.dataset) == 4798
    assert len(validation_loader.dataset) == 1200


if __name__ == "__main__":
    test_ctsd_annotations()
    test_ctsd_split()
    test_ctsd_dataloader()
    print("All CTSD tests passed.")