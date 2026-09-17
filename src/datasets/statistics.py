from pathlib import Path

import pandas as pd


def get_class_distribution(csv_file):
    """
    Calculate the number of samples in each traffic-sign class.

    Parameters
    ----------
    csv_file : str or Path
        Path to a CSV file containing a 'ClassId' column.

    Returns
    -------
    pandas.Series
        Sample count for every class.
    """

    csv_file = Path(csv_file)

    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}"
        )

    data = pd.read_csv(csv_file)

    if "ClassId" not in data.columns:
        raise ValueError(
            "CSV file must contain a 'ClassId' column."
        )

    return data["ClassId"].value_counts().sort_index()


def get_dataset_statistics(csv_file):
    """
    Calculate basic statistics for a traffic-sign dataset.

    Returns
    -------
    dict
        Dataset statistics.
    """

    distribution = get_class_distribution(csv_file)

    return {
        "number_of_classes": int(len(distribution)),
        "total_samples": int(distribution.sum()),
        "minimum_class_samples": int(distribution.min()),
        "maximum_class_samples": int(distribution.max()),
        "mean_samples_per_class": float(distribution.mean()),
        "class_distribution": distribution.to_dict(),
    }


def print_dataset_statistics(csv_file):
    """
    Print dataset statistics in a readable format.
    """

    stats = get_dataset_statistics(csv_file)

    print("\nDataset Statistics")
    print("------------------")

    print(f"Number of classes:       {stats['number_of_classes']}")
    print(f"Total samples:           {stats['total_samples']}")
    print(f"Minimum class samples:   {stats['minimum_class_samples']}")
    print(f"Maximum class samples:   {stats['maximum_class_samples']}")
    print(f"Mean samples per class:  {stats['mean_samples_per_class']:.2f}")

    print("\nClass Distribution")
    print("------------------")

    for class_id, count in stats["class_distribution"].items():
        print(f"Class {class_id:2d}: {count:5d}")