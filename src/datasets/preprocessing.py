from torchvision import transforms


# Standard image size used by the baseline pipeline.
IMAGE_SIZE = (32, 32)


def get_train_transform():
    """
    Return the preprocessing pipeline used for training.

    Training transformations should be applied only to the
    training dataset.
    """

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def get_eval_transform():
    """
    Return the preprocessing pipeline used for validation and testing.

    Validation and test transformations are deterministic.
    """

    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])