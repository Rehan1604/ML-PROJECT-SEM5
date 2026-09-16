# src/augmentations/severity.py

"""
Severity configuration for traffic-sign image corruptions.

Severity levels:
    1 = least severe
    5 = most severe
"""

SEVERITY_LEVELS = (1, 2, 3, 4, 5)

CORRUPTIONS = (
    "gaussian",
    "impulse",
    "shot",
    "snow",
    "fog",
    "frost",
    "motion_blur",
    "zoom_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "pixelate",
    "elastic_transform",
    "jpeg_compression",
)


# Parameters are deliberately kept in one place so that
# the corruption implementation does not need to be changed
# when severity values are adjusted.

SEVERITY_PARAMS = {

    "gaussian": {
        1: {"sigma": 0.05},
        2: {"sigma": 0.10},
        3: {"sigma": 0.15},
        4: {"sigma": 0.20},
        5: {"sigma": 0.25},
    },

    "impulse": {
        1: {"amount": 0.01},
        2: {"amount": 0.03},
        3: {"amount": 0.05},
        4: {"amount": 0.08},
        5: {"amount": 0.12},
    },

    "shot": {
        1: {"scale": 0.05},
        2: {"scale": 0.10},
        3: {"scale": 0.15},
        4: {"scale": 0.20},
        5: {"scale": 0.25},
    },

    "snow": {
        1: {"intensity": 0.05},
        2: {"intensity": 0.10},
        3: {"intensity": 0.15},
        4: {"intensity": 0.20},
        5: {"intensity": 0.30},
    },

    "fog": {
        1: {"strength": 0.10},
        2: {"strength": 0.20},
        3: {"strength": 0.30},
        4: {"strength": 0.40},
        5: {"strength": 0.50},
    },

    "frost": {
        1: {"strength": 0.10},
        2: {"strength": 0.20},
        3: {"strength": 0.30},
        4: {"strength": 0.40},
        5: {"strength": 0.50},
    },

    "motion_blur": {
        1: {"kernel_size": 3},
        2: {"kernel_size": 5},
        3: {"kernel_size": 7},
        4: {"kernel_size": 9},
        5: {"kernel_size": 13},
    },

    "zoom_blur": {
        1: {"strength": 0.05},
        2: {"strength": 0.10},
        3: {"strength": 0.15},
        4: {"strength": 0.20},
        5: {"strength": 0.25},
    },

    "defocus_blur": {
        1: {"kernel_size": 3},
        2: {"kernel_size": 5},
        3: {"kernel_size": 7},
        4: {"kernel_size": 9},
        5: {"kernel_size": 11},
    },

    "brightness": {
        1: {"factor": 0.20},
        2: {"factor": 0.40},
        3: {"factor": 0.60},
        4: {"factor": 0.80},
        5: {"factor": 1.00},
    },

    "contrast": {
        1: {"factor": 0.20},
        2: {"factor": 0.40},
        3: {"factor": 0.60},
        4: {"factor": 0.80},
        5: {"factor": 1.00},
    },

    "pixelate": {
        1: {"scale": 0.80},
        2: {"scale": 0.60},
        3: {"scale": 0.40},
        4: {"scale": 0.25},
        5: {"scale": 0.15},
    },

    "elastic_transform": {
        1: {"alpha": 5.0, "sigma": 4.0},
        2: {"alpha": 10.0, "sigma": 5.0},
        3: {"alpha": 15.0, "sigma": 6.0},
        4: {"alpha": 20.0, "sigma": 7.0},
        5: {"alpha": 25.0, "sigma": 8.0},
    },

    "jpeg_compression": {
        1: {"quality": 80},
        2: {"quality": 60},
        3: {"quality": 40},
        4: {"quality": 20},
        5: {"quality": 10},
    },
}


def validate_corruption(corruption: str, severity: int) -> None:
    """Validate corruption name and severity level."""

    if corruption not in CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{corruption}'. "
            f"Available: {CORRUPTIONS}"
        )

    if severity not in SEVERITY_LEVELS:
        raise ValueError(
            f"Severity must be one of {SEVERITY_LEVELS}, "
            f"got {severity}"
        )


def get_parameters(corruption: str, severity: int) -> dict:
    """Return parameters for a corruption/severity combination."""

    validate_corruption(corruption, severity)

    return SEVERITY_PARAMS[corruption][severity].copy()