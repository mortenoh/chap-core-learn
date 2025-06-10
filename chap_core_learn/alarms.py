# Improvement Suggestions:
# 1. Consider adding validation for `endemic_factor` and `probability_threshold` in `OutbreakParameters` (e.g., must be positive, `probability_threshold` between 0 and 1).
# 2. The `outbreak_prediction` function could benefit from more detailed error handling or logging for the `ValueError`.
# 3. Explore alternative statistical methods for outbreak detection beyond a simple threshold for more robust predictions.
# 4. Add unit tests specifically for edge cases in `outbreak_prediction` (e.g., empty `case_samples`, all samples equal, all samples below/above threshold).
# 5. Consider if `case_samples` should enforce a minimum length for statistical significance.

"""
This module defines parameters and functions for predicting disease outbreaks
based on probabilistic forecast samples.
"""

from typing import Iterable

import numpy as np
from pydantic import BaseModel


class OutbreakParameters(BaseModel):
    """
    Parameters used to determine outbreak likelihood.

    Attributes:
        endemic_factor (float): The multiplier for the endemic (expected) case rate.
        probability_threshold (float): Minimum proportion of samples that must exceed
                                       the threshold to classify as an outbreak.
    """

    endemic_factor: float
    probability_threshold: float


def outbreak_prediction(parameters: OutbreakParameters, case_samples: Iterable[float]) -> bool:
    """
    Predict whether an outbreak is occurring based on probabilistic forecast samples.

    Parameters:
        parameters (OutbreakParameters): Configurations for the outbreak rule.
        case_samples (Iterable[float]): A sample of predicted case counts.

    Returns:
        bool: True if an outbreak is predicted, False otherwise.

    Raises:
        ValueError: If `case_samples` is empty or contains non-finite values.
    """
    samples = np.array(list(case_samples))

    if len(samples) == 0 or not np.isfinite(samples).all():
        raise ValueError("Invalid case samples provided.")

    threshold = parameters.endemic_factor * np.mean(samples)
    proportion_above = np.mean(samples > threshold)

    return proportion_above > parameters.probability_threshold
