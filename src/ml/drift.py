import numpy as np
from typing import List, Union

def calculate_psi(expected: np.ndarray, actual: Union[np.ndarray, List], buckets: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.
    
    PSI = sum((Actual % - Expected %) * ln(Actual % / Expected %))
    
    Args:
        expected: Reference dataset (e.g., training data).
        actual: Current dataset (e.g., production data).
        buckets: Number of bins for discretization.
        
    Returns:
        float: The PSI score.
    """
    expected = np.array(expected)
    actual = np.array(actual)

    def scale_range(input_data, min_val, max_val):
        """Discretize the input data into buckets."""
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
        # Handle values exactly at max_val by putting them in the last bucket
        counts, _ = np.histogram(input_data, bins=breakpoints)
        return counts

    # Define range based on the union of both datasets
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    expected_counts = scale_range(expected, min_val, max_val)
    actual_counts = scale_range(actual, min_val, max_val)

    # Convert to percentages
    expected_percents = expected_counts / len(expected)
    actual_percents = actual_counts / len(actual)

    # Handle zero counts by adding a small epsilon to avoid division by zero and log of zero
    epsilon = 1e-6
    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    # Calculate PSI
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    psi_score = np.sum(psi_values)

    return float(psi_score)
