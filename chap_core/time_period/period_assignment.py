import numpy as np

from chap_core.time_period import PeriodRange
from chap_core.time_period.date_util_wrapper import delta_day


class PeriodAssignment:
    """
    Matches two period ranges (with possibly different time deltas) and computes overlap-based assignments.

    This is useful when aggregating or downsampling time series:
    e.g. mapping weekly values into monthly periods with appropriate weighting.

    Attributes
    ----------
    indices : np.ndarray
        Indices of the from_range periods contributing to each to_range period.
    weights : np.ndarray
        Relative weight of each from_range index in the corresponding to_range period.
    """

    def __init__(self, to_range: PeriodRange, from_range: PeriodRange):
        self.to_range = to_range
        self.from_range = from_range

        # Convert the from_range delta to number of days (e.g. 7 for weekly)
        self._from_range_days = from_range.delta // delta_day

        # Compute overlap assignments
        self.indices, self.weights = self._calculate_assignments()

    def _calculate_assignments(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the overlap assignments from `from_range` to `to_range`.

        Returns
        -------
        tuple
            - indices: shape (len(to_range), max_matches), int indices into from_range
            - weights: same shape, with float overlap weights (between 0 and 1)
        """
        assignments = []

        # For each period in the to_range, find contributing periods in from_range
        for period in self.to_range:
            matches = self._match_period(period)
            assignments.append(matches)

        # Determine the max number of overlaps any to_period has
        max_len = max(len(matches) for matches in assignments)

        # Initialize index and weight arrays
        indices = np.zeros((len(assignments), max_len), dtype=int)
        weights = np.zeros((len(assignments), max_len), dtype=float)

        # Fill in the values for each to_range period
        for i, matches in enumerate(assignments):
            for j, (index, weight) in enumerate(matches):
                indices[i, j] = index
                weights[i, j] = weight

        return indices, weights

    def _match_period(self, to_period) -> list[tuple[int, float]]:
        """
        Find overlapping from_range periods for a given to_range period.

        Parameters
        ----------
        to_period : TimePeriod
            Target time period to match.

        Returns
        -------
        List[Tuple[int, float]]
            List of (index, overlap_fraction) where overlap_fraction is relative to the from_range period.
        """
        matches = []

        for i, from_period in enumerate(self.from_range):
            # Find intersection range
            max_start = max(to_period.start_timestamp, from_period.start_timestamp)
            min_end = min(to_period.end_timestamp, from_period.end_timestamp)

            # Compute overlap in days
            overlap = max((min_end - max_start) // delta_day, 0)

            if overlap > 0:
                # Normalize overlap by the full duration of the from_period
                weight = overlap / self._from_range_days
                matches.append((i, weight))

        return matches
