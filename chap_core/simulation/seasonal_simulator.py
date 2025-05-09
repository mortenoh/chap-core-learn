import numpy as np


class SeasonalSingleVariableSimulator:
    """
    Simulates a 1D time series with seasonal peaks and valleys.
    Useful for modeling variables like disease incidence, temperature, or demand
    that exhibit seasonal behavior.
    """

    def __init__(
        self,
        n_seasons: int,
        n_data_points_per_season: int,
        mean_peak_height: float,
        peak_height_sd: float,
    ):
        """
        Initialize the seasonal simulator.

        Parameters:
            n_seasons (int): Number of distinct seasons (e.g., years).
            n_data_points_per_season (int): Number of time steps per season (e.g., months).
            mean_peak_height (float): Average height of seasonal peaks.
            peak_height_sd (float): Standard deviation of peak heights.
        """
        self.n_seasons = n_seasons
        self.n_data_points_per_season = n_data_points_per_season
        self.mean_peak_height = mean_peak_height
        self.peak_height_sd = peak_height_sd

        # Total number of data points = seasons × points per season
        self.data_size = self.n_seasons * self.n_data_points_per_season

    def simulate_peak_positions(self) -> np.ndarray:
        """
        Randomly assigns one peak position per season.

        Returns:
            np.ndarray: Array of peak indices.
        """
        bin_size = self.data_size // self.n_seasons
        peak_positions = np.zeros(self.n_seasons, dtype=int)

        for i in range(self.n_seasons):
            # Ensure each peak falls somewhere in its season's bin
            peak_positions[i] = np.random.randint(low=(i * bin_size) + 1, high=((i + 1) * bin_size) - 1)
        return peak_positions

    def simulate_peak_heights(self) -> np.ndarray:
        """
        Simulate peak heights using a normal distribution.

        Returns:
            np.ndarray: Array of peak height values.
        """
        return np.random.normal(loc=self.mean_peak_height, scale=self.peak_height_sd, size=self.n_seasons)

    def simulate_valley_positions(self, peak_positions: np.ndarray) -> np.ndarray:
        """
        Generate valley positions between and around peaks.

        Parameters:
            peak_positions (np.ndarray): Array of peak indices.

        Returns:
            np.ndarray: Array of valley indices (length = n_seasons + 1).
        """
        valley_positions = np.zeros(self.n_seasons + 1, dtype=int)

        # Valley before the first peak
        valley_positions[0] = np.random.randint(0, peak_positions[0])

        # Valleys between peaks
        for i in range(1, self.n_seasons):
            valley_positions[i] = np.random.randint(low=peak_positions[i - 1] + 1, high=peak_positions[i])

        # Valley after the last peak
        valley_positions[-1] = np.random.randint(low=peak_positions[-1] + 1, high=self.data_size)

        return valley_positions

    def simulate_valley_heights(self, peak_heights: np.ndarray) -> np.ndarray:
        """
        Generate valley heights as a fraction of peak heights with noise.

        Parameters:
            peak_heights (np.ndarray): Array of peak height values.

        Returns:
            np.ndarray: Array of valley height values (length = n_seasons + 1).
        """
        # Expected valley means are ~30% of the peaks
        valley_means = peak_heights * 0.3
        valley_means = np.insert(valley_means, 0, peak_heights[0] * 0.3)  # Add first valley

        # Add small random noise to valleys
        valley_heights = np.random.normal(loc=valley_means, scale=valley_means * 0.1)
        return valley_heights

    def simulate(self) -> np.ndarray:
        """
        Runs the full seasonal simulation and returns a smooth signal.

        Returns:
            np.ndarray: Simulated 1D array of data points with seasonal structure.
        """
        # Pre-allocate array with zeros
        data = np.zeros(self.data_size, dtype=int)

        # Generate all structural elements
        peak_positions = self.simulate_peak_positions()
        peak_heights = self.simulate_peak_heights()
        valley_positions = self.simulate_valley_positions(peak_positions)
        valley_heights = self.simulate_valley_heights(peak_heights)

        # Place peak and valley values in the data
        data[peak_positions] = peak_heights.astype(int)
        data[valley_positions] = valley_heights.astype(int)

        # Linearly interpolate between peaks and valleys
        nonzero_indices = np.nonzero(data)[0]
        for i in range(len(nonzero_indices) - 1):
            start = nonzero_indices[i]
            end = nonzero_indices[i + 1]
            # Linear interpolation between two known points
            data[start:end] = np.linspace(start=data[start], stop=data[end], num=end - start, dtype=int)

        return data
