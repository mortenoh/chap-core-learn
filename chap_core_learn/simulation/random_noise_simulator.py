import numpy as np

from chap_core.datatypes import ClimateHealthTimeSeries
from chap_core.simulation.simulator import Simulator
from chap_core.time_period import PeriodRange, Year


class RandomNoiseSimulator(Simulator):
    """
    A simple simulator that generates synthetic climate and disease data
    using random noise for exploratory or testing purposes.
    """

    def __init__(self, n_time_points: int):
        """
        Initialize the simulator with the number of time points to simulate.

        Parameters:
            n_time_points (int): Number of sequential time periods to simulate.
        """
        super().__init__()
        self.n_time_points = n_time_points

    def simulate(self) -> ClimateHealthTimeSeries:
        """
        Run the simulation and return a ClimateHealthTimeSeries object
        populated with synthetic (random) data.

        Returns:
            ClimateHealthTimeSeries: Simulated dataset containing:
              - Gaussian rainfall values
              - Gaussian temperature values
              - Poisson-distributed disease cases
        """
        return ClimateHealthTimeSeries(
            # Create a time period range starting at Year(1)
            time_period=PeriodRange.from_start_and_n_periods(Year(1), self.n_time_points),
            # Generate rainfall and temperature using standard normal distribution
            rainfall=np.random.randn(self.n_time_points),
            mean_temperature=np.random.randn(self.n_time_points),
            # Generate disease cases from a Poisson distribution with λ=10
            disease_cases=np.random.poisson(lam=10, size=self.n_time_points),
        )
