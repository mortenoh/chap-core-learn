from typing import Protocol

from chap_core.datatypes import ClimateData, ClimateHealthTimeSeries, HealthData


class Simulator:
    """
    Base simulator class for generating synthetic ClimateHealthTimeSeries data.
    Subclasses are expected to implement the `simulate` method.
    """

    def __init__(self):
        # No internal state yet; subclass may define its own parameters
        pass

    def simulate(self) -> ClimateHealthTimeSeries:
        """
        Run the simulation and return a ClimateHealthTimeSeries object.

        Returns:
            ClimateHealthTimeSeries: Simulated time series data for climate and health.
        """
        raise NotImplementedError("Subclasses must implement simulate()")


class IsDiseaseCaseSimulator(Protocol):
    """
    Protocol interface for any simulator that computes disease cases from climate data.
    Allows structural typing for plug-and-play compatibility with downstream components.
    """

    def simulate(self, climate_data: ClimateData) -> HealthData:
        """
        Given ClimateData, return simulated HealthData.

        Parameters:
            climate_data (ClimateData): Input time series for climate variables.

        Returns:
            HealthData: Simulated disease case time series.
        """
        ...
