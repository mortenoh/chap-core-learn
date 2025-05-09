from pathlib import Path
from typing import Literal

# Import data types and loading utilities from chap_core
from chap_core.datatypes import ClimateHealthTimeSeries, FullData
from chap_core.spatio_temporal_data.multi_country_dataset import MultiCountryDataSet
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class ExampleDataSet:
    """
    Represents an example dataset stored as a local CSV file.
    Defaults to using the ClimateHealthTimeSeries dataclass for parsing.
    """

    base_path = Path(__file__).resolve().parents[2] / "example_data"

    def __init__(self, name: str, dataclass=ClimateHealthTimeSeries):
        self._name = Path(name)
        self._dataclass = dataclass

    def filepath(self) -> Path:
        """Returns the full path to the CSV file."""
        return self.base_path / self._name.with_suffix(".csv")

    def load(self) -> DataSet:
        """Loads the dataset from a CSV file using the specified dataclass."""
        return DataSet.from_csv(self.filepath(), dataclass=self._dataclass)


class RemoteExampleDataSet:
    """
    Represents a remote dataset (e.g., a tar file hosted online).
    Uses MultiCountryDataSet.from_tar to load the contents.
    """

    def __init__(self, url: str):
        self._url = url

    def load(self) -> DataSet:
        """Loads the dataset from a remote tar archive."""
        return MultiCountryDataSet.from_tar(self._url)


class LocalDataSet(ExampleDataSet):
    """
    Specialization of ExampleDataSet pointing to a different local base path,
    used for full-scale datasets stored outside the main example_data folder.
    """

    base_path = Path(__file__).resolve().parents[4] / "Data"


# Names of available datasets, categorized into groups
dataset_names = [
    "hydro_met_subset",
    "hydromet_clean",
    "hydromet_10",
    "hydromet_5_filtered",  # This one uses a different dataclass (FullData)
]

local_datasets = [
    "laos_full_data",
    "uganda_data",
]

remote_datasets = {
    "ISIMIP_dengue_harmonized": "https://github.com/dhis2/chap-core/raw/dev/example_data/full_data.tar.gz"
}

# Literal type hint for static analysis or UI completion
DataSetType = Literal[*dataset_names, *local_datasets, *remote_datasets.keys()]

# Compose dataset registry combining:
# - Normal examples
# - Local large datasets
# - Remote downloadable datasets
datasets: dict[str, ExampleDataSet] = {
    name: ExampleDataSet(name) if name != "hydromet_5_filtered" else ExampleDataSet(name, FullData)
    for name in dataset_names
} | {name: LocalDataSet(name, FullData) for name in local_datasets}

# Add remote datasets into the registry
for name, url in remote_datasets.items():
    datasets[name] = RemoteExampleDataSet(url)
