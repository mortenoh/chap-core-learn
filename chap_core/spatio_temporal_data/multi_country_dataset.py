import tarfile
from pathlib import Path

import pooch  # Used to fetch remote files and cache them locally

from chap_core.datatypes import FullData
from chap_core.spatio_temporal_data.temporal_dataclass import DataSet


class MultiCountryDataSet:
    """
    A container for datasets from multiple countries, keyed by country name.
    Can be constructed from CSV files or tar.gz archives.
    """

    def __init__(self, data: dict[str, DataSet]):
        self._data = data

    def __getitem__(self, item):
        # Allow dictionary-style access (e.g. data['Denmark'])
        return self._data[item]

    @property
    def countries(self):
        # List of country names available in the dataset
        return list(self._data.keys())

    def keys(self):
        # Same as countries(), but dictionary-style
        return self._data.keys()

    @classmethod
    def from_tar(cls, url, dataclass=FullData):
        """
        Download and extract a tar.gz archive containing CSV files.
        Each CSV represents a country's dataset.
        """
        # Download the tar file using pooch (cached)
        tar_gz_file_name = pooch.retrieve(url, known_hash=None)

        # Open the tar file and extract all CSV members as file-like objects
        with tarfile.open(tar_gz_file_name, "r:gz") as tar_file:
            members = tar_file.getmembers()
            extracted_files = {Path(member.name).stem: tar_file.extractfile(member) for member in members}

            # Create a DataSet for each extracted file (if not None)
            data = {name: DataSet.from_csv(ef, dataclass) for name, ef in extracted_files.items() if ef is not None}

        return MultiCountryDataSet(data)

    def items(self):
        # Yield (country_name, DataSet) pairs
        return self._data.items()

    @classmethod
    def from_folder(cls, folder_path, dataclass=FullData):
        """
        Load CSV files from a local folder. Each file must be named <country>.csv.
        """
        csv_files = folder_path.glob("*.csv")

        # Map each file to its DataSet, keyed by the filename stem
        data = {file.stem: DataSet.from_csv(file, dataclass) for file in csv_files}

        return MultiCountryDataSet(data)

    @property
    def period_range(self):
        # Return the period range of the first country's dataset
        return list(self._data.values())[0].period_range

    def restrict_time_period(self, time_period):
        """
        Return a new MultiCountryDataSet restricted to a given time period.
        Applies restriction to each country's dataset individually.
        """
        return MultiCountryDataSet({name: data.restrict_time_period(time_period) for name, data in self._data.items()})


class LazyMultiCountryDataSet:
    """
    Like MultiCountryDataSet, but loads data lazily from tar.gz on demand.
    Only the requested file is loaded into memory.
    """

    def __init__(self, url, dataclass=FullData):
        self.url = url
        self.dataclass = dataclass
        self.__file_content = None  # Not used but could be for cache
        self.__file_name = None  # Will store resolved file path

    def _file_name(self):
        # Lazily resolve and cache file path
        if self.__file_name is None:
            self.__file_name = pooch.retrieve(self.url, known_hash=None)
        return self.__file_name

    def __getitem__(self, item):
        """
        Load a single country’s DataSet from the tar.gz archive based on name.
        """
        with tarfile.open(self._file_name(), "r:gz") as tar_file:
            members = tar_file.getmembers()

            # Find and extract the matching file
            extracted_file = next(tar_file.extractfile(member) for member in members if Path(member.name).stem == item)

            return DataSet.from_csv(extracted_file, self.dataclass)

    def items(self):
        """
        Generator yielding (country_name, DataSet) pairs, one at a time.
        Useful for memory-efficient iteration over large archives.
        """
        with tarfile.open(self._file_name(), "r:gz") as tar_file:
            for member in tar_file.getmembers():
                ef = tar_file.extractfile(member)
                if ef is not None:
                    yield Path(member.name).stem, DataSet.from_csv(ef, self.dataclass)
