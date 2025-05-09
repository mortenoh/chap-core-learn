import pooch

# Import cleaning functions (assumed to be functions named after dataset keys)
from . import cleaners

# Mapping of dataset names to their corresponding remote URLs
urls = {"hydromet": "https://github.com/drrachellowe/hydromet_dengue/raw/main/data/data_2000_2019.csv"}


def get_file(url: str) -> str:
    """
    Downloads a file from the given URL using pooch and returns the local filepath.
    `known_hash=None` allows any file hash (use only if security isn't critical).
    """
    return pooch.retrieve(url, known_hash=None)


def fetch_and_clean(name: str):
    """
    Downloads the dataset by name and applies the corresponding cleaner.

    Parameters:
        name (str): The key in the `urls` dictionary, which must also match a function in `cleaners`.

    Returns:
        Cleaned dataset object (type depends on cleaner implementation).
    """
    # Download file using pooch
    filename = get_file(urls[name])

    # Dynamically resolve cleaner function from the `cleaners` module
    cleaner_func = getattr(cleaners, name)

    # Apply cleaner to downloaded file and return result
    return cleaner_func(filename)
