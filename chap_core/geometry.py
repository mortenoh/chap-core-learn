# Improvement Suggestions:
# 1. **GADM URL Configuration**: The GADM data URL (`data_path`) is hardcoded. Consider making this a configurable constant or providing clear documentation about its version and source, especially if updates to GADM might break it.
# 2. **Enhanced Data Fetching Error Handling**: Functions using `pooch` (e.g., `get_data_file`) should include more specific error handling for network issues, hash mismatches (if `known_hash` were used), or if `pooch.retrieve` fails for other reasons.
# 3. **Clarify Custom Pydantic Models**: The purpose of `PFeatureModel` and `PFeatureCollectionModel` (with open `properties`) versus `DFeatureModel` and `DFeatureCollectionModel` (from `api_types`) should be clearly documented. Explain why these distinct versions are needed.
# 4. **Robustness of `add_id`**: The `add_id` function directly accesses `feature.properties[f"NAME_{admin_level}"]`. This could raise a `KeyError`. Implement a check (e.g., `get` with a default or a `try-except`) for this key.
# 5. **Error Handling in `Polygons._add_ids`**: The `try-except Exception` in `Polygons._add_ids` is broad. It should ideally catch more specific exceptions (e.g., `KeyError` if `id_property` is missing from `feature.properties`) and provide more context in error messages or allow features without the `id_property` if that's a permissible state.

"""
This module provides functionalities for fetching, processing, and managing
geographic polygon data, primarily focusing on administrative boundaries from GADM.

Key features include:
- Fetching country-level administrative boundary data from GADM.
- Normalizing names and assigning IDs to geographic features.
- Filtering and creating GeoJSON FeatureCollections for specific regions.
- A `Polygons` class to wrap FeatureCollections, providing utility methods
  for common geometric operations and data access.

It utilizes libraries like `pooch` for data fetching, `pycountry` for country
code lookups, `pydantic_geojson` for GeoJSON models, and `unidecode` for text normalization.
"""

import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # Added List, Any, Optional

import pooch
import pycountry
from pydantic_geojson import FeatureCollectionModel, FeatureModel
from unidecode import unidecode

from .api_types import FeatureCollectionModel as DFeatureCollectionModel
from .api_types import FeatureModel as DFeatureModel
from .geoutils import feature_bbox

logger = logging.getLogger(__name__)


# --- Custom Models with open 'properties' ---
class PFeatureModel(FeatureModel):
    """
    A Pydantic GeoJSON Feature model allowing for arbitrary 'properties'.

    This model extends the standard `pydantic_geojson.FeatureModel` by defining
    `properties` as a flexible `dict`, enabling it to parse GeoJSON features
    where the properties schema is not strictly defined beforehand or varies.
    """

    properties: Dict[str, Any]  # Allow any properties


class PFeatureCollectionModel(FeatureCollectionModel):
    """
    A Pydantic GeoJSON FeatureCollection model using `PFeatureModel`.

    This allows the collection to contain features with arbitrary properties.
    """

    features: List[PFeatureModel]


# --- Country Mapping and Data URL ---
# URL template for GADM version 4.1 JSON data.
GADM_DATA_URL_TEMPLATE = "https://geodata.ucdavis.edu/gadm/gadm41_{country_code}_{level}.json.zip"

# Predefined list of country names and their GADM (ISO 3166-1 alpha-3) codes.
# Used by get_all_data() if no other source is specified.
_COUNTRY_NAMES = [
    "brazil",
    "mexico",
    "el salvador",
    "paraguay",
    "peru",
    "colombia",
    "ecuador",
    "nicaragua",
    "panama",
    "argentina",
    "indonesia",
    "philippines",
    "thailand",
    "vietnam",
    "laos",
    "malaysia",
    "cambodia",
    "singapore",
]
_COUNTRY_CODES_L = [
    "BRA",
    "MEX",
    "SLV",
    "PRY",
    "PER",
    "COL",
    "ECU",
    "NIC",
    "PAN",
    "ARG",
    "IDN",
    "PHL",
    "THA",
    "VNM",
    "LAO",
    "MYS",
    "KHM",
    "SGP",
]
COUNTRY_CODE_MAP: Dict[str, str] = dict(zip(_COUNTRY_NAMES, _COUNTRY_CODES_L))


# --- Utility Functions ---
def normalize_name(name: str) -> str:
    """
    Normalizes a string by converting to lowercase, removing spaces, and transliterating to ASCII.

    Args:
        name (str): The input string to normalize.

    Returns:
        str: The normalized string.
    """
    return unidecode(name.replace(" ", "").lower())


def add_id(feature: PFeatureModel, admin_level: int = 1, lookup_dict: Optional[Dict[str, str]] = None) -> DFeatureModel:
    """
    Assigns an ID to a GeoJSON feature.

    The ID is derived from the 'NAME_<admin_level>' property of the feature.
    If a `lookup_dict` is provided, the extracted name is first normalized and
    then mapped through the dictionary to get the final ID.

    Args:
        feature (PFeatureModel): The input feature (with flexible properties).
        admin_level (int): The administrative level to use for extracting the name
                           (e.g., NAME_1 for provinces/states). Defaults to 1.
        lookup_dict (Optional[Dict[str, str]]): A dictionary mapping normalized names
                                                to desired final ID strings.

    Returns:
        DFeatureModel: A new feature model (from api_types) with the 'id' field populated.

    Raises:
        KeyError: If `f"NAME_{admin_level}"` is not in `feature.properties` or
                  if a name is not found in `lookup_dict` when provided.
    """
    name_key = f"NAME_{admin_level}"
    if name_key not in feature.properties:
        raise KeyError(f"Property '{name_key}' not found in feature properties: {feature.properties}")

    feature_id_val = feature.properties[name_key]
    if not isinstance(feature_id_val, str):  # Ensure it's a string before normalization
        feature_id_val = str(feature_id_val)

    if lookup_dict:
        normalized_name = normalize_name(feature_id_val)
        if normalized_name not in lookup_dict:
            raise KeyError(f"Normalized name '{normalized_name}' (from '{feature_id_val}') not found in lookup_dict.")
        feature_id_val = lookup_dict[normalized_name]

    return DFeatureModel(**feature.model_dump(), id=feature_id_val)


def get_area_polygons(country: str, regions: List[str], admin_level: int = 1) -> DFeatureCollectionModel:
    """
    Fetch GADM administrative boundaries for a given country and filter them for specified regions.

    Args:
        country (str): The name of the country (e.g., "Vietnam").
        regions (List[str]): A list of region names (e.g., provinces, states) to extract.
                             Names will be normalized for matching.
        admin_level (int): The GADM administrative level to fetch (default is 1, e.g., provinces/states).

    Returns:
        DFeatureCollectionModel: A GeoJSON FeatureCollection containing polygons for the requested regions.
                                 Features will have an 'id' field matching the original (non-normalized) region names.

    Raises:
        pycountry.LookupError: If the country name cannot be resolved to a country code.
        pooch.PoochError: If data fetching fails.
        zipfile.BadZipFile: If the downloaded file is not a valid zip archive.
        KeyError: If expected 'NAME_<admin_level>' property is missing in GADM data.
    """
    logger.info(f"Fetching polygons for country: '{country}', admin level: {admin_level}, regions: {regions}")
    country_gadm_data = get_country_data(country, admin_level)

    name_key = f"NAME_{admin_level}"
    # Build a dictionary mapping normalized GADM region names to their original PFeatureModel
    feature_dict: Dict[str, PFeatureModel] = {}
    for f in country_gadm_data.features:
        if name_key in f.properties:
            gadm_region_name = str(f.properties[name_key])  # Ensure string
            feature_dict[normalize_name(gadm_region_name)] = f
        else:
            logger.warning(f"Feature missing '{name_key}' property in {country} GADM data: {f.properties}")

    logger.debug(f"Available (normalized) polygon regions from GADM for {country}: {list(feature_dict.keys())}")
    normalized_requested_regions = {normalize_name(r): r for r in regions}
    logger.debug(f"Requested regions (normalized map): {normalized_requested_regions}")

    extracted_features: List[DFeatureModel] = []
    for norm_req_name, orig_req_name in normalized_requested_regions.items():
        if norm_req_name in feature_dict:
            # Use a simple lookup_dict that maps the normalized GADM name back to the original requested name for the ID
            # This ensures the output feature.id matches the user's input `regions` list.
            id_lookup = {norm_req_name: orig_req_name}
            extracted_features.append(add_id(feature_dict[norm_req_name], admin_level, id_lookup))
        else:
            logger.warning(
                f"Requested region '{orig_req_name}' (normalized: '{norm_req_name}') not found in GADM data for {country} at admin level {admin_level}."
            )

    return DFeatureCollectionModel(type="FeatureCollection", features=extracted_features)


def get_country_data_file(country: str, level: int = 1) -> str:
    """
    Get the local file path of the downloaded GADM data ZIP archive for a country by its common name.
    Uses `pycountry` to find the ISO 3166-1 alpha-3 code.

    Args:
        country (str): Common name of the country (e.g., "Vietnam").
        level (int): GADM administrative level (default is 1).

    Returns:
        str: Absolute path to the downloaded (or cached) ZIP file.

    Raises:
        pycountry.LookupError: If the country name is not found.
    """
    country_name_normalized = country.strip().capitalize()
    try:
        country_obj = pycountry.countries.get(name=country_name_normalized)
        if not country_obj:  # Handle cases where get returns None for some partial matches
            country_obj = pycountry.countries.search_fuzzy(country_name_normalized)[0]
        country_code = country_obj.alpha_3
    except LookupError:
        logger.error(f"Could not find country code for '{country_name_normalized}'.")
        raise
    return get_data_file(country_code, level)


def get_data_file(country_code: str, level: int = 1) -> str:
    """
    Download (or retrieve from cache) the GADM data ZIP archive for a given country ISO alpha-3 code and admin level.

    Args:
        country_code (str): ISO 3166-1 alpha-3 country code (e.g., "VNM").
        level (int): GADM administrative level (default is 1).

    Returns:
        str: Absolute path to the downloaded (or cached) ZIP file.

    Raises:
        pooch.PoochError: If downloading or retrieval fails.
    """
    url = GADM_DATA_URL_TEMPLATE.format(country_code=country_code.upper(), level=level)
    logger.info(f"Retrieving GADM data from URL: {url}")
    # `known_hash=None` means pooch won't verify checksum, useful if hashes change or are unavailable.
    # For production, providing known_hash is recommended for data integrity.
    try:
        file_path = pooch.retrieve(url, known_hash=None, progressbar=True)
        return str(file_path)  # Ensure string path
    except Exception as e:  # Catch pooch specific errors if possible, or general ones
        logger.error(f"Failed to retrieve data for country code {country_code}, level {level} from {url}: {e}")
        raise  # Re-raise as is, or wrap in a custom PoochError/DownloadError


def get_country_data(country: str, admin_level: int) -> PFeatureCollectionModel:
    """
    Fetch, extract, and parse a GeoJSON FeatureCollection for a country from the GADM ZIP archive.

    Args:
        country (str): Common name of the country.
        admin_level (int): GADM administrative level.

    Returns:
        PFeatureCollectionModel: A Pydantic model representing the GeoJSON FeatureCollection.

    Raises:
        Various exceptions from underlying calls (e.g., `FileNotFoundError`, `zipfile.BadZipFile`, `JSONDecodeError`).
    """
    zip_filename_path = get_country_data_file(country, admin_level)
    logger.info(f"Extracting GeoJSON from ZIP archive: {zip_filename_path}")

    try:
        with zipfile.ZipFile(zip_filename_path) as z:
            if not z.namelist():
                raise FileNotFoundError(f"ZIP archive '{zip_filename_path}' is empty or corrupted.")
            json_filename_in_zip = z.namelist()[0]  # Assumes the first file is the correct JSON
            logger.debug(f"Reading '{json_filename_in_zip}' from ZIP.")
            with z.open(json_filename_in_zip) as f:
                json_bytes = f.read()
                return PFeatureCollectionModel.model_validate_json(json_bytes)
    except zipfile.BadZipFile:
        logger.error(f"File '{zip_filename_path}' is not a valid ZIP archive.")
        raise
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from '{json_filename_in_zip}' in '{zip_filename_path}'.")
        raise


def get_all_data() -> iter(Tuple[str, PFeatureCollectionModel]):
    """
    Generator that yields (country_name, feature_collection) pairs for all countries
    defined in the module's `COUNTRY_CODE_MAP`. Fetches admin level 1 data.
    """
    logger.info("Fetching GADM data for all predefined countries.")
    for country_name in COUNTRY_CODE_MAP.keys():
        try:
            logger.debug(f"Processing data for {country_name}...")
            yield country_name, get_country_data(country_name, admin_level=1)
        except Exception as e:
            logger.error(f"Failed to get data for country '{country_name}': {e}")
            # Decide whether to continue to next country or stop
            # yield country_name, None # Optionally yield None or skip


# --- Polygon Wrapper Class ---
class Polygons:
    """
    A wrapper around a DFeatureCollectionModel (from api_types) providing
    utility methods for common operations like filtering, saving to file,
    and accessing geometric properties (e.g., bounding box).
    """

    def __init__(self, polygons: DFeatureCollectionModel):
        """
        Initializes the Polygons wrapper.

        Args:
            polygons (DFeatureCollectionModel): The GeoJSON FeatureCollection to wrap.
        """
        self._polygons = polygons

    def __eq__(self, other: Any) -> bool:
        """Checks equality based on the underlying FeatureCollectionModel."""
        if not isinstance(other, Polygons):
            return NotImplemented
        return self._polygons == other._polygons

    def __len__(self) -> int:
        """Returns the number of features in the collection."""
        return len(self._polygons.features)

    def __iter__(self) -> iter(DFeatureModel):
        """Returns an iterator over the features in the collection."""
        return iter(self._polygons.features)

    @property
    def data(self) -> DFeatureCollectionModel:
        """Provides access to the underlying DFeatureCollectionModel."""
        return self._polygons

    @property
    def __geo_interface__(self) -> Dict[str, Any]:
        """
        Returns the GeoJSON representation as a dictionary, implementing the __geo_interface__ protocol.
        """
        return self.to_geojson()

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        Calculates the overall bounding box (xmin, ymin, xmax, ymax) of all features in the collection.
        """
        if not self._polygons.features:
            return (float("nan"), float("nan"), float("nan"), float("nan"))  # Or raise error

        # feature_bbox is assumed to take a DFeatureModel and return (xmin, ymin, xmax, ymax)
        boxes = [feature_bbox(f) for f in self._polygons.features if f.geometry]  # Ensure geometry exists
        if not boxes:  # All features might lack geometry
            return (float("nan"), float("nan"), float("nan"), float("nan"))

        xmins, ymins, xmaxs, ymaxs = zip(*boxes)
        return min(xmins), min(ymins), max(xmaxs), max(ymaxs)

    def to_geojson(self) -> Dict[str, Any]:
        """
        Serializes the wrapped FeatureCollection to a GeoJSON-compliant dictionary.
        """
        return self._polygons.model_dump(exclude_none=True)  # exclude_none for cleaner output

    def feature_collection(self) -> DFeatureCollectionModel:
        """
        Returns the underlying DFeatureCollectionModel instance.
        (Identical to the `data` property).
        """
        return self._polygons

    def to_file(self, filename: str | Path):
        """
        Saves the FeatureCollection to a GeoJSON file.

        Args:
            filename (str | Path): The path to the output file.
        """
        with open(filename, "w") as f:
            json.dump(self.to_geojson(), f, indent=2)  # Add indent for readability
        logger.info(f"Polygons saved to file: {filename}")

    @classmethod
    def from_file(cls, filename: str | Path, id_property: str = "id") -> "Polygons":
        """
        Loads a Polygons instance from a GeoJSON file.

        Args:
            filename (str | Path): Path to the GeoJSON file.
            id_property (str): The property name in each feature's properties
                               to use for populating the feature's 'id' field.
                               Defaults to "id".

        Returns:
            Polygons: A new Polygons instance.
        """
        logger.info(f"Loading Polygons from file: {filename}, using id_property: '{id_property}'")
        with open(filename, "r") as f:
            geojson_data = json.load(f)
        return cls.from_geojson(geojson_data, id_property=id_property)

    @classmethod
    def from_geojson(cls, geojson: Dict[str, Any], id_property: str = "id") -> "Polygons":
        """
        Creates a Polygons instance from a GeoJSON dictionary.

        Features are validated against `DFeatureModel`. If `id_property` is specified,
        it attempts to populate the `id` field of each feature from its properties.

        Args:
            geojson (Dict[str, Any]): A GeoJSON FeatureCollection dictionary.
            id_property (str): The property name to use for feature IDs. Defaults to "id".

        Returns:
            Polygons: A new Polygons instance.
        """
        parsed_features: List[DFeatureModel] = []
        errors_count = 0

        for feat_dict in geojson.get("features", []):
            try:
                # Validate first, then try to add ID if necessary
                validated_feature = DFeatureModel.model_validate(feat_dict)
                parsed_features.append(validated_feature)
            except Exception as e:
                logger.warning(f"Skipping invalid GeoJSON feature during load: {e}. Feature data: {feat_dict}")
                errors_count += 1

        if errors_count > 0:
            logger.warning(f"Total skipped invalid GeoJSON features: {errors_count}")

        # Create a temporary collection to pass to _add_ids
        temp_collection = DFeatureCollectionModel(type="FeatureCollection", features=parsed_features)
        final_collection = cls._add_ids(temp_collection, id_property)  # _add_ids modifies features in place
        return cls(final_collection)

    @classmethod
    def _add_ids(cls, collection: DFeatureCollectionModel, id_property: str) -> DFeatureCollectionModel:
        """
        Populates the 'id' field of each feature in a DFeatureCollectionModel.

        The ID is taken from the feature's properties, using the key specified by `id_property`.
        If a feature's `id` is already set, it's preserved. If the `id_property` is not found
        or an error occurs, an error is logged, and the original exception is re-raised.
        Values are unidecoded.

        Args:
            collection (DFeatureCollectionModel): The feature collection whose features need IDs.
            id_property (str): The key in each feature's properties dictionary to use as the ID.

        Returns:
            DFeatureCollectionModel: The same collection, with feature IDs potentially updated.

        Raises:
            KeyError: If `id_property` is not found in a feature's properties and `feature.id` is not already set.
        """
        for feature in collection.features:
            if feature.id is None:  # Only set if not already present
                try:
                    if feature.properties and id_property in feature.properties:
                        id_value = feature.properties[id_property]
                        # Ensure id_value is a string before unidecode
                        feature.id = unidecode(str(id_value))
                    else:
                        # If id_property is critical and missing, this could be an error
                        logger.debug(
                            f"Property '{id_property}' not found for feature, ID remains None. Properties: {feature.properties}"
                        )
                except Exception as e:  # Catch any error during ID assignment
                    logger.error(
                        f"Failed to assign ID from property '{id_property}' to feature: {e}. Feature properties: {feature.properties}"
                    )
                    # Decide whether to raise, skip, or assign a default ID
                    raise  # Re-raise the caught exception to signal failure
        return collection

    def id_to_name_tuple_dict(self) -> Dict[str, Tuple[str, str]]:
        """
        Creates a dictionary mapping feature IDs to a tuple of (name, parent).

        'name' is taken from feature.properties['name'] or defaults to feature.id.
        'parent' is taken from feature.properties['parent'] or defaults to "-".

        Returns:
            Dict[str, Tuple[str, str]]: A dictionary where keys are feature IDs and
                                        values are (name, parent) tuples.
        """
        lookup: Dict[str, Tuple[str, str]] = {}
        for f in self._polygons.features:
            if f.id is None:  # Skip features without an ID
                logger.warning(f"Feature found without an ID during id_to_name_tuple_dict creation: {f.properties}")
                continue

            name = f.id  # Default name to ID
            parent = "-"  # Default parent

            if f.properties:
                name = str(f.properties.get("name", f.id))  # Ensure string
                parent = str(f.properties.get("parent", "-"))  # Ensure string

            lookup[f.id] = (name, parent)
        return lookup

    def get_parent_dict(self) -> Dict[str, str]:
        """
        Creates a dictionary mapping feature IDs to their 'parent' property.

        If a feature has no 'id', or no 'properties', or 'parent' property,
        the parent defaults to "-".

        Returns:
            Dict[str, str]: A dictionary where keys are feature IDs and values are parent identifiers.
        """
        parent_map: Dict[str, str] = {}
        for f in self._polygons.features:
            if f.id is None:
                logger.warning(f"Feature found without an ID during get_parent_dict creation: {f.properties}")
                continue

            parent_val = "-"
            if f.properties and "parent" in f.properties:
                parent_val = str(f.properties["parent"])  # Ensure string

            parent_map[f.id] = parent_val
        return parent_map

    def filter_locations(self, locations: List[str]) -> "Polygons":
        """
        Filters the features in the collection to include only those whose IDs are in the `locations` list.

        Args:
            locations (List[str]): A list of feature IDs to retain.

        Returns:
            Polygons: A new Polygons instance containing only the filtered features.
        """
        if not locations:  # Optimization: if locations list is empty, return empty Polygons
            return Polygons(DFeatureCollectionModel(type="FeatureCollection", features=[]))

        # Using a set for faster lookups if `locations` can be large
        location_set = set(locations)
        filtered_features = [f for f in self._polygons.features if f.id is not None and f.id in location_set]

        if len(filtered_features) < len(locations):
            found_ids = {f.id for f in filtered_features}
            missing_ids = location_set - found_ids
            if missing_ids:
                logger.warning(f"Some requested locations for filtering were not found: {missing_ids}")

        return Polygons(DFeatureCollectionModel(type="FeatureCollection", features=filtered_features))


# --- Script usage ---
if __name__ == "__main__":
    # Example: Fetch and save data for all predefined countries
    # Ensure the base directory exists
    output_base_dir = Path("./gadm_data_output")  # Changed to a local path for example
    output_base_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Script execution: Fetching GADM data to '{output_base_dir.resolve()}'")

    for country_key, country_data_fc in get_all_data():
        output_file_path = output_base_dir / f"{country_key}_admin1.json"
        try:
            with open(output_file_path, "w") as f_out:
                # country_data_fc is PFeatureCollectionModel, dump it
                json.dump(country_data_fc.model_dump(exclude_none=True), f_out, indent=2)
            logger.info(f"Successfully saved data for {country_key} to {output_file_path}")
        except Exception as e_main:
            logger.error(f"Error in __main__ processing {country_key}: {e_main}")
