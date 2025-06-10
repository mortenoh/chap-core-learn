# Improvement Suggestions:
# 1. **Error Handling**: Enhance functions like `geojson_to_shape` and `geojson_to_graph` with `try-except` blocks to gracefully handle file I/O errors (e.g., `FileNotFoundError`, `gpd.io.file.DriverError`) and potential issues during GeoPandas/libpysal operations.
# 2. **Configurable ID Property in `NeighbourGraph`**: The `NeighbourGraph` currently hardcodes "id" as the column name for location identifiers. Make this configurable (e.g., via an `id_column` parameter in `__init__` and `from_geojson_file`) for greater flexibility with different GeoJSON sources.
# 3. **Graph File Format Documentation/Standardization**: The custom graph file format written by `to_graph_file` should be clearly documented within its docstring. For broader interoperability, consider an option to export to standard graph formats (e.g., adjacency list CSV, GML, GraphML).
# 4. **Type Hinting for `IO`**: Clarify the type of `IO` objects accepted (e.g., `IO[str]` for text streams or `IO[bytes]` for binary streams). Since GeoJSON is text-based, `IO[str]` is more precise.
# 5. **`LocationMapping` Exception Type**: Change `AssertionError` in `LocationMapping.name_to_index` to a more standard exception like `KeyError` or a custom `LocationNameNotFoundError` for better semantic error handling by calling code.

"""
This module provides utility functions and classes for working with GeoJSON data.

It includes functionalities for:
- Converting GeoJSON files to Esri Shapefiles.
- Generating spatial neighbor graphs from GeoJSON features using Queen contiguity.
- Mapping location names to numerical indices.
- Storing and exporting neighbor graph information.

It relies on libraries such as GeoPandas and libpysal.
"""

from pathlib import Path
from typing import IO, List, Union  # Added List for type hint

import geopandas as gpd
from libpysal.weights import Queen

from .exceptions import ChapCoreException  # For custom exceptions

# Define a more specific IO type for text files if GeoJSON is always text
TextIO = IO[str]


class GeoJSONProcessingError(ChapCoreException):
    """Custom exception for errors during GeoJSON processing."""

    pass


def geojson_to_shape(geojson_filename: Union[str, Path], shape_filename: Union[str, Path]) -> None:
    """
    Convert a GeoJSON file to an Esri Shapefile format.

    Args:
        geojson_filename (Union[str, Path]): Path to the input GeoJSON file.
        shape_filename (Union[str, Path]): Path for the output Shapefile (e.g., 'output.shp').

    Raises:
        GeoJSONProcessingError: If conversion fails due to file errors or GeoPandas issues.
    """
    try:
        gdf = gpd.read_file(geojson_filename)
        gdf.to_file(shape_filename)
        logger.info(f"Successfully converted '{geojson_filename}' to '{shape_filename}'.")
    except FileNotFoundError:
        logger.error(f"GeoJSON file not found: {geojson_filename}")
        raise GeoJSONProcessingError(f"Input GeoJSON file not found: {geojson_filename}")
    except Exception as e:  # Catching generic geopandas/fiona errors
        logger.error(f"Error converting GeoJSON to Shapefile: {e}")
        raise GeoJSONProcessingError(f"Failed to convert '{geojson_filename}' to Shapefile: {e}")


def geojson_to_graph(geojson_filename: Union[str, TextIO], graph_filename: Union[str, Path]) -> None:
    """
    Build a spatial neighbor graph from a GeoJSON file and save it.

    The graph is based on Queen contiguity and saved to a custom text format.

    Args:
        geojson_filename (Union[str, TextIO]): Path or file-like object (text mode)
                                               of the input GeoJSON data.
        graph_filename (Union[str, Path]): Path for the output graph file.

    Raises:
        GeoJSONProcessingError: If graph generation or saving fails.
    """
    try:
        graph = NeighbourGraph.from_geojson_file(geojson_filename)
        graph.to_graph_file(graph_filename)
        logger.info(f"Successfully generated graph from '{geojson_filename}' and saved to '{graph_filename}'.")
    except Exception as e:
        logger.error(f"Error processing GeoJSON to graph: {e}")
        # Determine if geojson_filename is a path for error message
        fname_str = (
            geojson_filename
            if isinstance(geojson_filename, str)
            else getattr(geojson_filename, "name", "Unnamed GeoJSON stream")
        )
        raise GeoJSONProcessingError(f"Failed to generate graph from '{fname_str}': {e}")


class LocationMapping:
    """
    Maps region names (identifiers) to 1-based integer indices and vice-versa.

    This is useful for associating human-readable location names with numerical indices
    often used in graph algorithms or array indexing.
    """

    def __init__(self, ordered_locations: List[str]):
        """
        Initializes the LocationMapping.

        Args:
            ordered_locations (List[str]): A list of unique location names/identifiers.
                                           The order determines the 1-based indexing.

        Raises:
            ValueError: If `ordered_locations` contains duplicate names.
        """
        if len(ordered_locations) != len(set(ordered_locations)):
            # Find duplicates for a more informative error message
            seen = set()
            duplicates = {loc for loc in ordered_locations if loc in seen or seen.add(loc)}  # type: ignore
            raise ValueError(f"Duplicate location names found: {duplicates}")

        self._location_map: Dict[int, str] = {i + 1: location for i, location in enumerate(ordered_locations)}
        self._reverse_map: Dict[str, int] = {v: k for k, v in self._location_map.items()}

    def name_to_index(self, name: str) -> int:
        """
        Convert a location name to its 1-based integer index.

        Args:
            name (str): The location name.

        Returns:
            int: The 1-based index of the location.

        Raises:
            KeyError: If the name is not found in the location map.
        """
        try:
            return self._reverse_map[name]
        except KeyError:
            # Consider logging this event if it's unexpected in some contexts
            # logger.warning(f"Location name '{name}' not found in map. Available names: {list(self._reverse_map.keys())}")
            raise KeyError(f"Location name '{name}' not found. Available names: {list(self._reverse_map.keys())}")

    def index_to_name(self, index: int) -> str:
        """
        Convert a 1-based integer index back to its location name.

        Args:
            index (int): The 1-based index.

        Returns:
            str: The corresponding location name.

        Raises:
            KeyError: If the index is not found in the location map.
        """
        try:
            return self._location_map[index]
        except KeyError:
            raise KeyError(f"Index {index} not found. Available indices: {list(self._location_map.keys())}")


class NeighbourGraph:
    """
    Represents a spatial neighbor graph derived from geographic data.

    The graph is typically built using Queen contiguity from a GeoDataFrame,
    meaning two regions are neighbors if they share at least one vertex.
    It uses a `LocationMapping` to handle region identifiers.

    Attributes:
        location_map (LocationMapping): Maps region IDs to indices.
    """

    def __init__(self, regions: gpd.GeoDataFrame, graph: Queen, id_column: str = "id"):
        """
        Initializes the NeighbourGraph.

        Args:
            regions (gpd.GeoDataFrame): GeoDataFrame containing the geometries and IDs of regions.
            graph (libpysal.weights.Queen): A Queen contiguity spatial weights object.
            id_column (str): The name of the column in `regions` GeoDataFrame that contains
                             the unique location identifiers. Defaults to "id".

        Raises:
            KeyError: If `id_column` is not found in `regions`.
        """
        if id_column not in regions.columns:
            raise KeyError(
                f"Specified id_column '{id_column}' not found in GeoDataFrame columns: {regions.columns.tolist()}"
            )

        self._regions: gpd.GeoDataFrame = regions
        self._graph: Queen = graph
        self.id_column: str = id_column  # Store id_column
        self.location_map: LocationMapping = LocationMapping(regions[self.id_column].tolist())

    def __str__(self) -> str:
        """
        Return a string representation of the graph's adjacency list.
        Keys are 0-based internal indices from `libpysal.weights.Queen`.
        """
        return str(dict(self._graph.neighbors))

    @classmethod
    def from_geojson_file(cls, geo_json_file: Union[str, TextIO], id_column: str = "id") -> "NeighbourGraph":
        """
        Load regions from a GeoJSON file and build a Queen contiguity graph.

        Args:
            geo_json_file (Union[str, TextIO]): Path or file-like object (text mode)
                                                containing the GeoJSON data.
            id_column (str): The name of the property in GeoJSON features to use as
                             the location identifier. This property will become a column
                             in the internal GeoDataFrame. Defaults to "id".

        Returns:
            NeighbourGraph: An instance of NeighbourGraph.

        Raises:
            GeoJSONProcessingError: If reading GeoJSON or building the graph fails.
            KeyError: If `id_column` (after becoming a GeoDataFrame column) is not found.
        """
        try:
            regions_gdf = gpd.read_file(geo_json_file)
            if id_column not in regions_gdf.columns:
                # Attempt to find a common ID field if 'id' is missing, or raise clearly.
                # Common alternatives: 'ID', 'name', 'NAME', 'GEOID'.
                # This simple check might not be robust enough for all GeoJSONs.
                potential_ids = [col for col in ["ID", "NAME", "name", "GEOID"] if col in regions_gdf.columns]
                if potential_ids:
                    logger.warning(
                        f"Specified id_column '{id_column}' not found. Using first potential alternative: '{potential_ids[0]}'"
                    )
                    id_column = potential_ids[0]
                else:
                    raise KeyError(
                        f"id_column '{id_column}' not found in GeoJSON properties, and no common alternatives detected. "
                        f"Available properties: {regions_gdf.columns.tolist()}"
                    )

            # Ensure the ID column is of string type for consistent mapping
            regions_gdf[id_column] = regions_gdf[id_column].astype(str)

            graph = Queen.from_dataframe(regions_gdf)
            return cls(regions_gdf, graph, id_column=id_column)
        except Exception as e:
            fname_str = (
                geo_json_file
                if isinstance(geo_json_file, str)
                else getattr(geo_json_file, "name", "Unnamed GeoJSON stream")
            )
            raise GeoJSONProcessingError(f"Failed to create NeighbourGraph from '{fname_str}': {e}")

    def to_graph_file(self, graph_filename: Union[str, Path]) -> bool:
        """
        Write the graph to a custom text file format.

        The format is:
        Line 1: <number_of_regions>
        Subsequent lines: <region_index> <N_neighbors> <neighbor1_index> ... <neighborN_index>
        All indices are 1-based, derived from the `LocationMapping`.

        Args:
            graph_filename (Union[str, Path]): Path to the output file.

        Returns:
            bool: True if writing was successful. (Consider void return or raising errors)

        Raises:
            IOError: If file writing fails.
        """
        try:
            with open(graph_filename, "w") as f:
                f.write(f"{len(self.location_map._location_map)}\n")

                # self._graph.neighbors: dict where keys are 0-indexed internal libpysal indices
                # (corresponding to row order of GeoDataFrame at graph creation)
                # and values are lists of 0-indexed neighbor internal indices.
                for zero_based_node_idx, zero_based_neighbor_indices in self._graph.neighbors.items():
                    # Get the original ID string for the current node using its 0-based GDF row index
                    node_id_str = self._regions[self.id_column].iloc[zero_based_node_idx]

                    # Convert this ID string to its 1-based mapped index
                    one_based_node_mapped_idx = self.location_map.name_to_index(node_id_str)

                    # Convert neighbor 0-based GDF row indices to their ID strings, then to 1-based mapped indices
                    one_based_neighbor_mapped_indices = []
                    for neighbor_gdf_idx in zero_based_neighbor_indices:
                        neighbor_id_str = self._regions[self.id_column].iloc[neighbor_gdf_idx]
                        one_based_neighbor_mapped_indices.append(self.location_map.name_to_index(neighbor_id_str))

                    neighbor_line_parts = [
                        one_based_node_mapped_idx,
                        len(one_based_neighbor_mapped_indices),
                    ] + one_based_neighbor_mapped_indices
                    f.write(" ".join(map(str, neighbor_line_parts)) + "\n")
            return True
        except IOError as e:
            logger.error(f"Failed to write graph file to '{graph_filename}': {e}")
            raise
        except KeyError as e:  # Catch potential KeyErrors from name_to_index if an ID is somehow not in map
            logger.error(f"ID mapping error while writing graph to '{graph_filename}': {e}")
            raise GeoJSONProcessingError(f"ID mapping error during graph file writing: {e}")
        except Exception as e:  # Catch other unexpected errors
            logger.error(f"An unexpected error occurred while writing graph to '{graph_filename}': {e}")
            raise GeoJSONProcessingError(f"Error writing graph file: {e}")


# Add logger if not already present at module level
import logging

logger = logging.getLogger(__name__)

# For type hint Dict
from typing import Dict
