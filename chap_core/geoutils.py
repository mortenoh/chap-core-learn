# Improvement Suggestions:
# 1. **Type Hinting Consistency**: Ensure consistent and specific type hints for all function arguments and return values. For instance, `inspect_feature_collection`'s `collection` parameter should be explicitly typed (e.g., `FeatureCollectionModel`).
# 2. **Robust Error Handling**: Enhance error handling in geometry processing functions. `feature_bbox` should have a fallback or raise an error for unhandled geometry types. Functions using `shapely` or `topojson` should catch potential exceptions from these libraries.
# 3. **Handling "GeometryCollection" in `feature_bbox`**: The `feature_bbox` function does not explicitly handle the "GeometryCollection" type. If features can be GeometryCollections, their bounding box calculation logic should be added.
# 4. **Document Optional Dependencies for `render`**: The `render` function imports `matplotlib` and `PIL`. Clearly document these as optional dependencies required only for this visualization utility.
# 5. **Clarity on `simplify_topology` Advanced Parameters**: The comments in `simplify_topology` regarding `prequantize`, `presimplify`, etc., are useful. Either elaborate slightly on their potential use or remove them if they are not planned for implementation, to avoid confusion.

"""
Utility functions for working with geographic geometries and features.

This module provides a collection of helper functions for common geospatial tasks
such as calculating bounding boxes, buffering features, inspecting feature collections,
rendering polygons for visualization, and simplifying topologies. It primarily operates
on Pydantic GeoJSON models (`FeatureModel`, `FeatureCollectionModel`) and the custom
`Polygons` wrapper class.
"""

import io
import logging  # Added import for logger

from shapely.geometry import shape

from .api_types import FeatureCollectionModel, FeatureModel
from .geometry import Polygons

# Initialize logger for the module
logger = logging.getLogger(__name__)


def feature_bbox(feature: FeatureModel) -> tuple[float, float, float, float]:
    """
    Calculates the bounding box for a FeatureModel object.

    Parameters
    ----------
    feature : FeatureModel
        A `FeatureModel` object representing a feature with a geometry.

    Returns
    -------
    tuple
        A 4-tuple in the form of (xmin, ymin, xmax, ymax).

    Raises
    ------
    ValueError
        If the feature has no geometry or an unsupported geometry type.
    """
    if not feature.geometry:
        raise ValueError("Feature has no geometry to calculate bounding box.")

    geom = feature.geometry
    geotype = geom.type
    coords = geom.coordinates

    if geotype == "Point":
        x, y = coords
        bbox = [x, y, x, y]
    elif geotype in ("MultiPoint", "LineString"):
        if not coords:
            raise ValueError(f"{geotype} has no coordinates.")
        xs, ys = zip(*coords)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    elif geotype == "MultiLineString":
        if not any(coords):
            raise ValueError(f"{geotype} has no coordinates.")
        xs = [x for line in coords for x, y in line]
        ys = [y for line in coords for x, y in line]
        if not xs:
            raise ValueError(f"{geotype} resulted in empty coordinate lists after flattening.")
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    elif geotype == "Polygon":
        if not coords or not coords[0]:
            raise ValueError(f"{geotype} has no coordinates for exterior ring.")
        exterior = coords[0]
        xs, ys = zip(*exterior)
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    elif geotype == "MultiPolygon":
        if not any(coords):
            raise ValueError(f"{geotype} has no coordinates.")
        xs = [x for poly_coords in coords for x, y in poly_coords[0]]  # Assumes poly_coords[0] is the exterior ring
        ys = [y for poly_coords in coords for x, y in poly_coords[0]]
        if not xs:
            raise ValueError(f"{geotype} resulted in empty coordinate lists after flattening.")
        bbox = [min(xs), min(ys), max(xs), max(ys)]
    # Consider adding GeometryCollection handling if necessary
    # elif geotype == "GeometryCollection":
    #     # Iterate through geometries, calculate individual bboxes, then find overall min/max
    #     all_bboxes = [feature_bbox(FeatureModel(type="Feature", geometry=g, properties={})) for g in geom.geometries]
    #     xmins, ymins, xmaxs, ymaxs = zip(*all_bboxes)
    #     bbox = [min(xmins), min(ymins), max(xmaxs), max(ymaxs)]
    else:
        raise ValueError(f"Unsupported geometry type for bbox calculation: {geotype}")
    return tuple(bbox)


def buffer_feature(feature: FeatureModel, distance: float) -> FeatureModel:
    """
    Creates a buffer around a FeatureModel object. Features with point and line geometries will become polygons.

    Parameters
    ----------
    feature : FeatureModel
        A `FeatureModel` object representing a feature with a geometry.
    distance : float
        The distance to buffer around the geometry, given in the same coordinate system units as the feature geometry.
        For latitude-longitude geometries, a distance of 0.1 is approximately 10 km at the equator.

    Returns
    -------
    FeatureModel
        A `FeatureModel` object with the buffered geometry.

    Raises
    ------
    ValueError
        If the feature has no geometry.
    Exception
        Can re-raise exceptions from `shapely` or Pydantic model validation.
    """
    if not feature.geometry:
        raise ValueError("Feature has no geometry to buffer.")

    try:
        # Pydantic models are not directly compatible with __geo_interface__ expected by shapely sometimes.
        # Dumping to dict first is safer.
        feature_geoj = feature.model_dump()
        shp = shape(feature_geoj["geometry"])  # shapely.geometry.shape expects a dict-like geo_interface
        shp_buffered = shp.buffer(distance)
        feature_geoj["geometry"] = shp_buffered.__geo_interface__
        feature_buffered = FeatureModel.model_validate(feature_geoj)
        return feature_buffered
    except Exception as e:
        logger.error(f"Error buffering feature: {e}")
        raise  # Re-raise the original error for detailed traceback


def buffer_point_features(collection: FeatureCollectionModel, distance: float) -> FeatureCollectionModel:
    """
    For a given FeatureCollection, creates a buffer around point-type FeatureModel objects.
    Features with polygon or line geometries remain unaltered.

    Parameters
    ----------
    collection : FeatureCollectionModel
        A `FeatureCollectionModel` object representing a feature collection.
    distance : float
        The distance to buffer around the point geometries.
        For latitude-longitude geometries, a distance of 0.1 is approximately 10 km at the equator.

    Returns
    -------
    FeatureCollectionModel
        A new `FeatureCollectionModel` object with any point geometries converted to polygon buffers.

    Raises
    ------
    ValueError
        If `distance` is not positive when point features are present.
    """
    if not collection.features:
        return FeatureCollectionModel(type="FeatureCollection", features=[])  # Return empty collection

    features = []
    has_points = any("Point" in feature.geometry.type for feature in collection.features if feature.geometry)

    if has_points and (distance is None or distance <= 0):
        raise ValueError(
            f"Attempting to buffer point geometries but the buffer distance arg is not positive: {distance}"
        )

    for feature in collection.features:
        if feature.geometry and "Point" in feature.geometry.type:
            feature_buffered = buffer_feature(feature, distance=distance)
            features.append(feature_buffered)
        else:
            features.append(feature)  # Append non-point features or features without geometry as is

    collection_buffered = FeatureCollectionModel(type="FeatureCollection", features=features)
    return collection_buffered


def inspect_feature_collection(collection: FeatureCollectionModel) -> dict:
    """
    Inspect and return statistics of the contents of a FeatureCollectionModel object.

    Parameters
    ----------
    collection : FeatureCollectionModel
        A `FeatureCollectionModel` object representing a feature collection.

    Returns
    -------
    dict
        A `dict` object with basic count statistics of the different geometry types contained in the FeatureCollectionModel.
    """
    stats = {}
    stats["total_features"] = len(collection.features)
    stats["features_with_geometry"] = sum(1 for feat in collection.features if feat.geometry)
    stats["point_geometries"] = sum(
        1 for feat in collection.features if feat.geometry and "Point" in feat.geometry.type
    )
    stats["line_geometries"] = sum(
        1 for feat in collection.features if feat.geometry and "Line" in feat.geometry.type
    )  # Includes LineString, MultiLineString
    stats["polygon_geometries"] = sum(
        1 for feat in collection.features if feat.geometry and "Polygon" in feat.geometry.type
    )  # Includes Polygon, MultiPolygon
    # Could add counts for specific subtypes like MultiPoint, MultiLineString, MultiPolygon if needed
    return stats


def render(polygons: Polygons):
    """
    Simple utility to render a `Polygons` object on a map for inspecting and debugging purposes.
    Requires `matplotlib` and `Pillow (PIL)` to be installed.

    Parameters
    ----------
    polygons : Polygons
        A `Polygons` object representing the set of polygons to be rendered.

    Returns
    -------
    PIL.Image.Image
        The rendered map image.

    Raises
    ------
    ImportError
        If `matplotlib` or `Pillow` are not installed.
    """
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError as e:
        logger.error(f"Missing optional dependencies for render: {e}. Please install matplotlib and Pillow.")
        raise

    if not polygons or len(polygons) == 0:
        logger.warning("Render called with empty or no polygons. Returning None.")
        # Or create a blank image, or raise error, depending on desired behavior
        return None

    df = gpd.GeoDataFrame.from_features(polygons.__geo_interface__)
    fig, ax = plt.subplots(dpi=300)
    df.plot(ax=ax)
    ax.set_title("Rendered Polygons")  # Add a title for context
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Save to a BytesIO buffer
    buf = io.BytesIO()
    try:
        plt.savefig(buf, format="png", bbox_inches="tight")
    finally:  # Ensure figure is closed even if savefig fails
        plt.close(fig)

    # Load image from memory buffer
    buf.seek(0)
    img = Image.open(buf)
    return img


def simplify_topology(polygons: Polygons, threshold: float = None) -> Polygons:
    """
    Simplifies a `Polygons` object while preserving topology between adjacent polygons using `topojson`.

    Parameters
    ----------
    polygons : Polygons
        A `Polygons` object representing the set of polygons to be simplified.
    threshold : float, optional
        Coordinate distance threshold used for simplification (e.g., Douglas-Peucker).
        If None, it's auto-calculated as 0.001 times the longest dimension of the total bounding box.
        Units depend on the CRS of the input Polygons. For lat/lon, this is in decimal degrees.
        For more accurate simplification, use projected coordinates.
        Refer to topojson documentation for details on `toposimplify` parameter.

    Returns
    -------
    Polygons
        A new, simplified `Polygons` object with preserved topology.

    Raises
    ------
    ImportError
        If `topojson` library is not installed.
    ValueError
        If polygons object is empty or has no valid bounding box for auto-threshold.
    """
    try:
        import topojson as tp
    except ImportError as e:
        logger.error(f"Missing optional dependency for simplify_topology: {e}. Please install topojson.")
        raise

    if not polygons or len(polygons) == 0:
        raise ValueError("Cannot simplify empty Polygons object.")

    # auto calc threshold if not given
    if threshold is None:
        xmin, ymin, xmax, ymax = polygons.bbox
        if any(
            map(
                lambda val: val is None or not isinstance(val, (int, float)) or val == float("nan"),
                [xmin, ymin, xmax, ymax],
            )
        ):
            raise ValueError("Cannot auto-calculate threshold due to invalid bounding box on Polygons object.")

        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 and height <= 0:  # Handles case of single point or invalid bbox
            raise ValueError("Cannot auto-calculate threshold from zero-area bounding box.")

        longest_dim = max(width, height)
        threshold = longest_dim * 0.001
        logger.info(f"Auto-calculated simplification threshold: {threshold}")

    # Parameters for topojson.Topology.
    # `toposimplify` uses a Visvalingam-Whyatt variant by default if >0, or Douglas-Peucker if <0.
    # `prevent_oversimplify` is True by default in recent topojson versions.
    # For more control, specific simplification algorithms and parameters can be explored.
    # See https://mattijn.github.io/topojson/example/settings-tuning.html
    kwargs_topology = {
        "toposimplify": threshold,
        "prevent_oversimplify": True,
        # 'simplify_algorithm': 'dp', # Example: explicitly Douglas-Peucker
        # 'simplify_with': 'simplification', # Old parameter, toposimplify is preferred
    }

    try:
        # Create topology. __geo_interface__ is expected by topojson.
        topo = tp.Topology(polygons.__geo_interface__, **kwargs_topology)

        # Convert simplified topology back to GeoJSON FeatureCollection dict
        # The `topo.to_geojson()` method can convert specific objects, or the whole thing.
        # If we want the same set of features back, just simplified:
        simplified_geojson_fc_dict = topo.to_geojson(winding_order="CCW_RIGHT")  # Get standard GeoJSON

        # Return new Polygons object from the simplified GeoJSON dictionary
        return Polygons.from_geojson(simplified_geojson_fc_dict, id_property="id")  # Assuming 'id' is still relevant
    except Exception as e:
        logger.error(f"Error during topology simplification: {e}")
        raise  # Re-raise or wrap in a custom exception
