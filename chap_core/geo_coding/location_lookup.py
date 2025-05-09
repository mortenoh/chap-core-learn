from geopy.geocoders import ArcGIS, Nominatim

from chap_core.datatypes import Location
from chap_core.services.cache_manager import get_cache


class LocationLookup:
    """
    Geocodes location names using Nominatim or ArcGIS, with caching support.
    Can be queried for geolocation presence and coordinates.
    """

    def __init__(self, geolocator: str = "Nominatim"):
        """
        Initialize the location lookup system with a selected geolocation service.

        Parameters:
            geolocator (str): The geolocation backend to use, either 'Nominatim' or 'ArcGIS'.
        """
        self.dict_location: dict[str, object] = {}

        if geolocator == "ArcGIS":
            self.geolocator = ArcGIS()
        elif geolocator == "Nominatim":
            self.geolocator = Nominatim(user_agent="chap_core")
        else:
            raise ValueError(f"Unsupported geolocator: {geolocator}")

    def add_location(self, location_name: str) -> None:
        """
        Manually add a location by name to the internal dictionary via geocoding.
        No-op if the location already exists.
        """
        if location_name not in self.dict_location:
            location = self.geolocator.geocode(location_name)
            if location:
                self.dict_location[location_name] = location
                self._add_cache_location(location_name, location)

    def __contains__(self, location_name: str) -> bool:
        """
        Check if a location can be geocoded (from memory, cache, or live lookup).

        Returns:
            bool: True if the location is available, otherwise False.
        """
        return (
            location_name in self.dict_location
            or self._get_cache_location(location_name)
            or self._fetch_location(location_name)
        )

    def __getitem__(self, location_name: str) -> Location:
        """
        Retrieve the Location object (lat/lon) for the given name.

        Raises:
            KeyError: If the location cannot be resolved.
        """
        if location_name not in self and location_name not in self.dict_location:
            raise KeyError(location_name)

        resolved = self.dict_location[location_name]
        return Location(latitude=resolved.latitude, longitude=resolved.longitude)

    def __str__(self) -> str:
        """
        Returns a string representation of all currently stored locations.
        """
        return str(self.dict_location)

    def _generate_cache_key(self, geolocator: object, location_name: str) -> str:
        """
        Generate a unique cache key using the geolocator's domain and location name.
        """
        return f"{getattr(geolocator, 'domain', 'unknown')}_{location_name}"

    def _add_cache_location(self, location_name: str, location: object) -> None:
        """
        Store a successfully geocoded location in the cache.
        """
        cache = get_cache()
        key = self._generate_cache_key(self.geolocator, location_name)
        cache[key] = location

    def _get_cache_location(self, location_name: str) -> bool:
        """
        Attempt to retrieve the location from the cache.

        Returns:
            bool: True if found, False otherwise.
        """
        cache = get_cache()
        key = self._generate_cache_key(self.geolocator, location_name)
        cached = cache.get(key)

        if cached:
            self.dict_location[location_name] = cached
            return True
        return False

    def _fetch_location(self, location_name: str) -> bool:
        """
        Attempt live geocoding for the location and cache it if found.

        Returns:
            bool: True if found, False otherwise.
        """
        location = self.geolocator.geocode(location_name)
        if location:
            self.dict_location[location_name] = location
            self._add_cache_location(location_name, location)
            return True
        return False

    def try_connection(self) -> None:
        """
        Performs a test geocode to check if the geolocation service is responsive.
        """
        self.geolocator.geocode("Oslo")
