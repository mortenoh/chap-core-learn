# Weather Satellites: An Overview

## 1. Introduction to Weather Satellites

Weather satellites are Earth-orbiting spacecraft equipped with instruments to observe and collect data about the Earth's atmosphere, oceans, and land surfaces. This data is crucial for weather forecasting, climate monitoring, and understanding various Earth system processes. Since the launch of the first weather satellite, TIROS-1, in 1960, satellite meteorology has revolutionized our ability to observe global weather patterns in near real-time.

Weather satellites provide a continuous and global view that is impossible to achieve with ground-based observation networks alone. They are indispensable tools for:

- Tracking storms (hurricanes, cyclones, severe thunderstorms).
- Monitoring cloud cover and type.
- Measuring atmospheric temperature and humidity profiles.
- Observing sea surface temperatures and ocean currents.
- Detecting wildfires, volcanic ash, and dust storms.
- Monitoring snow and ice cover.
- Providing data for numerical weather prediction (NWP) models.
- Long-term climate monitoring.

## 2. Types of Orbits

Weather satellites primarily operate in two main types of orbits:

### a. Geostationary Orbit (GEO)

- **Altitude**: Approximately 35,786 kilometers (22,236 miles) above the Equator.
- **Characteristics**:
  - Satellites in GEO orbit at the same angular speed as the Earth's rotation. This makes them appear stationary relative to a point on the Earth's surface.
  - Each GEO satellite continuously views a large portion of the Earth (approximately one-third of the globe).
  - A constellation of GEO satellites (e.g., 5-6) can provide near-global coverage of lower and middle latitudes.
- **Advantages**:
  - Continuous observation of the same geographic area, allowing for frequent imagery (e.g., every 5-15 minutes) to track rapidly developing weather systems like thunderstorms and hurricanes.
  - Ideal for short-term forecasting and nowcasting.
- **Disadvantages**:
  - Poor coverage of polar regions due to the oblique viewing angle.
  - Lower spatial resolution compared to polar-orbiting satellites due to the greater distance from Earth.
- **Examples**: NOAA's GOES series, EUMETSAT's Meteosat series, JMA's Himawari series, CMA's Fengyun (FY-2, FY-4) series.

### b. Polar Orbit (or Low Earth Orbit - LEO)

- **Altitude**: Typically 500 to 1,000 kilometers (300 to 600 miles).
- **Characteristics**:
  - Satellites in polar orbits travel from pole to pole (or near-pole to near-pole) as the Earth rotates beneath them.
  - Each orbit takes about 90-100 minutes.
  - A single polar-orbiting satellite can observe the entire Earth's surface over a period of about 12-24 hours.
  - Often sun-synchronous, meaning they pass over any given point on Earth at roughly the same local solar time each day.
- **Advantages**:
  - Provides global coverage, including detailed views of polar regions.
  - Higher spatial resolution due to closer proximity to Earth.
  - Can carry a wider variety of instruments, including those that require lower altitudes (e.g., some microwave sounders, scatterometers).
- **Disadvantages**:
  - Less frequent observation of a specific location (typically twice a day per satellite).
  - Not ideal for continuous monitoring of rapidly evolving weather systems.
- **Examples**: NOAA's POES/JPSS series, EUMETSAT's MetOp series, CMA's Fengyun (FY-1, FY-3) series, ESA's Sentinel series (some of which contribute to weather/climate).

## 3. Types of Instruments and Data

Weather satellites carry a suite of instruments (sensors) to measure different aspects of the Earth system, primarily by detecting electromagnetic radiation.

### a. Imagers

- **Function**: Capture images of the Earth in various spectral bands.
- **Common Spectral Bands**:
  - **Visible (VIS)**: Detects reflected sunlight, showing clouds, snow, ice, and land features. Similar to what the human eye sees. Only available during daylight hours.
  - **Infrared (IR)**: Detects thermal radiation emitted by the Earth's surface and atmosphere (clouds). Can be used day and night. Different IR bands are sensitive to different temperatures and can be used to estimate cloud-top temperatures (related to cloud height), sea surface temperature, and land surface temperature. Water vapor channels in the IR spectrum detect moisture in the mid to upper atmosphere.
  - **Near-Infrared (NIR)**: Used for vegetation monitoring, land/water discrimination.
- **Data Products**: Cloud imagery, cloud type, cloud motion vectors (for wind estimation), sea/land surface temperature, vegetation indices, snow/ice cover, detection of fires and volcanic ash.

### b. Sounders

- **Function**: Measure atmospheric profiles of temperature, humidity, and trace gases at various altitudes. They do this by measuring radiation emitted or absorbed by the atmosphere at specific wavelengths.
- **Types**:
  - **Infrared Sounders**: Measure upwelling infrared radiation at many narrow spectral channels.
  - **Microwave Sounders**: Measure microwave radiation. Have the advantage of being able to "see" through non-precipitating clouds, providing data in cloudy conditions where IR sounders are limited.
- **Data Products**: Vertical profiles of temperature and moisture, atmospheric stability indices, total precipitable water, ozone profiles. This data is critical for initializing NWP models.

### c. Other Specialized Instruments

- **Scatterometers**: Measure backscattered microwave radiation to estimate sea surface wind speed and direction.
- **Altimeters**: Measure sea surface height, significant wave height.
- **Radiometers**: Measure specific radiation components (e.g., Earth's radiation budget).
- **GPS Radio Occultation Sounders**: Use GPS signals passing through the atmosphere to derive temperature and humidity profiles.
- **Lightning Mappers**: Detect and map lightning flashes (primarily on geostationary satellites).

## 4. Key International Satellite Programs and Agencies

Several countries and international organizations operate weather satellites:

- **NOAA (National Oceanic and Atmospheric Administration, USA)**:
  - **GOES (Geostationary Operational Environmental Satellite)**: Provides coverage over the Americas and parts of the Atlantic and Pacific Oceans. (e.g., GOES-R series: GOES-16, GOES-17, GOES-18).
  - **POES (Polar-orbiting Operational Environmental Satellite)** / **JPSS (Joint Polar Satellite System)**: Provides global coverage. (e.g., NOAA-20, Suomi NPP).
- **EUMETSAT (European Organisation for the Exploitation of Meteorological Satellites)**:
  - **Meteosat**: Geostationary satellites covering Europe, Africa, the Middle East, and parts of the Atlantic/Indian Oceans. (e.g., Meteosat Second Generation - MSG, Meteosat Third Generation - MTG).
  - **MetOp (Meteorological Operational satellite)**: Polar-orbiting satellites providing global data. (e.g., MetOp-A, -B, -C; and the upcoming MetOp-SG).
- **JMA (Japan Meteorological Agency)**:
  - **Himawari**: Geostationary satellites covering East Asia and the Western Pacific. (e.g., Himawari-8, Himawari-9).
- **CMA (China Meteorological Administration)**:
  - **Fengyun (FY)**: A series of both geostationary (FY-2, FY-4) and polar-orbiting (FY-1, FY-3) satellites.
- **KMA (Korea Meteorological Administration)**:
  - **GEO-KOMPSAT**: Geostationary satellites.
- **Roscosmos (Russian Federal Space Agency) / Roshydromet (Russian Federal Service for Hydrometeorology and Environmental Monitoring)**:
  - **Meteor-M** (polar-orbiting), **Elektro-L** (geostationary).
- **ISRO (Indian Space Research Organisation)**:
  - **INSAT / Kalpana / GSAT**: Geostationary satellites.
  - **Oceansat / Resourcesat / Cartosat**: Polar-orbiting Earth observation satellites with meteorological applications.
- **ESA (European Space Agency)**:
  - Develops satellites for EUMETSAT (e.g., Meteosat, MetOp).
  - Operates Earth observation missions like the **Sentinel series** as part of the Copernicus Programme, some of which provide valuable data for weather and climate (e.g., Sentinel-3 for SST and ocean color, Sentinel-5P for atmospheric composition).

## 5. Applications of Weather Satellite Data

- **Weather Forecasting**:
  - Initializing NWP models with global atmospheric data.
  - Tracking and forecasting hurricanes, typhoons, and cyclones.
  - Identifying and monitoring severe thunderstorms, tornadoes.
  - Cloud analysis for aviation and general forecasting.
  - Fog and low cloud detection.
- **Climate Monitoring**:
  - Long-term records of sea surface temperature, sea ice extent, cloud cover, Earth's radiation budget, atmospheric temperature, and greenhouse gas concentrations.
  - Monitoring El Niño/La Niña events.
- **Environmental Applications**:
  - Wildfire detection and monitoring.
  - Volcanic ash plume tracking for aviation safety.
  - Dust storm monitoring.
  - Oil spill detection.
  - Vegetation health assessment (e.g., NDVI).
  - Snow and ice mapping, flood monitoring.
- **Health Applications (Indirect Links)**:
  - **Air Quality**: Satellite data (e.g., aerosol optical depth from MODIS, VIIRS; NO₂, SO₂ from TROPOMI on Sentinel-5P) can be used to estimate ground-level air pollution, which has significant health impacts.
  - **Vector-Borne Diseases**: Land surface temperature, vegetation indices, and soil moisture estimates from satellites can help identify areas suitable for vector habitats (e.g., mosquitoes, ticks).
  - **Heat Stress**: Land surface temperature data can help identify urban heat islands and areas prone to extreme heat.
  - **Dust Storms**: Monitoring dust plumes, which can carry pathogens or exacerbate respiratory conditions.
  - **Harmful Algal Blooms (HABs)**: Ocean color sensors can detect HABs, which can have human health impacts through seafood contamination or aerosol exposure.

## 6. Conceptual Python Examples for Working with Satellite Data

Accessing and processing raw satellite data can be complex. However, many agencies provide processed data products, and libraries like `SatPy` simplify working with various satellite formats. Cloud platforms (AWS, Google Cloud, Microsoft Planetary Computer) also host large archives of analysis-ready satellite data.

### Example 1: Using `SatPy` for Local Data (Conceptual)

`SatPy` is a Python library for reading, manipulating, and writing data from various meteorological satellite instruments.

```python
from satpy import Scene
from glob import glob
import matplotlib.pyplot as plt

# This example assumes you have downloaded satellite data files (e.g., for GOES or Meteosat)
# and SatPy knows how to read them (you might need to install specific readers).
# File paths and reader names are placeholders.

# Example: Glob for files from a specific satellite and time
# files = glob('/path/to/your/satellite_data/goes16_abi_level2_*.nc')
# reader_name = 'abi_l2b' # Example reader for GOES ABI Level 2

# --- MOCKING for demonstration as real data loading is complex ---
# In a real scenario, Scene(filenames=files, reader=reader_name) would load data.
# We'll simulate a very simple Scene object with a dummy composite.
class MockImage:
    def __init__(self, data, attrs):
        self.data = data
        self.attrs = attrs
    def plot(self, ax, cmap, vmin, vmax): # Simplified plot
        ax.imshow(self.data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(self.attrs.get('name', 'Mock Image'))

class MockScene:
    def __init__(self):
        self.composites = {}
    def load(self, composites_to_load):
        # Simulate loading a 'true_color' composite
        if 'true_color' in composites_to_load:
            import numpy as np
            # Create a dummy RGB-like image (single band for simplicity here)
            dummy_rgb_data = np.random.rand(256, 256) * 255
            self.composites['true_color'] = MockImage(dummy_rgb_data, {'name': 'True Color (Mocked)', 'units': 'DN'})
    def show(self, composite_name): # Simplified show
        if composite_name in self.composites:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            # For a real RGB, SatPy handles multi-band plotting.
            # Here, we'll just plot the dummy single band.
            self.composites[composite_name].plot(ax=ax, cmap='gray', vmin=0, vmax=255)
            # plt.show()
            print(f"Displayed mocked composite: {composite_name}")
        else:
            print(f"Composite {composite_name} not loaded/available.")

scn = MockScene()
# --- END MOCKING ---

try:
    # if files:
    #     scn = Scene(filenames=files, reader=reader_name)

        # Define composites or channels to load
        # Common composites: 'true_color', 'natural_color', 'airmass', 'day_microphysics'
        # Common channels: 'C01', 'C02', ..., 'C13' (for GOES ABI), or specific wavelengths
    composites_to_load = ['true_color'] # Example
    scn.load(composites_to_load)

    print("--- Satellite Data Loaded (Mocked Scene) ---")
    # print(scn.available_composite_names())
    # print(scn.available_dataset_names()) # For individual channels

        # Display a composite
    if 'true_color' in scn.composites: # Check if composite is loaded
        scn.show('true_color')

        # You can also access data as xarray DataArrays:
        # true_color_dataarray = scn['true_color']
        # print(true_color_dataarray)

    # else:
    #     print("No satellite data files found to process.")

except Exception as e:
    print(f"An error occurred with SatPy: {e}")
```

### Example 2: Accessing Sentinel-2 Data via Planetary Computer (Cloud-Optimized GeoTIFFs)

While Sentinel-2 is more for land observation, this illustrates accessing cloud-hosted satellite data, a common pattern. Similar approaches exist for weather satellite data on cloud platforms.

```python
import pystac_client
import planetary_computer
import rioxarray
import matplotlib.pyplot as plt

try:
    # Define area of interest (AOI) and time range
    aoi_geojson = {
        "type": "Polygon",
        "coordinates": [[
            [-76.1, 38.9], [-76.1, 39.1], [-75.9, 39.1], [-75.9, 38.9], [-76.1, 38.9]
        ]]
    } # Example: Small area near Chesapeake Bay
    time_range = "2023-07-01/2023-07-05"

    # Connect to the STAC API for Planetary Computer
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for Sentinel-2 L2A data
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=aoi_geojson,
        datetime=time_range,
        query={"eo:cloud_cover": {"lt": 20}} # Filter for low cloud cover
    )
    items = search.item_collection()
    print(f"Found {len(items)} Sentinel-2 items matching criteria.")

    if items:
        # Select the first item (least cloudy, often)
        item = items[0]
        print(f"Selected item: {item.id} from {item.datetime.strftime('%Y-%m-%d')}")

        # Get the href for a specific band, e.g., Red (B04)
        # For weather satellites, you'd look for relevant bands like VIS, IR channels.
        # asset_href = item.assets["B04"].href # Example for Sentinel-2 Red band

        # --- MOCKING for demonstration as this requires network and specific asset ---
        print("--- Conceptual: Accessing a specific band (e.g., Red) ---")
        # red_band = rioxarray.open_rasterio(asset_href)
        # print(red_band)

        # plt.figure(figsize=(8,8))
        # red_band.plot(cmap='Reds', robust=True)
        # plt.title(f"Sentinel-2 Red Band (B04) - {item.id}")
        # plt.show()
        print("Mocked: Would load and plot a band here if asset_href was real and accessible.")
        # --- END MOCKING ---
    else:
        print("No Sentinel-2 items found for the specified criteria.")

except Exception as e:
    print(f"An error occurred accessing Planetary Computer: {e}")
```

## 7. Strengths of Weather Satellites

- **Global and Regional Coverage**: Provide a synoptic view not possible with ground stations alone.
- **Continuous Monitoring (GEO)**: Essential for tracking rapidly developing weather.
- **High Resolution (LEO)**: Provides detailed imagery and soundings.
- **Diverse Data Types**: Measure a wide range of atmospheric, land, and ocean parameters.
- **Improved Forecasts**: Satellite data significantly improves the accuracy of NWP models.
- **Early Warnings**: Crucial for issuing warnings for severe weather, hurricanes, etc.

## 8. Limitations

- **Indirect Measurements**: Satellites measure radiation, from which physical parameters are inferred (retrieved). Retrievals can have errors and uncertainties.
- **Cloud Obstruction**: Visible and infrared sensors cannot see through thick clouds to the surface or lower atmosphere (microwave sensors can penetrate clouds better).
- **Spatial and Temporal Resolution Trade-offs**: GEO has good temporal but coarser spatial resolution; LEO has good spatial but poorer temporal resolution for a fixed point.
- **Data Volume**: Modern satellites generate vast amounts of data, requiring significant infrastructure for processing, storage, and dissemination.
- **Calibration and Validation**: Ensuring long-term consistency and accuracy of satellite data requires ongoing calibration and validation efforts.
- **Cost**: Designing, building, launching, and operating satellite missions is expensive.

## 9. Conclusion

Weather satellites are indispensable tools in modern meteorology, climate science, and environmental monitoring. They provide a wealth of data that enhances our understanding of the Earth system, improves weather forecasts, and supports a wide range of applications critical for safety, economy, and well-being, including growing applications in public health.
