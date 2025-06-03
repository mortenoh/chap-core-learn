# CHIRPS: Climate Hazards Group InfraRed Precipitation with Station data

## 1. Introduction to CHIRPS

CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is a quasi-global (50°S-50°N) rainfall dataset. It combines 0.05° resolution satellite imagery with in-situ station data to create gridded rainfall time series. CHIRPS is developed by the Climate Hazards Group at the University of California, Santa Barbara (UCSB) in collaboration with the U.S. Geological Survey (USGS) Earth Resources Observation and Science (EROS) Center.

**Key Purpose**: To provide high-resolution, long-term precipitation estimates, primarily for drought monitoring, trend analysis, and climate variability studies, especially in regions where ground-based observation networks are sparse.

**Relevance to CHAP-Core**: CHIRPS is a valuable data source for CHAP-Core, offering detailed precipitation data crucial for climate-health models. Its high spatial resolution and extensive temporal coverage are particularly useful for analyzing the relationship between rainfall patterns and health outcomes, such as vector-borne or water-borne diseases.

## 2. Key Characteristics of CHIRPS

- **Producing Organization**: Climate Hazards Group (CHG) at UCSB, and USGS EROS.
- **Temporal Coverage**:
  - Spans from 1981 to the near-present (updated regularly).
- **Temporal Resolution**:
  - Daily
  - Pentadal (5-day)
  - Dekadal (10-day)
  - Monthly
- **Spatial Resolution**:
  - 0.05 degrees (approximately 5.55 km at the equator).
- **Data Assimilation System**: CHIRPS methodology involves:
  1.  **CHIRP (Climate Hazards group InfraRed Precipitation)**: A preliminary satellite-only precipitation estimate based on Cold Cloud Duration (CCD) from thermal infrared satellite observations.
  2.  **Station Blending**: CHIRP data is then blended with in-situ precipitation data from a variety of global and regional sources (e.g., GTS, national meteorological services) to reduce bias and improve accuracy.
- **Input Observations**:
  - Satellite: Thermal Infrared (TIR) imagery from geostationary satellites (GOES, Meteosat, GMS/Himawari).
  - Station Data: Monthly and daily precipitation totals from numerous ground stations worldwide.
  - Atmospheric Model Rainfall Fields: Uses NOAA CFSv2 precipitation fields to help define precipitation patterns in areas with sparse station data.
- **Output Format**: Typically GeoTIFF, NetCDF.

## 3. Main Variables Available

CHIRPS primarily provides one key variable:

- **Precipitation**:
  - Accumulated precipitation (mm) for the respective temporal resolution (daily, pentadal, etc.).

## 4. Data Access

CHIRPS data is accessible through several platforms:

- **Climate Hazards Group (CHG) Data Portal**:
  - Website: [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps)
  - Provides direct download access via FTP and HTTP for various temporal resolutions and formats.
- **USGS EarthExplorer**:
  - Website: [https://earthexplorer.usgs.gov/](https://earthexplorer.usgs.gov/)
  - CHIRPS datasets can be found under the "Precipitation" category.
- **Google Earth Engine (GEE)**:

  - CHIRPS datasets are available in the GEE data catalog (e.g., `UCSB-CHG/CHIRPS/DAILY`, `UCSB-CHG/CHIRPS/PENTADAL`).
  - This allows for server-side processing and analysis, which is highly convenient for regional studies and integration with other geospatial data. CHAP-Core likely utilizes GEE for accessing CHIRPS.
  - Example GEE Python API usage:

    ```python
    import ee
    # ee.Authenticate() # if not already done
    ee.Initialize() # Initialize the library.

    chirps_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
                        .filterDate('2023-01-01', '2023-01-31') \
                        .select('precipitation')

    # Example: Get mean precipitation over a region
    # region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
    # mean_precipitation = chirps_collection.mean().reduceRegion(
    #     reducer=ee.Reducer.mean(),
    #     geometry=region,
    #     scale=5566 # CHIRPS resolution in meters (approx)
    # ).get('precipitation')
    # print('Mean precipitation:', mean_precipitation.getInfo())
    ```

- **IRI Data Library (Columbia University)**:
  - Provides access to CHIRPS data along with analysis and visualization tools.
  - Website: [http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/](http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/)

## 5. Strengths of CHIRPS

- **High Spatial Resolution**: 0.05° resolution is beneficial for local and regional applications.
- **Long and Consistent Time Series**: Data from 1981 enables long-term trend analysis and climatology studies.
- **Good Performance in Data-Sparse Regions**: Specifically designed to provide reliable precipitation estimates in areas with limited ground station coverage, particularly in Africa, Asia, and Latin America.
- **Low Latency**: Updates are relatively frequent, making it suitable for near real-time monitoring.
- **Focus on Drought**: Widely used for agricultural drought monitoring and early warning systems.
- **Open Access**: Data is freely available to the public.
- **Extensive Validation**: The dataset has undergone considerable validation, particularly in drought-prone regions.

## 6. Limitations and Considerations

- **Precipitation Only**: CHIRPS is exclusively a precipitation dataset and does not include other climate variables like temperature, humidity, or wind.
- **Satellite Algorithm Limitations**:
  - The CCD method can struggle to accurately estimate precipitation from warm clouds (common in tropics) or in complex terrain (orographic effects).
  - Snowfall is not well captured; it's primarily a rainfall dataset.
- **Station Data Dependency**: The quality and accuracy of CHIRPS are influenced by the density and quality of the in-situ station data used for blending. In regions with very few or unreliable stations, accuracy may be lower.
- **Homogeneity**: Changes in satellite sensors, algorithms, or the station network over the long record could potentially introduce inhomogeneities, though efforts are made to mitigate these.
- **Extreme Events**: While generally good for accumulated rainfall, the representation of very localized, short-duration, high-intensity rainfall events might be smoothed or underestimated due to the nature of the input data and blending process.
- **Coastal and Oceanic Areas**: Primarily designed for land areas; accuracy over oceans or small islands might be limited.

## 7. CHIRPS vs. CHIRP and CHIRPS-GEFS

- **CHIRP (Climate Hazards group InfraRed Precipitation)**: This is the preliminary, satellite-only (TIR CCD based) precipitation product. It has higher latency but does not include station data. CHIRPS is the improved product that blends CHIRP with station data.
- **CHIRPS-GEFS**: This is a forecast product that blends historical CHIRPS data with precipitation forecasts from the Global Ensemble Forecast System (GEFS). It provides probabilistic precipitation forecasts up to several weeks ahead, useful for anticipatory drought action.

## 8. Usage in CHAP-Core

Given its strengths, CHIRPS is a highly relevant dataset for CHAP-Core:

- **Input Features**: Precipitation data from CHIRPS serves as a critical environmental covariate for models predicting the incidence or risk of climate-sensitive health outcomes (e.g., malaria, dengue fever, cholera).
- **Data Processing**: Within CHAP-Core, CHIRPS data would likely be:
  - Accessed via Google Earth Engine or direct downloads.
  - Extracted for specific administrative regions or study areas.
  - Aggregated or resampled from daily to other temporal resolutions (e.g., weekly, monthly) as required by specific health models.
  - Potentially used to derive other indicators like rainfall anomalies, dry spell duration, etc.
- **Model Development**: The high spatial resolution allows for more localized risk assessment, and the long time series supports the development of robust statistical models capturing climate-health relationships.
- **Early Warning Systems**: If CHAP-Core incorporates forecasting, CHIRPS-GEFS could be a valuable input for predicting potential upsurges in disease linked to forecasted rainfall conditions.

## 9. Further Resources

- **CHG CHIRPS Main Page**: [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps)
- **CHIRPS Scientific Data Article (Funk et al., 2015)**: Funk, C., Peterson, P., Landsfeld, M. et al. "The climate hazards infrared precipitation with stations". _Scientific Data_ 2, 150066 (2015). DOI: [10.1038/sdata.2015.66](https://doi.org/10.1038/sdata.2015.66)
- **USGS CHIRPS Information**: [https://www.usgs.gov/centers/eros/science/usgs-eros-archive-climate-hazards-group-infrared-precipitation-station-chirps](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-climate-hazards-group-infrared-precipitation-station-chirps)
- **Google Earth Engine Data Catalog (CHIRPS Daily)**: [https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_DAILY)
- **Google Earth Engine Data Catalog (CHIRPS Pentadal)**: [https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_PENTADAL](https://developers.google.com/earth-engine/datasets/catalog/UCSB-CHG_CHIRPS_PENTADAL)

This document provides an overview of the CHIRPS dataset. For the most current details, specific technical information, or access methods, refer to the official CHG and USGS documentation.
