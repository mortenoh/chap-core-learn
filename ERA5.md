# ERA5: ECMWF Atmospheric Reanalysis

## 1. Introduction to ERA5

ERA5 is the fifth generation atmospheric reanalysis of the global climate produced by the European Centre for Medium-Range Weather Forecasts (ECMWF). It provides a globally complete and consistent hourly description of the Earth's atmosphere, land surface, and ocean waves from 1940 to near real-time. Reanalysis combines vast amounts of historical observations from satellites and in-situ stations with advanced weather forecasting models using data assimilation techniques to produce a comprehensive, gridded dataset.

**Key Purpose**: To provide a detailed historical record of weather and climate variables, supporting climate monitoring, research, education, and various applications in sectors like agriculture, energy, and health.

**Relevance to CHAP-Core**: ERA5 is a crucial data source for CHAP-Core, providing essential climate variables (e.g., temperature, precipitation, humidity) that serve as input features for climate-health models. Its global coverage and consistent long-term record are invaluable for analyzing historical climate patterns and their relationship with health outcomes.

## 2. Key Characteristics of ERA5

- **Producing Organization**: European Centre for Medium-Range Weather Forecasts (ECMWF), as part of the Copernicus Climate Change Service (C3S).
- **Temporal Coverage**:
  - Currently spans from 1 January 1940 to present (updated with a delay of about 5 days).
  - Preliminary extension back to 1850 is also being developed (ERA5 BE).
- **Temporal Resolution**:
  - Hourly analysis fields for atmospheric variables on model levels, pressure levels, and single levels.
  - Hourly surface and single-level fields.
  - 3-hourly for some ocean wave parameters.
- **Spatial Resolution**:
  - Atmospheric data: Approximately 31 km horizontal resolution (0.28125 degrees) on a reduced Gaussian grid (T639).
  - Ocean wave data: Approximately 40 km horizontal resolution.
- **Data Assimilation System**: Uses the Integrated Forecasting System (IFS) Cy41r2, operational at ECMWF in 2016.
- **Input Observations**: Assimilates a vast array of observations from satellites (radiance, scatterometer data, etc.), ground stations (SYNOP, METAR), radiosondes, aircraft, buoys, and ships. The quality and quantity of observations improve over time, especially with the advent of the satellite era (post-1979).
- **Output Format**: Typically GRIB (GRIdded Binary) or NetCDF.

## 3. Main Variables Available

ERA5 provides a comprehensive set of variables. These are broadly categorized:

### a. Surface / Single-Level Variables (most commonly used for impacts studies)

- **Temperature**:
  - 2m air temperature (t2m)
  - Skin temperature (skt)
  - Sea surface temperature (sst)
  - Soil temperature (various levels)
  - Lake temperatures
- **Precipitation**:
  - Total precipitation (tp) - accumulated
  - Convective precipitation
  - Large-scale precipitation
  - Snowfall
- **Humidity**:
  - 2m dewpoint temperature (d2m)
  - Relative humidity (derived)
- **Wind**:
  - 10m u-component of wind (u10)
  - 10m v-component of wind (v10)
  - 100m u-component of wind
  - 100m v-component of wind
- **Pressure**:
  - Surface pressure (sp)
  - Mean sea level pressure (msl)
- **Radiation**:
  - Surface net solar radiation (ssr)
  - Surface net thermal radiation (str)
  - Downward and upward components
- **Cloud Cover**:
  - Total cloud cover (tcc)
  - High, medium, low cloud cover
- **Evaporation & Runoff**:
  - Evaporation (e)
  - Surface runoff, sub-surface runoff
- **Soil Moisture**: Volumetric soil water (various layers)
- **Snow**: Snow depth (sd), snow density, snow water equivalent (swe)
- **Other**: Albedo, leaf area index, etc.

### b. Upper-Air / Pressure Level Variables

- Geopotential
- Temperature
- U and V components of wind
- Specific humidity, relative humidity
- Vertical velocity
- Ozone mass mixing ratio
- Cloud liquid water content, cloud ice water content
- (Available on multiple pressure levels, e.g., 1000 hPa to 1 hPa)

### c. Model Level Variables

- Similar variables as pressure levels but on the native model vertical grid.

### d. Ocean Wave Variables (from WAM model coupled to IFS)

- Significant height of combined wind waves and swell
- Mean wave direction
- Mean wave period
- And many more specific wave parameters.

## 4. Data Access

ERA5 data is primarily accessed through the Copernicus Climate Data Store (CDS):

- **CDS Website**: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)
  - Interactive web interface to browse and download data.
  - Requires registration.
- **CDS API**:
  - Python-based API (`cdsapi`) for programmatic access.
  - Allows users to script data requests, specifying variables, levels, area, time, and format.
  - Requests are queued and processed; data is then downloaded.
  - Example usage:
    ```python
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': ['2m_temperature', 'total_precipitation'],
            'year': '2023',
            'month': ['01', '02'],
            'day': ['01', '02', '03'],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [60, -10, 50, 2], # North, West, South, East
            'format': 'netcdf',
        },
        'download.nc')
    ```
- **Google Earth Engine (GEE)**:
  - ERA5 datasets (e.g., "ECMWF/ERA5_LAND/HOURLY", "ECMWF/ERA5/DAILY") are available in the GEE data catalog.
  - Allows for server-side processing and analysis without downloading raw data.
  - Convenient for regional analysis and integration with other geospatial datasets.
  - CHAP-Core utilizes GEE for accessing ERA5 data, as noted in `EXTERNAL.MD`.
- **Other Portals/Methods**: Some research institutions or national weather services might provide access to ERA5 subsets.

## 5. Strengths of ERA5

- **Consistency**: Provides a physically consistent dataset across time and space.
- **High Resolution**: Offers significantly higher spatial and temporal resolution compared to previous reanalyses (e.g., ERA-Interim).
- **Comprehensive Variables**: Includes a wide range of atmospheric, land, and ocean wave parameters.
- **Long Time Series**: Data from 1940 allows for long-term climate studies.
- **Improved Model and Data Assimilation**: Benefits from advancements in ECMWF's IFS model and data assimilation techniques.
- **Uncertainty Quantification**: Provides an ensemble of 10 members (at a lower resolution) to estimate uncertainty in the reanalysis.
- **Open Access**: Freely available through the Copernicus program.

## 6. Limitations and Considerations

- **Computational Cost**: The high resolution and volume of data can be challenging to download, store, and process.
- **Model Dependence**: As a model-based product, it can inherit biases or systematic errors from the underlying forecast model, especially in data-sparse regions or for variables not well-constrained by observations.
- **Observation Changes Over Time**: The quality and density of assimilated observations change over the reanalysis period (e.g., fewer satellite observations before 1979). This can introduce inhomogeneities or affect trend analysis.
- **Spin-up Issues**: For some land surface variables, there might be spin-up issues in the early part of the record.
- **Extreme Events**: While generally good, the representation of very localized or very intense extreme weather events might be smoothed due to model resolution.
- **Data Volume**: Hourly global data is very large. Users often need to select specific regions, time periods, and variables, or use aggregated versions (e.g., daily or monthly means).
- **ERA5-Land**: For land surface applications, ERA5-Land provides an enhanced land component at a higher resolution (approx. 9 km) by re-running the land surface model forced by ERA5 atmospheric fields. It often provides more accurate land surface variables.

## 7. ERA5 vs. ERA5-Land

- **ERA5**: The full atmospheric reanalysis.
- **ERA5-Land**: A land-only reanalysis driven by ERA5 atmospheric forcing. It provides a more detailed and often more accurate representation of land surface conditions (soil moisture, temperature, snowpack, etc.) at a higher spatial resolution (~9 km).
- **Recommendation**: For applications focused purely on land surface variables, ERA5-Land is often preferred if available and suitable for the study region and period. CHAP-Core might use either depending on the specific needs and variables.

## 8. Usage in CHAP-Core

As indicated in `EXTERNAL.MD` and various code modules (e.g., `chap_core.climate_data`, `chap_core.google_earth_engine`), CHAP-Core leverages ERA5 data, primarily through Google Earth Engine.

- **Input Features**: ERA5 variables like temperature, precipitation, humidity, and radiation are critical inputs for the health models developed and run within CHAP-Core.
- **Data Processing**: CHAP-Core likely includes functionalities to:
  - Fetch ERA5 data for specific regions (defined by polygons) and time periods.
  - Aggregate hourly data to daily or monthly resolutions as needed by models.
  - Perform unit conversions (e.g., Kelvin to Celsius for temperature, meter to mm for precipitation).
  - Harmonize ERA5 data with health data and other geospatial information.
- **Model Development**: The long and consistent record of ERA5 allows for training robust models that can capture climate-health relationships over various time scales.

## 9. Further Resources

- **ERA5 Main Page (ECMWF)**: [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **ERA5 Data Documentation (Copernicus Confluence)**: [https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation)
- **ERA5 Journal Article (Hersbach et al., 2020)**: "The ERA5 global reanalysis" - Quarterly Journal of the Royal Meteorological Society. (DOI: 10.1002/qj.3803)
- **ERA5-Land Documentation**: [https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation)
- **Copernicus Climate Data Store (CDS)**: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/)

This document provides a comprehensive overview of the ERA5 dataset. For specific technical details or the latest updates, always refer to the official ECMWF and Copernicus documentation.
