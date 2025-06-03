# Open-Meteo: Free and Open-Source Weather APIs

## 1. Introduction to Open-Meteo

Open-Meteo is a free, open-source weather API that provides access to a wide range of historical, forecast, and climate data. It aggregates data from numerous national and international weather services and numerical weather prediction (NWP) models, making it available through a simple and consistent API interface. The project emphasizes ease of use, open data principles, and high performance.

**Key Purpose**: To offer developers, researchers, and enthusiasts free and easy access to global weather data without restrictive licenses or fees. It aims to democratize access to weather information for various applications, including web and mobile apps, research, data analysis, and personal projects.

**Relevance to CHAP-Core**: Open-Meteo can be a highly valuable data source for CHAP-Core due to its global coverage, extensive range of variables (including historical and forecast data), and ease of integration. It can provide essential meteorological inputs for climate-health models across diverse geographical regions.

## 2. Key Characteristics of Open-Meteo

- **Producing Organization/Project**: Open-Meteo is an independent project, leveraging open data from various meteorological institutions.
- **Data Philosophy**: Strictly open-source and free for non-commercial and commercial use (under Creative Commons CC BY 4.0 or CC BY-NC 4.0 for some specific data sources like ECMWF). Attribution is required.
- **Geographical Coverage**: Global.
- **Temporal Coverage**:
  - Forecasts: Up to 16 days for standard models, longer for ensemble models or climate projections.
  - Historical Data: Varies by model and location, often going back several decades (e.g., ERA5 reanalysis back to 1940).
  - Climate Projections: Data from CMIP6 models.
- **Temporal Resolution**:
  - Forecasts: Hourly, daily.
  - Historical Data: Hourly, daily.
- **Data Formats**: Primarily JSON. Also offers CSV for some endpoints.
- **API Types**: RESTful HTTP GET APIs.
- **Documentation**: Comprehensive and interactive documentation available on the Open-Meteo website ([https://open-meteo.com/](https://open-meteo.com/)).

## 3. Main Data/Services Available via APIs

Open-Meteo provides several distinct API endpoints, each tailored to specific types of weather data:

### a. Forecast API

- **Description**: Provides detailed weather forecasts for any location worldwide.
- **Variables**: Temperature (air, apparent, dew point), precipitation (sum, probability, type), wind (speed, direction, gusts), humidity, cloud cover, pressure, solar radiation, snow depth, weather codes (WMO), and many more.
- **Models**: Aggregates data from multiple NWP models (e.g., ICON, GFS, ECMWF IFS, MET Norway, DWD). Users can often select preferred models.
- **Resolution**: Hourly data for up to 16 days.

### b. Historical Weather API

- **Description**: Provides historical weather data for any location worldwide.
- **Variables**: Similar to the forecast API, including temperature, precipitation, wind, humidity, pressure, solar radiation, snow depth.
- **Models/Sources**: Primarily based on reanalysis datasets like ERA5 and ERA5-Land.
- **Coverage**: Typically from 1940 (ERA5) or 1950 (ERA5-Land) to near-present.

### c. Marine Weather API

- **Description**: Provides marine forecasts.
- **Variables**: Wave height, period, direction; swell forecasts; sea surface temperature; ocean currents.
- **Coverage**: Global.

### d. Air Quality API

- **Description**: Provides air quality forecasts.
- **Variables**: PM10, PM2.5, Carbon Monoxide, Nitrogen Dioxide, Sulphur Dioxide, Ozone, Dust, Ammoniac, Aldehydes. European and North American specific pollutants.
- **Models**: Uses models like CAMS (Copernicus Atmosphere Monitoring Service).

### e. Geocoding API

- **Description**: A utility API to search for locations and get their latitude/longitude coordinates.
- **Functionality**: Search by city name, returns coordinates, country, admin regions, etc.

### f. Elevation API

- **Description**: Provides elevation data for given coordinates.

### g. Flood API

- **Description**: Provides river discharge and flood forecasts based on the GloFAS model from ECMWF.
- **Variables**: River discharge, flood warnings.

### h. Climate Change API (CMIP6 Data)

- **Description**: Provides access to downscaled climate change projection data from CMIP6 models.
- **Variables**: Temperature, precipitation, and other variables under different SSP scenarios.

## 4. Data Access

Accessing Open-Meteo APIs is designed to be very simple:

- **Website**: The official website ([https://open-meteo.com/](https://open-meteo.com/)) provides interactive API builders and detailed documentation for all endpoints.
- **API Endpoints**: All APIs are accessed via HTTP GET requests to specific URLs.
- **Parameters**: Users specify parameters like latitude, longitude, variables (e.g., `hourly=temperature_2m,precipitation`), date ranges, models, etc., directly in the URL query string.
- **Authentication**: No API key is required for most uses.
- **Rate Limiting**: Generous rate limits, designed to support most applications. High-volume users are encouraged to self-host or contact the project.
- **Data Formats**: JSON is the default. CSV can be requested for some endpoints by setting the `format=csv` parameter.
- **Example Usage (Forecast API)**:

  ```
  # URL for hourly temperature and precipitation for Oslo for the next 7 days:
  # https://api.open-meteo.com/v1/forecast?latitude=59.9139&longitude=10.7522&hourly=temperature_2m,precipitation&forecast_days=7

  # Expected JSON Response (simplified):
  # {
  #   "latitude": 59.91,
  #   "longitude": 10.75,
  #   "hourly_units": { "time": "iso8601", "temperature_2m": "°C", "precipitation": "mm" },
  #   "hourly": {
  #     "time": ["2024-05-30T00:00", "2024-05-30T01:00", ...],
  #     "temperature_2m": [10.0, 9.8, ...],
  #     "precipitation": [0.0, 0.1, ...]
  #   }
  # }
  ```

## 5. Strengths of Open-Meteo

- **Truly Free and Open-Source**: No fees for commercial or non-commercial use for most data; clear licensing.
- **Ease of Use**: Simple API design, no authentication hurdles, excellent interactive documentation.
- **Comprehensive Data**: Wide range of variables, models, and historical depth.
- **Global Coverage**: Provides data for any point on Earth.
- **Aggregation of Best Models**: Combines data from leading global and regional weather models.
- **High Performance**: APIs are optimized for speed and reliability.
- **Active Development**: Continuously adding new features, models, and datasets.
- **Community Support**: Active community and responsive developers.
- **Self-Hosting Option**: For very high-demand applications, the entire platform can be self-hosted.

## 6. Limitations and Considerations

- **Attribution Required**: Users must attribute Open-Meteo and, where specified, the original data providers (e.g., ECMWF for ERA5 data).
- **Data Source Reliance**: The quality and availability of data depend on the underlying models and institutions providing the source data.
- **Model Specifics**: Users should be aware of the characteristics, strengths, and weaknesses of the different weather models they choose to use data from.
- **Experimental Features**: Some newer APIs or parameters might be experimental.
- **No SLAs for Free Tier**: While reliable, the free public API does not come with formal Service Level Agreements.
- **Focus on Gridded Data**: Primarily provides data for specific coordinates (interpolated from model grids) rather than raw model grid files for large areas (though historical data can be requested for larger bounding boxes with some limitations).

## 7. Usage in CHAP-Core

Open-Meteo is an excellent candidate for integration into CHAP-Core:

- **Primary Data Source**: Could serve as a primary source for both forecast and historical weather data globally.
- **Input for Health Models**: Provides a rich set of variables (temperature, precipitation, humidity, air quality, solar radiation) crucial for climate-health modeling.
- **Flexibility**: Allows selection of different weather models, enabling comparison or use of best-available data for a region.
- **Historical Analysis**: Access to ERA5 via Open-Meteo simplifies fetching long-term historical data for trend analysis and model training.
- **Climate Change Impact Studies**: The CMIP6 data API can be used for assessing future health impacts under different climate scenarios.
- **Air Quality Integration**: The Air Quality API offers relevant data for studies on respiratory illnesses or other health issues linked to pollution.
- **Ease of Integration**: The simple JSON-based API can be easily integrated into Python-based workflows within CHAP-Core.

## 8. Further Resources

- **Open-Meteo Official Website**: [https://open-meteo.com/](https://open-meteo.com/) (Main source for documentation, API builders, and blog updates)
- **Open-Meteo GitHub**: [https://github.com/open-meteo](https://github.com/open-meteo) (Source code for the project, issue tracking)
- **ECMWF (for ERA5 data details)**: [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **Copernicus (CAMS for air quality, GloFAS for floods)**: Relevant Copernicus service websites.

Always refer to the official Open-Meteo website for the latest API specifications, available models, variable lists, and terms of use.
