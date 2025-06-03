# MET.no: Norwegian Meteorological Institute Weather APIs

## 1. Introduction to MET.no APIs

The Norwegian Meteorological Institute (MET Norway) provides a comprehensive suite of free and open weather data APIs. These APIs offer access to a wide range of meteorological data, including observations, forecasts, climatological data, and specific products like air quality forecasts and oceanographic data. The services are part of MET Norway's commitment to open data, aiming to foster innovation and public benefit.

**Key Purpose**: To provide developers, researchers, and the public with easy access to high-quality weather and climate data for Norway and surrounding areas, as well as global model data. This supports a wide array of applications, from weather forecasting apps and research projects to integration into various information systems.

**Relevance to CHAP-Core**: MET.no APIs can be a valuable data source for CHAP-Core, especially for projects focusing on Norway or regions covered by their specific models (e.g., Nordic regions, Arctic). They can provide localized weather forecasts, historical observations, and potentially specific environmental data (like air quality) that could be relevant for climate-health studies.

## 2. Key Characteristics of MET.no APIs

- **Producing Organization**: Norwegian Meteorological Institute (MET Norway).
- **Data Philosophy**: Strong emphasis on open data, with most services free to use under a Creative Commons license (typically CC BY 4.0 or similar, requiring attribution).
- **Geographical Coverage**:
  - High-resolution data primarily for Norway and adjacent Nordic/Arctic areas.
  - Global coverage for some forecast models (e.g., ECMWF data redistributed).
- **Temporal Coverage**:
  - Forecasts: Short-term to medium-term.
  - Observations: Historical data available, varying by station and parameter.
  - Climatological data: Long-term historical records.
- **Temporal Resolution**: Varies by product (e.g., hourly forecasts, 10-minute observations).
- **Data Formats**: Primarily JSON, XML. Some products might be available in other formats like NetCDF or GRIB for bulk download or specific model data.
- **API Types**: Mostly RESTful APIs.
- **Documentation**: Extensive documentation available on the MET Norway developer portal ([https://api.met.no/](https://api.met.no/) or similar official links).

## 3. Main Data/Services Available via APIs

MET Norway offers a diverse range of APIs. Key services include:

### a. LocationForecast (MET Nordic / Nowcast)

- **Description**: Provides detailed weather forecasts for specific geographic coordinates.
- **Variables**: Temperature, precipitation (amount, type, probability), wind speed and direction, humidity, cloud cover, air pressure, etc.
- **Resolution**: Typically hourly for the first few days.
- **Coverage**: High resolution for Norway and Nordic areas; can also provide data for global locations using underlying global models.
- **Versions**: `LocationForecastLTS` (Long Term Support) and more experimental versions.

### b. Frost API (Observations and Climate Data)

- **Description**: Provides access to historical weather observations from MET Norway's network of weather stations and climatological data.
- **Variables**: Temperature, precipitation, wind, snow depth, humidity, air pressure, etc.
- **Data Sources**: Weather stations, buoys, etc.
- **Functionality**: Query data by station ID, time period, and specific elements.

### c. WeatherIcon (Symbolic Forecasts)

- **Description**: Provides weather symbols for forecasts, suitable for graphical representation in applications.
- **Output**: Symbolic representation of weather conditions.

### d. TextForecast

- **Description**: Provides textual weather forecasts for defined regions or locations.

### e. OceanForecast

- **Description**: Provides forecasts for oceanographic parameters.
- **Variables**: Sea temperature, salinity, currents, wave height, ice concentration, etc.
- **Coverage**: Primarily Norwegian coastal and Arctic waters.

### f. AirQualityForecast

- **Description**: Provides forecasts for air quality parameters.
- **Variables**: Concentrations of pollutants like NO2, PM2.5, PM10, Ozone.
- **Coverage**: Major Norwegian cities and regions.

### g. Radar

- **Description**: Provides access to weather radar imagery (precipitation).
- **Format**: Typically image files (e.g., PNG).

### h. Satellite Imagery

- **Description**: Access to satellite images relevant for weather analysis.

### i. Climatological Data (via Frost API or specific portals)

- **Description**: Long-term climate statistics, normals, and historical series.

### j. Global Model Data (e.g., from ECMWF)

- MET Norway sometimes provides access to subsets or specific products derived from global models like ECMWF, though direct access to full ECMWF data is usually through Copernicus/ECMWF channels.

## 4. Data Access

Access to MET.no APIs is generally straightforward:

- **Developer Portal**: The primary source for documentation, API keys (if required for some services or higher rate limits), and terms of use is the official MET Norway developer portal (e.g., [https://api.met.no/](https://api.met.no/), [https://developer.yr.no/](https://developer.yr.no/) - YR.no is a service by MET Norway and NRK).
- **API Endpoints**: Each service has specific HTTP GET endpoints.
- **Authentication**:
  - Many services are open and only require a `User-Agent` identifying the application.
  - Some services or higher usage tiers might require an API key.
- **Rate Limiting**: APIs are subject to rate limits to ensure fair usage. Details are provided in the documentation.
- **Data Formats**: JSON is the most common response format, making it easy to integrate with web and mobile applications.
- **Example Usage (Conceptual for LocationForecast)**:

  ```
  # Pseudocode for a GET request
  # URL: https://api.met.no/weatherapi/locationforecast/2.0/compact?lat=60.10&lon=9.58
  # Headers: User-Agent: "MyApplication/1.0 myemail@example.com"

  # Expected JSON Response (simplified):
  # {
  #   "properties": {
  #     "timeseries": [
  #       {
  #         "time": "2024-05-30T12:00:00Z",
  #         "data": {
  #           "instant": { "details": { "air_temperature": 15.5 } },
  #           "next_1_hours": { "summary": { "symbol_code": "clearsky_day" }, "details": { "precipitation_amount": 0.0 } }
  #         }
  #       },
  #       // ... more time steps
  #     ]
  #   }
  # }
  ```

## 5. Strengths of MET.no APIs

- **Open and Free**: Most data is freely available, promoting wide usage.
- **High-Quality Data**: MET Norway is a reputable national meteorological service.
- **Comprehensive Documentation**: Generally good developer documentation.
- **Variety of Products**: Covers forecasts, observations, radar, ocean, air quality, etc.
- **Modern API Standards**: Use of REST and JSON facilitates easy integration.
- **High Resolution for Nordic Region**: Offers detailed forecasts and observations for its primary area of responsibility.
- **Active Development**: APIs are generally well-maintained and updated.

## 6. Limitations and Considerations

- **Geographical Focus**: While some global data is available, the highest resolution and most specialized products are focused on Norway and surrounding regions.
- **Rate Limiting**: Free tiers have rate limits; high-volume applications might need to discuss terms or look for premium options if available.
- **Attribution Requirement**: Users must comply with licensing terms, typically requiring attribution to MET Norway.
- **Data Availability**: Historical data availability can vary significantly by station and parameter, especially for older records.
- **API Versioning**: Users need to be aware of API versioning and potential deprecation of older versions.
- **Complexity**: The sheer number of available APIs and products can be initially overwhelming; users need to identify the specific service that meets their needs.
- **Forecast Model Specifics**: The accuracy and characteristics of forecasts depend on the underlying numerical weather prediction models used by MET Norway.

## 7. Usage in CHAP-Core

MET.no APIs could be integrated into CHAP-Core in several ways:

- **Localized Forecasts/Observations**: For studies or applications within Norway or the Nordic region, MET.no APIs can provide high-resolution current weather data and forecasts.
- **Input for Health Models**: Temperature, precipitation, humidity, and air quality data can be used as input variables for climate-health models.
- **Data Augmentation**: Can supplement other data sources like ERA5 or CHIRPS, especially for near real-time data or specific local conditions not well-captured by global datasets.
- **Air Quality Studies**: The AirQualityForecast API is particularly relevant for health impact studies related to air pollution.
- **Extreme Weather Alerts**: Forecast APIs could potentially be used to develop alerts for extreme weather conditions that might impact health.
- **Tooling**: CHAP-Core could include modules or adaptors to fetch and process data from MET.no APIs, similar to how it might handle GEE or other data sources.

## 8. Further Resources

- **MET Norway API Portal**: [https://api.met.no/](https://api.met.no/) (Primary portal for API documentation and status)
- **MET Norway Weather API Documentation**: [https://developer.yr.no/](https://developer.yr.no/) (YR.no is a service by MET Norway and NRK, often hosts relevant API docs)
- **MET Norway Main Website**: [https://www.met.no/](https://www.met.no/)
- **Frost API Documentation**: Specific documentation pages linked from the main API portal.
- **GitHub (MET Norway)**: MET Norway often shares tools, client libraries, or examples on GitHub ([https://github.com/metno](https://github.com/metno)).

Always refer to the official MET Norway developer portals for the most up-to-date information on API endpoints, parameters, data formats, licensing, and best practices.
