# OpenWeatherMap: Weather Data APIs

## 1. Introduction to OpenWeatherMap APIs

OpenWeatherMap is a popular online service that provides global weather data via a range of APIs. It offers current weather, forecasts (short to long-term), historical data, weather maps, and specialized APIs for air pollution, UV index, and more. OpenWeatherMap caters to a broad audience, from individual hobbyists and small application developers to large enterprises.

**Key Purpose**: To provide accessible and affordable weather data to a wide range of users for various applications, including weather widgets, mobile apps, web services, data analytics, and research.

**Relevance to CHAP-Core**: OpenWeatherMap can be a useful data source for CHAP-Core, offering global weather data that can be integrated into climate-health models. Its ease of use and variety of data products (current, forecast, historical, air pollution) make it a flexible option, particularly for projects requiring quick integration or broad geographical coverage.

## 2. Key Characteristics of OpenWeatherMap

- **Producing Organization**: OpenWeather Ltd.
- **Data Philosophy**: Offers a freemium model. A free tier provides basic access, while paid subscription plans offer higher API call limits, more detailed data, additional APIs, and better support.
- **Geographical Coverage**: Global.
- **Temporal Coverage**:
  - Current Weather: Real-time or near real-time.
  - Forecasts: From hourly forecasts for a few days up to 16-day daily forecasts, and longer-term (30-day) forecasts on higher tiers.
  - Historical Data: Available for several years back, depending on the subscription plan and specific API.
- **Temporal Resolution**:
  - Forecasts: Hourly, 3-hourly, daily.
  - Historical Data: Hourly, daily.
- **Data Formats**: Primarily JSON. XML and HTML are also supported for some endpoints.
- **API Types**: RESTful HTTP GET APIs.
- **Documentation**: Available on the OpenWeatherMap website ([https://openweathermap.org/api](https://openweathermap.org/api)).

## 3. Main Data/Services Available via APIs

OpenWeatherMap provides a wide array of APIs, often categorized by data type and subscription level:

### a. Current Weather Data

- **Description**: Access to current weather observations for any location.
- **Variables**: Temperature, feels like temperature, pressure, humidity, wind speed and direction, cloudiness, rain/snow volume, weather condition codes and icons.
- **Access**: Available on free and paid tiers.

### b. Forecast APIs

- **Hourly Forecast 4 days**: Hourly forecast data for 96 hours.
- **Daily Forecast 16 days / 30 days (paid)**: Daily aggregated forecast.
- **5 day / 3 hour Forecast**: Forecast data with a 3-hour step.
- **Variables**: Similar to current weather, plus precipitation probability.

### c. Historical Weather Data

- **Description**: Access to archived weather data.
  - **History API**: Bulk historical data for a specific location.
  - **Statistical Weather Data API (paid)**: Aggregated historical data (daily, monthly, yearly) for cities.
  - **History Bulk (paid)**: Bulk downloads of historical data.
- **Variables**: Temperature, precipitation, wind, etc.
- **Coverage**: Varies, often several years back.

### d. Weather Maps

- **Description**: Tile-based weather map layers for precipitation, clouds, temperature, pressure, wind.
- **Usage**: Suitable for integration into interactive maps (e.g., Leaflet, OpenLayers).

### e. Air Pollution API

- **Description**: Provides current, forecast, and historical air pollution data.
- **Variables**: CO, O3, NO2, SO2, PM2.5, PM10, NH3, and Air Quality Index (AQI).
- **Coverage**: Global.

### f. Geocoding API

- **Description**: Utility to convert city names or zip/post codes to geographical coordinates and vice-versa.

### g. UV Index API

- **Description**: Provides current, forecast, and historical UV index data.

### h. One Call API (versions 2.5 and 3.0 - 3.0 is subscription-based)

- **Description**: A consolidated API endpoint to get various weather data types (current, minute forecast for 1 hour, hourly forecast for 48 hours, daily forecast for 7/8 days, historical data for 5 previous days, weather alerts) in a single API call. Version 3.0 requires a subscription.

## 4. Data Access

- **Website & API Keys**: Users need to register on the OpenWeatherMap website ([https://openweathermap.org/](https://openweathermap.org/)) to get an API key (APPID). This key must be included in all API requests.
- **API Endpoints**: Specific URLs for each API product.
- **Parameters**: Location (city name, city ID, geographic coordinates, ZIP code), units (standard, metric, imperial), language, etc., are passed as URL query parameters along with the APPID.
- **Rate Limiting**: The free tier has significant rate limits (e.g., 60 calls/minute, 1,000,000 calls/month for One Call API 2.5). Paid plans offer higher limits.
- **Data Formats**: JSON is the most common and recommended format.
- **Example Usage (Current Weather API)**:

  ```
  # URL for current weather in London using API key:
  # https://api.openweathermap.org/data/2.5/weather?q=London,uk&appid=YOUR_API_KEY&units=metric

  # Expected JSON Response (simplified):
  # {
  #   "coord": { "lon": -0.1257, "lat": 51.5085 },
  #   "weather": [ { "id": 800, "main": "Clear", "description": "clear sky", "icon": "01d" } ],
  #   "main": {
  #     "temp": 15.0,
  #     "feels_like": 14.5,
  #     "pressure": 1012,
  #     "humidity": 70
  #   },
  #   "wind": { "speed": 3.6, "deg": 240 },
  #   "name": "London"
  # }
  ```

## 5. Strengths of OpenWeatherMap

- **Ease of Access**: Simple registration and API key usage.
- **Global Coverage**: Provides weather data for almost any location.
- **Variety of Data**: Offers a broad spectrum of weather parameters and API products.
- **Free Tier**: A functional free tier allows developers to test and build small applications.
- **Good Documentation**: Generally clear documentation with examples.
- **Large User Base**: Widely used, with many community resources and client libraries available.
- **Weather Maps**: Useful feature for visual representation of weather data.

## 6. Limitations and Considerations

- **API Key Requirement**: All calls require an API key.
- **Rate Limits and Costs**: The free tier has strict rate limits. Significant usage or access to more advanced data/APIs requires paid subscriptions, which can become costly.
- **Data Accuracy and Sources**: While generally reliable for common use cases, the exact sources and accuracy of data, especially for the free tier, might not always be as transparent or as rigorously validated as data from national meteorological services or premium providers like ECMWF directly.
- **Historical Data Limitations (Free Tier)**: Access to extensive historical data is typically a paid feature. The One Call API 2.5 (free) only offers 5 days of historical data.
- **API Versioning and Changes**: Users need to stay updated on API versions (e.g., One Call API 2.5 vs 3.0) as features and access terms can change.
- **Attribution**: While not always strictly enforced like CC licenses, their terms often imply or suggest attribution.

## 7. Usage in CHAP-Core

OpenWeatherMap could be utilized in CHAP-Core, particularly for:

- **Rapid Prototyping**: The free tier can be useful for quickly developing and testing models or applications that require weather data.
- **Global Applications**: When broad, global coverage is needed and the free/lower-tier data quality is sufficient.
- **Supplementary Data**: Can provide current weather or short-term forecasts to supplement other historical or specialized datasets.
- **Air Pollution Data**: The Air Pollution API could be valuable for health impact studies focusing on air quality.
- **Educational Purposes**: Useful for demonstrating how to integrate weather APIs into applications.
- **Consideration for Cost**: If CHAP-Core projects require high-volume API calls or extensive historical data, the costs associated with OpenWeatherMap's paid tiers would need to be evaluated against other data sources.

## 8. Further Resources

- **OpenWeatherMap API Documentation**: [https://openweathermap.org/api](https://openweathermap.org/api)
- **OpenWeatherMap Pricing/Subscription Plans**: [https://openweathermap.org/price](https://openweathermap.org/price)
- **OpenWeatherMap Blog/News**: [https://openweather.co.uk/blog/category/weather](https://openweather.co.uk/blog/category/weather) (Note: .co.uk for blog, .org for main site)
- **Account Registration/Dashboard**: [https://home.openweathermap.org/users/sign_up](https://home.openweathermap.org/users/sign_up)

Always check the official OpenWeatherMap website for the most current information on API offerings, pricing, terms of service, and data policies.
