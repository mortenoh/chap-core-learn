# OpenAQ: Aggregating Global Air Quality Data

## 1. Introduction to OpenAQ

**OpenAQ** is a non-profit organization that aggregates and harmonizes publicly available, real-time and historical air quality data from around the world. Its mission is to fight air inequality by opening up air quality data, making it universally accessible and usable. OpenAQ provides a free, open-source platform that collects data from government monitoring stations, research-grade instruments, and validated low-cost sensor networks.

By providing a centralized and standardized way to access disparate air quality datasets, OpenAQ empowers scientists, journalists, policymakers, and citizens to understand and address air pollution.

## 2. Key Characteristics and Goals

- **Mission**: To enable a global community to combat air pollution by providing open, real-time, and historical air quality data.
- **Data Aggregation**: Collects data from numerous official sources (e.g., national environmental protection agencies) and other validated providers.
- **Data Harmonization**: Standardizes units and parameter names to allow for easier comparison and analysis across different data sources.
- **Open Access**: All data on the platform is freely available under open licenses (often CC BY 4.0 for the aggregated data, with original source licenses also noted).
- **Real-time and Historical Data**: Provides both current air quality readings and access to archived data.
- **Global Coverage**: Aims for comprehensive global coverage, continually adding new data sources.
- **Community-Driven**: Fosters a community of users and contributors.

## 3. Data Available on OpenAQ

OpenAQ primarily focuses on data from physical sensors that measure ambient (outdoor) air pollution. Key parameters typically include:

- **Particulate Matter**:
  - **PM2.5**: Fine particulate matter (≤ 2.5 micrometers)
  - **PM10**: Particulate matter (≤ 10 micrometers)
  - Occasionally PM1 (≤ 1 micrometer) or other particle metrics if provided by the source.
- **Gaseous Pollutants**:
  - **Ozone (O₃)**
  - **Carbon Monoxide (CO)**
  - **Sulfur Dioxide (SO₂)**
  - **Nitrogen Dioxide (NO₂)**
  - Other nitrogen oxides (NOx, NO)
- **Meteorological Data**: Some sources co-locate meteorological sensors, so parameters like temperature, relative humidity, pressure, wind speed, and wind direction might also be available, though this is not the primary focus.

**Data Granularity**:

- Data is typically available at an hourly or sub-hourly resolution, depending on the reporting frequency of the original source.
- Historical data availability varies by source and location.

## 4. Accessing OpenAQ Data

OpenAQ provides several ways to access its data:

### a. OpenAQ Website ([https://openaq.org/](https://openaq.org/))

- **Explorer Map**: An interactive map to browse monitoring locations and view current and historical air quality data.
- **Data Download**: Options to download data for specific locations or countries directly from the website, often in CSV format.
- **Source Information**: Provides metadata about the original data sources.

### b. OpenAQ API ([https://docs.openaq.org/](https://docs.openaq.org/))

- **RESTful API**: The primary way for programmatic access. It allows users to query data based on various parameters like location (coordinates, country, city), pollutant type, date range, and specific monitoring stations.
- **Endpoints**:
  - `/latest`: Get the latest measurements.
  - `/measurements`: Get historical measurements.
  - `/locations`: Get information about monitoring locations.
  - `/sources`: Get information about data sources.
  - `/parameters`: List available air quality parameters.
- **Authentication**: While some basic access might be anonymous, registering for a free API key is generally recommended for higher rate limits and more stable access.
- **Response Format**: Primarily JSON.

### c. OpenAQ Data on Cloud Platforms

- **Amazon S3**: OpenAQ makes its full historical archive available as part of the AWS Open Data Sponsorship Program. Data is often stored in Parquet or CSV format, allowing for efficient bulk analysis using tools like AWS Athena, Spark, or Python with Dask.

### d. Third-Party Tools and Libraries

- Various community-developed tools and client libraries in different programming languages (including Python) can simplify interaction with the OpenAQ API.

## 5. Relevance and Use Cases

OpenAQ data is valuable for a wide range of applications:

- **Public Health Research**:
  - Studying the health impacts of air pollution exposure.
  - Linking air quality data with health outcomes (e.g., hospital admissions, respiratory illnesses).
  - Assessing exposure disparities across different communities.
- **Environmental Science**:
  - Analyzing air pollution trends and patterns.
  - Validating air quality models.
  - Studying the impact of events like wildfires or industrial accidents on air quality.
- **Journalism and Advocacy**:
  - Reporting on air quality issues and raising public awareness.
  - Holding polluters and governments accountable.
- **Software and Application Development**:
  - Building air quality dashboards, mobile apps, and alert systems.
- **Policy Making**:
  - Informing the development of air quality management strategies and regulations.
- **Citizen Science**:
  - Enabling individuals and communities to access and understand local air quality.

## 6. Conceptual Python Examples for Accessing OpenAQ Data

### Example 1: Using the OpenAQ API with `requests`

```python
import requests
import pandas as pd
import json

# It's good practice to use an API key if you have one.
# Store it securely, e.g., in an environment variable or config file.
# API_KEY = "YOUR_OPENAQ_API_KEY"
# headers = {"X-API-Key": API_KEY} # If you have a key

# For this example, we'll try without a key (lower rate limits)
headers = {}

# --- Get latest measurements for a specific city (e.g., London) ---
try:
    params_latest = {
        "city": "London",
        "parameter": "pm25", # Focus on PM2.5
        "limit": 5,          # Get the 5 most recent PM2.5 readings in London
        "sort": "desc",
        "order_by": "datetime"
    }
    response_latest = requests.get("https://api.openaq.org/v2/latest", params=params_latest, headers=headers)
    response_latest.raise_for_status() # Raise an exception for HTTP errors
    latest_data = response_latest.json()

    print("--- Latest PM2.5 Measurements for London (Sample) ---")
    if latest_data['results']:
        for record in latest_data['results']:
            print(f"Location: {record['location']}, Value: {record['value']} {record['unit']}, Time: {record['measurements'][0]['lastUpdated']}")
    else:
        print("No latest PM2.5 data found for London with these parameters.")

except requests.exceptions.RequestException as e:
    print(f"Error fetching latest data: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response for latest data.")
except KeyError as e:
    print(f"Unexpected data structure in latest data response: {e}")


# --- Get historical measurements for a specific location ID and parameter ---
# First, you might need to find a location ID.
# You can use the /locations endpoint or the OpenAQ website.
# Let's assume we found a location ID for a station in, say, Paris.
# This is a HYPOTHETICAL location ID.
location_id_paris_station = 1234 # Replace with a real ID from OpenAQ

# For this example, let's use a known public location ID if the above hypothetical one doesn't work
# Example: A location in Delhi, IN (often has data, but check OpenAQ for current valid IDs)
# This is still illustrative, IDs can change or locations can go offline.
# It's best to query /locations first.
# For demonstration, let's try to find a location ID for "Delhi" and "pm25"
try:
    params_locations = {
        "city": "Delhi",
        "parameter": "pm25",
        "limit": 1,
        "country": "IN" # Be specific
    }
    response_locations = requests.get("https://api.openaq.org/v2/locations", params=params_locations, headers=headers)
    response_locations.raise_for_status()
    locations_data = response_locations.json()

    actual_location_id = None
    if locations_data['results']:
        actual_location_id = locations_data['results'][0]['id']
        print(f"\nFound location ID for Delhi PM2.5 station: {actual_location_id}")
    else:
        print("\nCould not find a PM2.5 station ID for Delhi via API for historical data example.")

    if actual_location_id:
        params_historical = {
            "location_id": actual_location_id,
            "parameter": "pm25",
            "date_from": "2024-01-01T00:00:00Z", # Start date
            "date_to": "2024-01-02T23:59:59Z",   # End date (e.g., 2 days of data)
            "limit": 100, # Max 10000 per page, adjust as needed
            "sort": "asc",
            "order_by": "datetime"
        }
        response_historical = requests.get("https://api.openaq.org/v2/measurements", params=params_historical, headers=headers)
        response_historical.raise_for_status()
        historical_data = response_historical.json()

        print(f"\n--- Historical PM2.5 Measurements for Location ID {actual_location_id} (Sample) ---")
        if historical_data['results']:
            df_historical = pd.DataFrame(historical_data['results'])
            df_historical['datetime'] = pd.to_datetime(df_historical['date'].apply(lambda x: x['utc']))
            df_historical = df_historical[['datetime', 'value', 'unit', 'parameter']].set_index('datetime')
            print(df_historical.head())

            # Plotting (requires matplotlib)
            # import matplotlib.pyplot as plt
            # df_historical['value'].plot(figsize=(10,5))
            # plt.title(f"PM2.5 at Location ID {actual_location_id}")
            # plt.ylabel(f"PM2.5 ({df_historical['unit'].iloc[0] if not df_historical.empty else 'µg/m³'})")
            # plt.xlabel("Datetime (UTC)")
            # plt.show()
        else:
            print(f"No historical PM2.5 data found for location ID {actual_location_id} in the specified date range.")

except requests.exceptions.RequestException as e:
    print(f"Error during historical data retrieval: {e}")
except json.JSONDecodeError:
    print("Error decoding JSON response for historical data.")
except KeyError as e:
    print(f"Unexpected data structure in historical data response: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

**Note**: The OpenAQ API (v2) has rate limits. For extensive data pulls, using their AWS S3 archive or batch processing with API calls respecting limits is advisable. Always check the [OpenAQ API documentation](https://docs.openaq.org/) for the latest endpoints, parameters, and best practices.

## 7. Strengths of OpenAQ

- **Openness and Accessibility**: Truly open data and open-source platform.
- **Global Scope**: Aggregates data from a vast number of countries and stations.
- **Data Harmonization**: Simplifies working with data from diverse sources.
- **Community Focus**: Strong community engagement and support.
- **Real-time Data**: Provides up-to-date information where available from sources.

## 8. Limitations and Considerations

- **Data Source Dependency**: The quality, completeness, and timeliness of data depend entirely on the original data providers. OpenAQ reflects what these sources publish.
- **Metadata Variability**: While OpenAQ harmonizes parameters, the richness of metadata (e.g., specific instrument type, exact coordinates) can vary by source.
- **Not a Primary Data Generator**: OpenAQ aggregates data; it does not generate its own measurements.
- **Potential Gaps**: Data may have gaps if the original monitoring stations were offline or did not report data.
- **Low-Cost Sensor Data**: While OpenAQ is increasingly incorporating data from validated low-cost sensor networks, users should be aware of the potential differences in data quality compared to reference-grade monitors. OpenAQ provides source information to help users assess this.

## 9. Conclusion

OpenAQ is an invaluable resource for anyone needing access to global air quality data. By breaking down barriers to data access and fostering a global community, it plays a crucial role in advancing research, policy, and public awareness to combat air pollution and its detrimental health impacts. Its API and data archives are particularly useful for large-scale analyses and integration into various applications.
