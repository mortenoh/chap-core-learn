# ECMWF HRES: High-Resolution Forecast System Overview

## 1. Introduction to ECMWF HRES

The **European Centre for Medium-Range Weather Forecasts (ECMWF)** is an independent intergovernmental organisation supported by most European nations. It is renowned for producing some of the world's leading global numerical weather predictions.

The **HRES (High-Resolution Forecast system)** is ECMWF's flagship deterministic global numerical weather prediction (NWP) system. It provides detailed forecasts of atmospheric conditions across the globe for the medium range (up to 10 days ahead). HRES is a critical tool for national meteorological services, emergency management, and various weather-sensitive industries.

Alongside HRES, ECMWF also runs an ensemble prediction system (ENS) to provide probabilistic forecasts and assess forecast uncertainty. This document focuses primarily on the deterministic HRES.

## 2. Key Characteristics of HRES

- **Producing Organization**: European Centre for Medium-Range Weather Forecasts (ECMWF).
- **Model System**: Based on ECMWF's **Integrated Forecasting System (IFS)**, a comprehensive Earth system model that includes atmospheric, land surface, and ocean wave components. The IFS is continually updated with the latest scientific advancements.
- **Spatial Resolution**:
  - Approximately **9 km** globally for the atmospheric model. This high resolution allows for a more detailed representation of weather systems and terrain effects.
  - Ocean wave model coupled to HRES also runs at a high resolution.
- **Temporal Resolution (Output Frequency)**:
  - Forecast data is typically output at hourly intervals for the first few days (e.g., up to day 3 or T+90 hours), then often at 3-hourly or 6-hourly intervals for the remainder of the forecast period.
- **Forecast Length (Lead Time)**:
  - HRES provides forecasts up to **10 days** (T+240 hours) ahead.
- **Key Variables Provided**:
  - A comprehensive suite of atmospheric variables on model levels, pressure levels, and single levels (surface or near-surface).
  - **Surface/Single-Level**: 2m temperature, 2m dewpoint temperature, 10m wind (u and v components), mean sea level pressure, surface pressure, total precipitation, snowfall, cloud cover (total, high, medium, low), CAPE, convective inhibition (CIN), soil temperature, soil moisture, snow depth, etc.
  - **Upper-Air (Pressure Levels)**: Geopotential height, temperature, wind components, humidity, vertical velocity, vorticity, divergence on multiple pressure levels (e.g., 1000 hPa to 10 hPa).
  - **Model Level Data**: Variables on the native vertical grid of the model.
  - **Ocean Wave Parameters**: Significant wave height, mean wave period, mean wave direction, swell components.
- **Update Cycle**:
  - HRES is run twice a day, initialized with data from 00 UTC and 12 UTC.
  - The IFS undergoes regular upgrades (cycle updates) which bring improvements in model physics, data assimilation, and resolution.

## 3. Data Access

Access to ECMWF's operational HRES data is primarily for its Member and Co-operating States' National Meteorological Services and for commercial customers through specific licensing agreements. It is generally not freely available in real-time to the general public or for all research purposes directly from ECMWF's operational dissemination.

However, there are ways to access HRES-related data or similar products:

- **ECMWF Data Services**:
  - **Member/Co-operating States**: Receive operational data directly through dedicated dissemination systems (e.g., ECPDS, dissemination via RMDCN).
  - **Commercial Licenses**: ECMWF licenses its forecast products to commercial entities.
  - **Research Experiments/Projects**: Specific research projects may gain access to data under particular agreements.
- **Copernicus Climate Data Store (CDS)**:
  - While the raw, real-time operational HRES feed isn't directly on CDS for public download, some **Copernicus Atmosphere Monitoring Service (CAMS)** global forecast products are based on the IFS.
  - **ERA5 Reanalysis**: Although not a forecast, ERA5 is produced using a version of the IFS and provides a valuable historical atmospheric dataset. Accessing ERA5 via CDS can familiarize users with ECMWF data formats and variables.
- **National Meteorological Services (NMSs)**:
  - NMSs often use HRES data as input for their own forecasting services and may provide derived products to their national users.
- **Data Formats**:
  - The primary format for ECMWF operational data is **GRIB** (GRIdded Binary). GRIB edition 1 (GRIB1) and GRIB edition 2 (GRIB2) are used.
  - Data can also be converted to **NetCDF** by users or through some ECMWF tools/services.

## 4. Relevance and Applications

HRES data is crucial for a wide range of applications:

- **Operational Weather Forecasting**: Forms the basis of weather forecasts issued by NMSs worldwide.
- **Severe Weather Prediction**: High resolution helps in forecasting the location, intensity, and timing of severe weather events like:
  - Heavy precipitation and flood risks.
  - Strong winds and storms.
  - Heatwaves and cold spells.
  - Snowfall events.
- **Impact Modeling**:
  - **Hydrological Models**: For flood forecasting and water resource management.
  - **Agricultural Models**: For crop yield prediction and irrigation scheduling.
  - **Air Quality Models**: As meteorological input.
  - **Health Impact Models**: Predicting conditions favorable for heat stress, cold stress, or the spread of certain vector-borne or air-borne diseases. For example, forecasting temperature and humidity for malaria early warning systems.
- **Aviation, Marine, and Energy Sectors**: Providing tailored forecasts for route planning, safety, and operations.
- **Driving Limited Area Models (LAMs)**: HRES outputs are often used as boundary conditions for higher-resolution regional NWP models.
- **Research**: Understanding atmospheric processes, model development, and forecast verification.

## 5. Working with HRES Data (Conceptual Python Examples)

Since direct HRES operational data access is restricted, these examples focus on:

1.  Using `cdsapi` to download a related dataset (ERA5) from Copernicus CDS, which shares similarities in format and variables.
2.  Opening and plotting a GRIB file (assuming a user has obtained HRES data in GRIB format) using `cfgrib` and `xarray`.

### Example 1: Downloading ERA5 Data (similar structure) via `cdsapi`

```python
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # For plotting maps

# Ensure you have cdsapi installed and your CDS API key configured in ~/.cdsapirc
# pip install cdsapi

c = cdsapi.Client()

try:
    c.retrieve(
        'reanalysis-era5-single-levels', # ERA5 single levels dataset
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature', 'total_precipitation', '10m_u_component_of_wind', '10m_v_component_of_wind',
            ],
            'year': '2023',
            'month': '07',
            'day': '15',
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': [ # North, West, South, East (e.g., Europe)
                70, -10, 30, 40,
            ],
            'format': 'netcdf', # Request NetCDF format for easier use with xarray
        },
        'era5_sample_data.nc') # Output file name

    print("ERA5 data downloaded successfully as era5_sample_data.nc")

    # Load and inspect the NetCDF file
    ds_era5 = xr.open_dataset('era5_sample_data.nc')
    print("\n--- ERA5 Dataset Sample ---")
    print(ds_era5)

    # Example: Plot 2m temperature for the first time step
    t2m = ds_era5['t2m'].isel(time=0) - 273.15 # Convert from Kelvin to Celsius

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    t2m.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
             cbar_kwargs={'label': '2m Temperature (°C)'})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title(f"ERA5 2m Temperature - {t2m.time.dt.strftime('%Y-%m-%d %H:%M UTC').item()}")
    # plt.show() # Uncomment to display plot

except Exception as e:
    print(f"An error occurred with CDS API or data processing: {e}")

```

### Example 2: Opening and Plotting a GRIB file (e.g., HRES data)

```python
import xarray as xr
import cfgrib # For reading GRIB files with xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Ensure you have cfgrib and its dependencies (like eccodes) installed
# pip install cfgrib eccodes

# Assume you have an HRES GRIB file named 'hres_sample.grib'
# This is a placeholder; you'd need an actual GRIB file.
grib_file_path = 'hres_sample.grib'

# Create a dummy GRIB file for demonstration if you don't have one
# This part is complex and usually not done manually.
# For now, we'll just show the loading part assuming the file exists.

try:
    # Load the GRIB file using xarray with the cfgrib engine
    # You might need to pass backend_kwargs for specific GRIB flavors
    # ds_hres = xr.open_dataset(grib_file_path, engine='cfgrib')

    # --- MOCKING a dataset for demonstration as creating a valid GRIB is complex ---
    import numpy as np
    import pandas as pd
    lats_mock = np.arange(60, 40, -1, dtype=np.float32) # Example latitudes
    lons_mock = np.arange(0, 20, 1, dtype=np.float32)   # Example longitudes
    time_mock = pd.to_datetime(['2024-01-01T12:00:00'])
    t2m_data_mock = 273.15 + 10 + 5 * np.random.rand(len(time_mock), len(lats_mock), len(lons_mock))
    ds_hres = xr.Dataset(
        {'t2m': (('time', 'latitude', 'longitude'), t2m_data_mock, {'units': 'K', 'long_name': '2 metre temperature'})},
        coords={'time': time_mock, 'latitude': lats_mock, 'longitude': lons_mock}
    )
    print("--- HRES Dataset (Mocked Sample) ---")
    # --- END MOCKING ---

    print(ds_hres)

    # Assuming 't2m' (2m temperature) is a variable in the dataset
    if 't2m' in ds_hres:
        hres_t2m = ds_hres['t2m'].isel(time=0) # Select first time step
        if hres_t2m.attrs.get('units', '').lower() == 'k':
             hres_t2m = hres_t2m - 273.15 # Convert K to Celsius
             hres_t2m.attrs['units'] = '°C'

        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        hres_t2m.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
                      cbar_kwargs={'label': f"HRES 2m Temperature ({hres_t2m.attrs.get('units', 'Unknown')})"})
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        plt.title(f"HRES 2m Temperature (Mocked) - {pd.to_datetime(hres_t2m.time.item()).strftime('%Y-%m-%d %H:%M UTC')}")
        # plt.show()
    else:
        print("Variable 't2m' not found in the GRIB dataset.")

# except FileNotFoundError:
#     print(f"Error: GRIB file '{grib_file_path}' not found. This example requires a sample GRIB file.")
except Exception as e:
    print(f"An error occurred processing the GRIB file: {e}")

```

## 6. Strengths of HRES

- **High Global Resolution**: One of the highest-resolution global NWP systems.
- **Skillful Forecasts**: ECMWF forecasts are consistently among the most accurate globally, especially in the medium range.
- **Comprehensive System**: Includes advanced data assimilation, sophisticated model physics, and coupling with ocean wave and land surface models.
- **Regular Updates**: The IFS undergoes continuous development and improvement.

## 7. Limitations

- **Deterministic Nature**: Being a single deterministic forecast, HRES does not inherently provide information about forecast uncertainty. This is addressed by ECMWF's Ensemble Prediction System (ENS).
- **Data Access Restrictions**: Operational real-time data is not freely available to everyone.
- **Computational Cost**: Running such a high-resolution global model is computationally very expensive.
- **Model Biases**: Like all NWP models, HRES can have systematic biases or errors, though ECMWF works continuously to reduce them.

## 8. Further Resources

- **ECMWF Website**: [https://www.ecmwf.int/](https://www.ecmwf.int/)
  - **Forecast Charts**: [https://www.ecmwf.int/en/forecasts/charts/](https://www.ecmwf.int/en/forecasts/charts/) (Visualizations of HRES and other ECMWF products)
  - **IFS Documentation**: Technical documentation on IFS model cycles.
- **Copernicus Climate Data Store (CDS)**: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) (For ERA5 and CAMS data)
- **Copernicus Atmosphere Monitoring Service (CAMS)**: [https://atmosphere.copernicus.eu/](https://atmosphere.copernicus.eu/)

ECMWF's HRES is a cornerstone of modern numerical weather prediction, providing invaluable information for a multitude of applications, including those at the intersection of climate, weather, and health.
