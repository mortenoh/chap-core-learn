# Reanalysis: Creating Consistent Climate Records

## 1. What is Reanalysis?

**Reanalysis** is a scientific method that systematically combines historical weather observations with modern numerical weather prediction (NWP) models using data assimilation techniques. The goal is to produce a comprehensive, consistent, and spatially complete gridded dataset representing the state of the Earth's atmosphere, land surface, and oceans over an extended period, often spanning several decades.

Essentially, reanalysis "re-analyzes" past weather conditions using a fixed, state-of-the-art weather forecasting model and data assimilation system, incorporating all available historical observations. This approach overcomes inconsistencies that would arise from using evolving operational forecast models over time.

The output is a dynamically consistent, gridded "best estimate" of various climate variables (e.g., temperature, precipitation, wind, humidity, pressure) at regular time intervals (e.g., hourly, 3-hourly, 6-hourly) and spatial resolutions.

## 2. Purpose and Importance of Reanalysis

Reanalysis datasets are crucial for several reasons:

- **Climate Monitoring and Change Detection**: Provide long, consistent time series of climate variables, essential for monitoring climate variability, detecting long-term climate change trends, and understanding past climate events.
- **Understanding Atmospheric Processes**: Offer a 4D (three spatial dimensions + time) view of the atmosphere, allowing researchers to study atmospheric dynamics, weather systems, and climate phenomena.
- **Model Evaluation**: Used as a benchmark for evaluating the performance of climate models.
- **Driving Other Models**: Serve as input or boundary conditions for regional climate models, hydrological models, air quality models, and impact assessment models (e.g., for agriculture, health).
- **Climate Services**: Provide baseline climate information for various sectors.
- **Filling Observational Gaps**: Offer estimates of atmospheric conditions in data-sparse regions or for periods where direct observations are limited.
- **Consistency Over Time**: By using a fixed model and assimilation system, reanalysis avoids artificial jumps or trends that could be introduced by changes in operational forecasting systems over the years.

## 3. How Reanalysis is Produced (General Methodology)

The production of a reanalysis dataset involves several key components:

1.  **Numerical Weather Prediction (NWP) Model**:
    - A sophisticated, physically-based model of the Earth's atmosphere (and often coupled land surface and ocean wave models).
    - The _same version_ of this model is used throughout the entire reanalysis period to ensure consistency.
2.  **Data Assimilation System**:
    - A system that optimally combines short-range model forecasts (the "first guess" or "background field") with available observations to produce an improved estimate of the atmospheric state (the "analysis").
    - Common techniques include 3D-Var, 4D-Var (used in advanced systems like ECMWF's), and Ensemble Kalman Filter (EnKF).
3.  **Historical Observations**:
    - A vast collection of diverse historical observations is gathered and quality-controlled. These include:
      - **Surface observations**: From weather stations (SYNOP, METAR), buoys, ships.
      - **Upper-air observations**: Radiosondes (weather balloons), aircraft reports (AMDAR, AIREP), pilot balloons.
      - **Satellite observations**: Radiances from various satellite instruments (sounders, imagers), scatterometer winds, GPS radio occultation data. The availability and type of satellite data change significantly over the reanalysis period, with far more data available post-1979 (the "satellite era").
4.  **Process**:
    a. The NWP model is run to produce a short-range forecast (e.g., 6-12 hours).
    b. This forecast serves as the first guess for the state of the atmosphere at the next analysis time.
    c. The data assimilation system then ingests all available observations valid around that analysis time.
    d. It combines the first guess with the observations, taking into account their respective uncertainties, to produce an updated, more accurate analysis of the atmospheric state.
    e. This analysis becomes the initial condition for the next short-range forecast, and the cycle repeats.

This process is computationally intensive and requires significant supercomputing resources and data management capabilities.

## 4. Key Characteristics of Reanalysis Datasets

- **Spatial and Temporal Completeness**: Provide gridded data covering the entire globe (or a large region) at regular time intervals, even in areas with no direct observations.
- **Consistency**: Use a frozen (unchanging) model and data assimilation system, ensuring that changes seen in the dataset are more likely to reflect real atmospheric changes rather than artifacts from evolving analysis systems.
- **Multi-variable**: Offer a wide range of atmospheric, land, and sometimes ocean variables that are physically consistent with each other.
- **Long Time Series**: Typically span several decades, allowing for the study of long-term climate variability and change.
- **Gridded Format**: Data is provided on a regular latitude-longitude grid or other standard model grids (e.g., Gaussian grids), often in GRIB or NetCDF format.
- **Uncertainty Estimates**: Some modern reanalyses (like ERA5) provide ensemble members to help quantify uncertainty in the estimates.

## 5. ERA5: A Prominent Example of Reanalysis

**ERA5** is the fifth generation global atmospheric reanalysis produced by the European Centre for Medium-Range Weather Forecasts (ECMWF) as part of the Copernicus Climate Change Service (C3S).

- **Model System**: Uses a 2016 version of ECMWF's Integrated Forecasting System (IFS Cycle 41r2).
- **Spatial Resolution**: Approximately 31 km globally (0.28125 degrees).
- **Temporal Resolution**: Hourly output for many variables.
- **Temporal Coverage**: From 1940 to near real-time (with a delay of a few days).
- **Data Assimilation**: Advanced 4D-Var system.
- **Observations Used**: Assimilates a vast amount of historical in-situ and satellite observations. The quality and quantity of observations improve over time, especially after 1979.
- **Variables**: Provides a comprehensive set of atmospheric variables on pressure levels, model levels, and single levels (surface), as well as ocean wave parameters. (Refer to `ERA5.md` or `ECMWF_HRES_OVERVIEW.md` for more variable details).
- **Uncertainty**: Provides a 10-member ensemble at a coarser resolution (EDA - Ensemble of Data Assimilations) to estimate uncertainty.
- **ERA5-Land**: A companion dataset providing enhanced land surface variables at a higher resolution (~9 km) by re-running the land component of the IFS forced by ERA5 atmospheric fields.
- **Access**: Freely available through the Copernicus Climate Data Store (CDS).

ERA5 is widely considered one of the most advanced and comprehensive reanalysis datasets currently available.

## 6. Other Major Reanalysis Datasets

Besides ERA5, several other important reanalysis datasets have been produced by various meteorological centers:

- **NCEP/NCAR Reanalysis 1 (R1)**: One of the earliest widely used global reanalyses, produced by the U.S. National Centers for Environmental Prediction (NCEP) and the National Center for Atmospheric Research (NCAR). Covers 1948 to present. Coarser resolution and older model system compared to modern reanalyses.
- **NCEP-DOE Reanalysis 2 (R2)**: An updated version of R1 with some bug fixes and improved physics.
- **CFSR (Climate Forecast System Reanalysis)**: Produced by NCEP. Higher resolution than R1/R2.
- **MERRA-2 (Modern-Era Retrospective analysis for Research and Applications, Version 2)**: Produced by NASA's Global Modeling and Assimilation Office (GMAO). Focuses on aerosol assimilation. Covers 1980 to present.
- **JRA-55 (Japanese 55-year Reanalysis)**: Produced by the Japan Meteorological Agency (JMA). Covers 1958 to present.
- **Regional Reanalyses**: Some projects focus on producing high-resolution reanalyses over specific regions (e.g., HARMONIE-Climate for Europe, NARR for North America).

Each reanalysis dataset has its own strengths, weaknesses, specific model versions, assimilated observations, and resolutions. The choice of reanalysis often depends on the specific application, region of interest, and variables required.

## 7. Applications of Reanalysis Data

- **Climate Science**: Studying climate variability (e.g., ENSO, NAO), trends, extreme events, atmospheric circulation patterns, and energy/water cycles.
- **Renewable Energy**: Assessing wind and solar resources.
- **Agriculture**: Understanding climate impacts on crop yields, planning for irrigation.
- **Water Resource Management**: Hydrological modeling, drought and flood assessment.
- **Air Quality**: Providing meteorological inputs for air pollution models.
- **Health Impact Studies**:
  - Assessing exposure to weather variables (temperature, humidity, precipitation) linked to health outcomes (e.g., heat stress, vector-borne diseases, respiratory illnesses).
  - Providing historical climate context for epidemiological studies.
  - Driving biometeorological models.
- **Insurance and Risk Management**: Assessing risks associated with extreme weather events.
- **Education and Outreach**: Visualizing past weather and climate.

## 8. Strengths of Reanalysis

- **Spatial and Temporal Completeness**: Provides a global, gap-free record.
- **Consistency**: Uses a fixed modeling system, reducing artificial trends.
- **Multi-variable and Physical Consistency**: Variables are dynamically consistent within the model physics.
- **Accessibility**: Many reanalysis datasets are freely available.
- **Long Records**: Enable the study of decadal and longer-term variability.

## 9. Limitations and Considerations

- **Model Dependence**: Reanalysis products are model outputs, not direct observations. They inherit biases and systematic errors from the underlying NWP model, especially in data-sparse regions or for variables not well-constrained by observations.
- **Observation Changes Over Time**: The observing system changes significantly over the reanalysis period (e.g., introduction of new satellite instruments, decline in radiosonde networks in some areas). While data assimilation aims to handle this, it can still introduce inhomogeneities or affect trend analysis, particularly before the modern satellite era (pre-1979).
- **Resolution**: While modern reanalyses have improved resolution, they may still smooth out very localized or very intense extreme weather events.
- **Uncertainty**: All reanalysis fields have associated uncertainties. These can be larger in data-sparse regions or for poorly observed variables. Ensemble-based reanalyses (like ERA5's EDA) help quantify this.
- **"Spurious" Trends**: Despite efforts, subtle changes in observation types or assimilation techniques can sometimes lead to artificial trends in some variables or regions.
- **Land Surface Processes**: Representation of land surface variables (e.g., soil moisture, snowpack) can be challenging and may vary significantly between different reanalysis products. Specialized land reanalyses (like ERA5-Land) often provide improvements.
- **Data Volume**: High-resolution global reanalyses are very large datasets, requiring significant storage and processing capabilities.

## 10. Conceptual Python Example: Accessing ERA5 Data via CDS API

This example is similar to one shown in `ECMWF_HRES_OVERVIEW.md` as ERA5 is a key ECMWF product accessed via CDS.

```python
import cdsapi
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Ensure cdsapi is installed and ~/.cdsapirc is configured
# pip install cdsapi

c = cdsapi.Client()

try:
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': '2m_temperature',
            'year': '2022',
            'month': '07',
            'day': ['01', '02', '03'], # First 3 days of July 2022
            'time': '12:00',          # Noon UTC
            'area': [60, -10, 35, 30], # North, West, South, East (e.g., Europe)
            'format': 'netcdf',
        },
        'era5_july_2022_sample.nc')

    print("ERA5 data for July 2022 (sample) downloaded.")

    # Load and inspect the data
    ds = xr.open_dataset('era5_july_2022_sample.nc')
    print("\n--- ERA5 Sample Dataset ---")
    print(ds)

    # Plot 2m temperature for the first day (2022-07-01 12:00 UTC)
    t2m_day1 = ds['t2m'].sel(time='2022-07-01T12:00:00') - 273.15 # to Celsius

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    t2m_day1.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm',
                  cbar_kwargs={'label': '2m Temperature (°C)'})
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    plt.title("ERA5 2m Temperature - 2022-07-01 12:00 UTC")
    # plt.show() # Uncomment to display

except Exception as e:
    print(f"An error occurred: {e}")
```

## 11. Conclusion

Reanalysis datasets are indispensable tools in Earth system science, providing a consistent and comprehensive historical record of atmospheric, land, and ocean conditions. They bridge the gap between sparse observations and complete global fields, enabling a wide range of research and applications, including crucial work in understanding climate change and its impacts on health and other sectors. While they have limitations, ongoing improvements in models, data assimilation techniques, and the incorporation of more observations continue to enhance their quality and utility.
