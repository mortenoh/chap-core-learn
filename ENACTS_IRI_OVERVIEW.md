# ENACTS and IRI: Climate Services for Development

## 1. Introduction

### International Research Institute for Climate and Society (IRI)

The **International Research Institute for Climate and Society (IRI)**, part of Columbia University's Climate School, aims to enhance society's capability to understand, anticipate, and manage the impacts of climate in order to improve human welfare and the environment, especially in developing countries. IRI conducts fundamental and applied research, provides education and training, and develops practical tools and information products.

A cornerstone of IRI's work is its **Data Library**, an extensive online platform that provides access to a vast collection of climate and environmental datasets, along with powerful analysis and visualization tools.

### ENACTS (Enhancing National Climate Services)

**ENACTS** is an initiative led by IRI in collaboration with National Meteorological and Hydrological Services (NMHSs) primarily in Africa, and also in other regions like South Asia and Latin America. The core goal of ENACTS is to improve the availability, access, and use of climate information at national and sub-national levels to support decision-making in climate-sensitive sectors like agriculture, health, and disaster risk management.

A key output of ENACTS is the creation of high-resolution, quality-controlled historical climate datasets by combining national station observations with satellite and other proxy data. These datasets are then made available through online "Maprooms" built upon the IRI Data Library platform.

## 2. Key Characteristics and Goals

### IRI

- **Mission**: To bridge climate science with societal needs, particularly for sustainable development and risk management.
- **Focus Areas**: Climate prediction, climate monitoring, agriculture and food security, water resources, public health, disaster risk reduction, climate adaptation.
- **Approach**: Interdisciplinary research, capacity building, development of climate services, policy engagement.
- **IRI Data Library**: A unique resource that allows users to access, analyze, visualize, and download a wide array of climate and related datasets. It uses a powerful data manipulation language (Ingrid) accessible through a web interface.

### ENACTS

- **Primary Goal**: To strengthen the capacity of NMHSs to produce and deliver reliable and timely climate information tailored to user needs.
- **Methodology**:
  1.  **Data Rescue and Quality Control**: Collecting and digitizing historical station data.
  2.  **Data Blending**: Combining station data with satellite rainfall estimates (e.g., CHIRPS, TRMM, ARC2) and reanalysis temperature data (e.g., ERA5) to create spatially and temporally complete gridded datasets (typically for rainfall and temperature). This process addresses gaps in station records and provides data for locations without stations.
  3.  **Product Generation**: Developing derived climate information products (e.g., onset of rains, length of growing season, drought indices).
  4.  **Dissemination**: Making data and products available through online Maprooms (customized versions of the IRI Data Library) and other formats.
  5.  **Training and Capacity Building**: Working closely with NMHS staff and users from various sectors.
- **Outcome**: Improved climate data availability at national levels, enabling better climate risk management and adaptation.

## 3. Data Products and Tools (via IRI Data Library & ENACTS Maprooms)

The IRI Data Library is the engine behind ENACTS Maprooms and hosts a vast array of datasets beyond ENACTS products.

### a. ENACTS Datasets (Country-Specific)

- **Gridded Historical Rainfall**: Typically daily, at high resolution (e.g., ~4-5 km), often spanning 30+ years. Created by blending station data with satellite proxies.
- **Gridded Historical Temperature (Min & Max)**: Similar to rainfall, blending station data with reanalysis products.
- **Derived Climate Variables**:
  - Onset and cessation of rainy seasons.
  - Length of growing period.
  - Frequency of dry/wet spells.
  - Standardized Precipitation Index (SPI) for drought monitoring.
  - Decadal/pentadal summaries.
- **Availability**: Through national ENACTS Maprooms hosted on the IRI Data Library platform (e.g., for Ethiopia, Rwanda, Ghana, Zambia, Bangladesh, etc.).

### b. Other Datasets in the IRI Data Library

The IRI Data Library provides access to a much broader collection, including:

- **Global Gridded Observational Datasets**: Precipitation (e.g., CHIRPS, GPCP, CMORPH), Temperature (e.g., CRU, GHCN).
- **Reanalysis Products**: ERA5, NCEP/NCAR Reanalysis, MERRA-2.
- **Satellite Data**: NDVI, TRMM, GPM.
- **Climate Model Outputs**:
  - **Seasonal Forecasts**: From various global producing centers (e.g., NMME, C3S).
  - **Subseasonal Forecasts**.
  - **CMIP Projections**: Output from GCMs used in IPCC reports.
- **Oceanographic Data**: Sea Surface Temperatures (SSTs), El Niño/Southern Oscillation (ENSO) indices.
- **Health and Socioeconomic Data**: Some datasets relevant to vulnerability and exposure.

### c. IRI Data Library Tools (Maprooms)

- **Visualization**: Create maps, time series plots, Hovmöller diagrams, cross-sections.
- **Analysis**:
  - Calculate averages, anomalies, climatologies, trends.
  - Perform spatial and temporal subsetting.
  - Apply filters, correlations, regressions.
  - Calculate derived variables on-the-fly using Ingrid scripting.
- **Download**: Data can be downloaded in various formats (NetCDF, CSV, GeoTIFF, etc.).
- **Function Expert Mode**: Allows users to write custom Ingrid scripts for complex analyses.

## 4. Data Access

- **IRI Data Library**: Accessible directly via its website: [http://iridl.ldeo.columbia.edu/](http://iridl.ldeo.columbia.edu/)
- **ENACTS Maprooms**: Specific URLs for each participating country, usually linked from the NMHS website or IRI's ENACTS pages. For example, a country's ENACTS maproom might be `http://[country_name].maproom.iri.columbia.edu/`.
- **Programmatic Access**:
  - The IRI Data Library allows data access via URLs constructed according to its data serving protocols. This means data can be fetched using tools like `wget`, `curl`, or Python libraries.
  - Some datasets might have specific APIs or OPeNDAP access.

## 5. Relevance to Climate-Health Applications

ENACTS and the IRI Data Library are highly relevant for climate and health studies:

- **High-Quality Local Climate Data**: ENACTS provides crucial baseline climate data (rainfall, temperature) at national and sub-national scales, often in regions where such data was previously sparse or inaccessible. This is essential for understanding local climate drivers of health outcomes.
- **Exposure Assessment**: Gridded ENACTS data can be used to estimate climate exposures for specific populations or at health facility locations.
- **Early Warning Systems**:
  - Historical data can be used to define thresholds for climate-sensitive diseases (e.g., temperature ranges for vector activity, rainfall anomalies for waterborne diseases).
  - Seasonal and subseasonal forecasts from the IRI Data Library can be integrated into health early warning systems (e.g., for malaria, dengue, meningitis).
- **Vulnerability Mapping**: Combining climate data with socioeconomic and health data to identify areas and populations most vulnerable to climate-sensitive diseases.
- **Climate Change Impact Studies**: While ENACTS focuses on historical data, the broader IRI Data Library provides access to CMIP projections, which can be used to assess future health risks under different climate scenarios.
- **Research and Capacity Building**: IRI and ENACTS play a significant role in training researchers and public health practitioners in the use of climate information for health decision-making.

## 6. Conceptual Python Examples for Accessing IRI Data Library

The IRI Data Library serves data through URLs. You can construct these URLs to get data in various formats, including plain text/CSV which can be easily read by Python. The structure of these URLs can be complex as they embed Ingrid queries.

A common way to explore is to use the "Expert Mode" in a Maproom to construct a query, then find the link to the data in a simple format.

**Example: Fetching a time series of ENACTS rainfall for a point (Conceptual)**

Let's assume we found a URL from an ENACTS Maproom (e.g., for a specific point in Ethiopia) that provides monthly rainfall data in a text format.

```python
import pandas as pd
import requests
from io import StringIO

# This is a HYPOTHETICAL URL structure.
# You would typically get this by:
# 1. Navigating the IRI Data Library/Maproom to your desired dataset and variable.
# 2. Selecting a point or region.
# 3. Viewing the data as a table or time series.
# 4. Looking for a link to download the data in a simple text/CSV format.
#    The URL often ends with something like '/table.tsv' or similar.

# Example: Hypothetical URL for monthly rainfall at a point (lon=38.75, lat=9.00)
# from an Ethiopian ENACTS dataset.
# NOTE: This URL is purely illustrative and will NOT work.
# You need to find a real one from the IRI Data Library.
hypothetical_iri_data_url = "http://iridl.ldeo.columbia.edu/SOURCES/.SOME_COUNTRY/.ENACTS/.RAINFALL/.MONTHLY/X/38.75/VALUE/Y/9.00/VALUE/T/%28Jan%201981%29%28Dec%202020%29RANGEEDGES/table.tsv"

try:
    # response = requests.get(hypothetical_iri_data_url)
    # response.raise_for_status() # Raise an exception for HTTP errors

    # --- MOCKING A RESPONSE for demonstration as the URL is fake ---
    # This simulates what a TSV (Tab-Separated Values) response might look like
    mock_tsv_data = (
        "T\trainfall_mm\n"
        "1981-01-01\t5.2\n"
        "1981-02-01\t10.1\n"
        "1981-03-01\t85.3\n"
        # ... more data
        "2020-12-01\t2.5\n"
    )
    # --- END MOCKING ---

    # For real usage, uncomment the requests.get part and comment out mock_tsv_data
    # data_io = StringIO(response.text)
    data_io = StringIO(mock_tsv_data) # Using mocked data for this example

    # Read the tab-separated data into a pandas DataFrame
    # The actual parsing might need adjustment based on the specific format from IRI.
    # Sometimes there are header lines to skip, or comments.
    df_rainfall = pd.read_csv(data_io, sep='\t') # Assuming tab-separated

    # Convert 'T' column to datetime if it's not already
    df_rainfall['T'] = pd.to_datetime(df_rainfall['T'])
    df_rainfall.rename(columns={'T': 'date', 'rainfall_mm': 'rainfall_mm'}, inplace=True)
    df_rainfall.set_index('date', inplace=True)

    print("--- Rainfall Data from IRI (Mocked Example) ---")
    print(df_rainfall.head())

    # Now you can analyze or plot this data
    # import matplotlib.pyplot as plt
    # df_rainfall['rainfall_mm'].plot(kind='line', figsize=(10,5))
    # plt.title("Monthly Rainfall at Point (38.75E, 9.00N) - Mocked")
    # plt.ylabel("Rainfall (mm)")
    # plt.xlabel("Date")
    # plt.show()

except requests.exceptions.RequestException as e:
    print(f"Error fetching data from IRI: {e}")
except pd.errors.ParserError as e:
    print(f"Error parsing data: {e}. Check the data format from IRI.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

```

**Note on Python Access**:

- The `xarray` library can also open NetCDF files directly via OPeNDAP URLs if the IRI Data Library serves them in that format (e.g., `xr.open_dataset(dap_url)`).
- Constructing the correct URLs for direct data download often requires navigating the Data Library interface first to identify the dataset path and desired parameters (spatial subset, temporal range, variable).

## 7. Further Resources

- **IRI Website**: [https://iri.columbia.edu/](https://iri.columbia.edu/)
- **IRI Data Library**: [http://iridl.ldeo.columbia.edu/](http://iridl.ldeo.columbia.edu/)
  - **Tutorials**: The Data Library often has tutorials on how to use its interface and tools.
- **ENACTS Information (via IRI)**: [https://iri.columbia.edu/resources/enacts/](https://iri.columbia.edu/resources/enacts/)
- **Specific ENACTS Country Maprooms**: Search for "[Country Name] ENACTS Maproom" or check NMHS websites of participating countries.

ENACTS and the IRI Data Library represent powerful resources for accessing and utilizing climate information for societal benefit, with strong applications in the health sector for understanding and managing climate-sensitive health risks.
