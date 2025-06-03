# Overview of Example Data (`example_data/`)

This directory contains a variety of sample datasets and example files used for testing, demonstration, and development purposes within the CHAP (Climate and Health Analysis Platform) core project.

## File Categories and Descriptions

### 1. Request/Response Examples (JSON)

- `anonymous_chap_request.json`: Example of an anonymized CHAP request.
- `anonymous_make_dataset_request.json`: Example of an anonymized request to create a dataset.
- `anonymous_make_prediction_request.json`: Example of an anonymized request to make a prediction.
- `dhis_response.json`: Sample response from a DHIS2-like system.
- `population_Laos.json`: Population data for Laos in JSON format.
- `precipitation_seasonal_forecast.json`: Seasonal forecast data for precipitation.
- `temperature_seasonal_forecast.json`: Seasonal forecast data for temperature.

### 2. Climate and Health Data (CSV/TSV)

- `climate_data_daily.csv`: Daily climate data.
- `climate_data.csv`: General climate data.
- `data_prepared_for_lagged_regression.csv`: Data preprocessed for lagged regression analysis.
- `data.csv`: A generic data file.
- `dengue_input_from_source_v10.csv`: Dengue-related input data.
- `health_population_data.csv`: Combined health and population data.
- `hydro_met_subset.csv`: Subset of hydro-meteorological data.
- `hydromet_5_filtered.csv`: Filtered hydro-meteorological data.
- `laos_pulled_data.csv`: Data pulled for Laos.
- `masked_data_til_2005.csv`: Data masked up to the year 2005.
- `Minimalist_example_data.csv`: A minimal dataset for example purposes.
- `Minimalist_multiregion_example_data.csv`: A minimal multi-region dataset.
- `monthly_data.csv`: Monthly aggregated data.
- `nicaragua_weekly_data.csv`: Weekly data for Nicaragua.
- `obfuscated_laos_data.tsv`: Obfuscated data for Laos (tab-separated).
- `small_laos_data_with_polygons.csv`: Small dataset for Laos, including polygon information.
- `training_data_til_2005.csv`: Training data up to the year 2005.
- `vietnam_monthly.csv`: Monthly data for Vietnam.

### 3. Geospatial Data (GeoJSON)

- `example_polygons.geojson`: Example polygon data.
- `Organisation units.geojson`: GeoJSON file representing organizational units.
- `small_laos_data_with_polygons.geojson`: GeoJSON representation of the small Laos dataset with polygons.
- `vietnam_monthly.geojson`: GeoJSON representation of monthly Vietnam data.

### 4. Model and Script Files

- `DESCRIPTION`: Likely a description file, possibly for an R package or similar.
- `dhis_test_model`: A test model, format unspecified (could be a directory or a binary file).
- `example_r_script.r`: An example R script.
- `map.graph`: A graph file, possibly for network analysis or mapping.

### 5. Archived Data

- `full_data.tar.gz`: A tarball containing more extensive data.
- `sample_chap_app_output.zip`: Zipped output from a sample CHAP application run.
- `sample_dhis_data.zip`: Zipped sample DHIS2 data.

### 6. Subdirectories (Containing further structured examples)

These directories likely contain more specific or complex examples, possibly organized by model type, API version, or data processing stage.

- `debug_model/`: Data or configurations for debugging a model.
- `nonstandard_separate/`: Examples of non-standard data formats (separate files).
- `nonstandard_stacked/`: Examples of non-standard data formats (stacked).
- `seasonal_forecasts/`: More detailed seasonal forecast data.
- `train_test_splitted/`: Data split into training and testing sets.
- `v0/`: Examples related to version 0 of an API or data format.
- `v1_api/`: Examples related to version 1 of an API.

This collection of example data serves as a valuable resource for understanding the types of data the CHAP system can process and for developing and testing new features.
