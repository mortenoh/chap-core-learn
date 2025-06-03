# PyCPT: Python Interface for the Climate Predictability Tool (CPT)

## 1. Introduction

### a. What is CPT?

The **Climate Predictability Tool (CPT)** is a software package developed by the International Research Institute for Climate and Society (IRI) at Columbia University. It is designed to produce tailored, optimally skillful seasonal climate forecasts and to perform forecast validation. CPT primarily uses statistical methods to link predictor variables (e.g., sea surface temperatures, atmospheric circulation patterns) with predictand variables (e.g., rainfall, temperature at specific locations).

Key statistical methods implemented in CPT include:

- Canonical Correlation Analysis (CCA)
- Principal Components Regression (PCR)
- Multiple Linear Regression (MLR)

CPT also supports Model Output Statistics (MOS) techniques to calibrate forecasts from General Circulation Models (GCMs). It is widely used by National Meteorological Services and researchers, particularly for operational seasonal forecasting.

### b. What is PyCPT?

**PyCPT** is a Python-based interface to CPT. It allows users to script and automate CPT tasks, manage CPT projects, and integrate CPT's functionalities into larger Python-based workflows. PyCPT essentially provides a way to control CPT's command-line version through Python scripts, making it easier to:

- Run batch forecasts for multiple regions or variables.
- Systematically explore different CPT settings and model configurations.
- Pre-process input data and post-process CPT outputs using Python's rich data science ecosystem.
- Integrate CPT forecasts into other applications or decision-support systems.

## 2. Why Use PyCPT?

- **Automation**: Automate repetitive CPT tasks, such as generating forecasts for many different predictands or using various predictor configurations.
- **Reproducibility**: Python scripts provide a clear, reproducible record of the forecasting process.
- **Integration**: Seamlessly integrate CPT with other Python libraries for data manipulation (`pandas`, `xarray`), analysis (`numpy`, `scipy`), and visualization (`matplotlib`, `cartopy`).
- **Customization**: Easily customize CPT parameters and experimental designs.
- **Scalability**: Facilitates running CPT for larger domains or more complex experimental setups.
- **Workflow Management**: Helps in organizing complex forecasting workflows involving data preparation, CPT execution, and results analysis.

## 3. Installation

### a. Installing CPT

PyCPT requires a working installation of CPT. CPT is primarily designed for Linux and macOS environments, though it can sometimes be run on Windows via compatibility layers like Cygwin or WSL.

- Download CPT from the IRI website: [https://iri.columbia.edu/resources/software-tools/cpt/](https://iri.columbia.edu/resources/software-tools/cpt/)
- Follow the installation instructions provided with CPT. Ensure the CPT executable (e.g., `CPT.x` or `CPT_batch.x`) is in your system's PATH or that PyCPT knows where to find it.

### b. Installing PyCPT

PyCPT is typically installed using pip:

```bash
pip install pycpt
```

Alternatively, you might install it from its source repository (e.g., from IRI's GitHub).

## 4. Core Concepts in CPT/PyCPT

Understanding these CPT concepts is crucial for using PyCPT effectively:

- **Predictand (Y)**: The variable to be predicted (e.g., seasonal rainfall total at a specific station or gridded rainfall over a region).
- **Predictor (X)**: The variable(s) used to make the prediction (e.g., sea surface temperature anomalies in a specific ocean basin).
- **Cross-Validation**: A technique used to estimate the out-of-sample skill of the forecast model. CPT employs leave-one-out cross-validation or k-fold cross-validation.
- **Hindcasts (Retroactive Forecasts)**: Forecasts made for past periods using data that would have been available at that time. Used for model training and skill assessment.
- **Forecast Models**:
  - **CCA (Canonical Correlation Analysis)**: Finds linear combinations of X and Y variables that are maximally correlated.
  - **PCR (Principal Components Regression)**: Reduces dimensionality of X variables using PCA, then performs regression.
- **Model Output Statistics (MOS)**: Techniques to correct systematic errors or improve the skill of dynamical model (GCM) forecasts. PyCPT can configure CPT for MOS applications like:
  - **Perfect Prog (Perfect Prognosis)**: Uses observed predictors (X) and predictands (Y) to build a statistical model, then applies this model to GCM-forecasted X to predict Y.
  - **Model Analogs**: Finds past GCM forecasts similar to the current GCM forecast and uses the corresponding observed outcomes.
- **Skill Scores**: Metrics to evaluate forecast quality (e.g., Pearson correlation, Spearman correlation, RMSE, ROC Area, Hit Score, Heidke Skill Score).
- **Training Period**: The historical period used to build the statistical model.
- **Forecast Period**: The period for which actual forecasts are generated.

## 5. Basic PyCPT Workflow

A typical PyCPT workflow involves:

1.  **Setup**: Define paths to CPT executable, input data, and output directories.
2.  **Project Initialization**: Create a PyCPT project instance.
3.  **Data Input**: Specify paths to predictor (X) and predictand (Y) data files. These are often text files in CPT format or NetCDF files.
4.  **Configuration**: Set CPT parameters:
    - Forecast model (CCA, PCR).
    - Number of modes (for CCA/PCR).
    - Cross-validation settings.
    - Training and forecast periods.
    - Predictor and predictand domains (spatial extent).
    - MOS settings if applicable.
5.  **Execution**: Run the CPT analysis through PyCPT.
6.  **Results**: Access and analyze CPT outputs (forecasts, skill scores, model parameters).

## 6. PyCPT Examples

**Note**: These examples are conceptual and assume you have CPT installed and appropriate data files. Paths and specific parameters will need to be adapted.

### a. Setting up a PyCPT Project

```python
from pycpt import PyCPT
import os

# --- 1. Setup ---
# Path to CPT executable (adjust if CPT.vXYZ_batch.x is used)
cpt_executable = '/path/to/your/CPT/17.6.2/bin/CPT.x'
# Base directory for your project
project_dir = '/path/to/your/pycpt_project'
os.makedirs(project_dir, exist_ok=True)

# Input data directories (assuming data is prepared)
x_dir = os.path.join(project_dir, 'input', 'X') # Predictor data
y_dir = os.path.join(project_dir, 'input', 'Y') # Predictand data
os.makedirs(x_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

# Output directory for CPT results
output_dir = os.path.join(project_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# --- 2. Initialize PyCPT ---
# Create a PyCPT instance, associating it with a CPT version
pcpt = PyCPT(version='17.6.2', cpt_exe=cpt_executable) # Adjust version as needed

# Set project workspace
pcpt.workdir = project_dir
```

### b. Defining Inputs (X and Y data)

CPT typically expects data in specific text formats or NetCDF. For this example, let's assume you have:

- `sst_data.tsv`: Predictor (X) - Sea Surface Temperature anomalies.
- `rainfall_data.tsv`: Predictand (Y) - Station rainfall totals.

These files would need to be in CPT format (often a simple tab-separated value file with station IDs/coordinates and time series).

```python
# Assume these files are placed in x_dir and y_dir respectively
# For gridded data, NetCDF files are common.

# Example: Define file paths (these are placeholders)
x_file = os.path.join(x_dir, 'ERSSTv5.1981-2020.cpt_format.nc') # Example NetCDF predictor
y_file = os.path.join(y_dir, 'CHIRPS.1981-2020.SAfrica.cpt_format.nc') # Example NetCDF predictand

# Define spatial domains (example for Southern Africa)
# These are often needed if your input files cover larger areas
# For station data, you might list station IDs or use a station file.
# For gridded data:
pcpt.x_begin_lon = -20  # Min longitude for X
pcpt.x_end_lon = 60    # Max longitude for X
pcpt.x_begin_lat = -40  # Min latitude for X
pcpt.x_end_lat = 10    # Max latitude for X

pcpt.y_begin_lon = 10   # Min longitude for Y
pcpt.y_end_lon = 40   # Max longitude for Y
pcpt.y_begin_lat = -35  # Min latitude for Y
pcpt.y_end_lat = -5   # Max latitude for Y

# Define file paths for PyCPT
pcpt.x_file = x_file
pcpt.y_file = y_file
```

### c. Running a Simple CCA Forecast

```python
# --- 3. Configuration ---
pcpt.model = 'CCA'  # Canonical Correlation Analysis
pcpt.predictand = 'precip' # Name of variable in Y file
pcpt.predictor = 'sst'     # Name of variable in X file

# Training and forecast periods
pcpt.train_begin = 1981
pcpt.train_end = 2010
pcpt.forecast_begin = 2011 # First year to forecast
pcpt.forecast_end = 2020   # Last year to forecast

# Seasonal parameters
pcpt.season_begin_month = 6  # June (e.g., for JJA season)
pcpt.season_length = 3       # 3 months (June, July, August)
pcpt.lead_time = 1           # e.g., 1 month lead (forecast made in May for JJA)

# CCA specific settings
pcpt.cca_xmodes_min = 1
pcpt.cca_xmodes_max = 5
pcpt.cca_ymodes_min = 1
pcpt.cca_ymodes_max = 5
# CPT will select the optimal number of modes within these ranges based on skill.

# Cross-validation settings
pcpt.crossvalidation_mode = 'leave-one-out' # or 'k-fold'

# Output settings
pcpt.forecast_type = 'probabilistic' # 'deterministic' or 'probabilistic'
pcpt.probabilistic_boundaries = 'terciles' # e.g., Below Normal, Normal, Above Normal

# --- 4. Execution ---
# This will generate CPT command scripts and run them.
# The `run()` method often takes a descriptive name for the run.
run_name = 'JJA_Rainfall_CCA_SST_SAfrica'
pcpt.run(run_name)

print(f"PyCPT run '{run_name}' completed. Outputs are in {os.path.join(output_dir, run_name)}")
```

### d. Accessing Results and Skill Scores

PyCPT creates an output directory for each run (e.g., `output/JJA_Rainfall_CCA_SST_SAfrica/`). This directory contains:

- CPT script files (`.cpt`)
- Output data files (forecasts, hindcasts, skill scores) often in NetCDF or text format.
- Log files.

```python
# Results are typically loaded from files generated by CPT.
# PyCPT might offer helper functions to load common outputs,
# or you can use xarray/pandas directly.

# Example: Conceptual loading of a skill score file (format depends on CPT version/settings)
skill_file_path = os.path.join(output_dir, run_name, 'CCA_skill_goodness_index.txt') # Placeholder
# if os.path.exists(skill_file_path):
#     skill_df = pd.read_csv(skill_file_path, delim_whitespace=True) # Example
#     print("\n--- Skill Scores (Example) ---")
#     print(skill_df.head())

# Example: Conceptual loading of probabilistic forecasts (often NetCDF)
forecast_file_path = os.path.join(output_dir, run_name, 'CCA_forecasts_probabilistic.nc') # Placeholder
# if os.path.exists(forecast_file_path):
#     import xarray as xr
#     forecast_ds = xr.open_dataset(forecast_file_path)
#     print("\n--- Probabilistic Forecasts (Example) ---")
#     print(forecast_ds)
```

### e. Visualizing Results (Conceptual)

```python
# Using matplotlib and cartopy for gridded forecast data
# (Assuming forecast_ds is an xarray.Dataset with lat, lon, time, category dimensions)
# if 'forecast_ds' in locals() and isinstance(forecast_ds, xr.Dataset):
#     import matplotlib.pyplot as plt
#     import cartopy.crs as ccrs

#     # Example: Plot probability of 'Above Normal' for the first forecast time
#     prob_above = forecast_ds['prob_above_normal'].isel(time=0) # Adjust variable name

#     plt.figure(figsize=(10, 8))
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     prob_above.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='YlGnBu',
#                     cbar_kwargs={'label': 'Probability of Above Normal Rainfall'})
#     ax.coastlines()
#     ax.gridlines(draw_labels=True)
#     plt.title(f"Forecast Probability (Above Normal) - {run_name}")
#     # plt.show()
```

## 7. Advanced PyCPT Features and Examples

### a. Principal Components Regression (PCR)

```python
# ... (setup and input definitions as before) ...
pcpt.model = 'PCR'
pcpt.predictor = 'sst'
pcpt.predictand = 'precip'

# PCR specific settings
pcpt.pcr_xmodes_min = 1
pcpt.pcr_xmodes_max = 8
# pcpt.pcr_ymlmodes_min = 1 # If Y is also gridded and PCA applied
# pcpt.pcr_ymlmodes_max = 5

# ... (other settings like season, lead, periods) ...
# pcpt.run('JJA_Rainfall_PCR_SST_SAfrica_PCR')
```

### b. Configuring MOS (Perfect Prog Example)

For MOS, Y is typically station data (observations), and X is GCM hindcast data for the same variable (e.g., GCM rainfall).

```python
# --- MOS Perfect Prog Setup ---
# pcpt.mos = 'perfect_prog' # or 'cca', 'pcr' for different MOS types
# pcpt.gcm_file = '/path/to/gcm_hindcasts.nc' # GCM hindcast data for X
# pcpt.obs_file = '/path/to/station_observations.tsv' # Observed station data for Y

# pcpt.x_file = pcpt.gcm_file # Predictor is GCM output
# pcpt.y_file = pcpt.obs_file # Predictand is station observation

# pcpt.model = 'CCA' # Statistical model for MOS correction

# ... (other settings: domains, season, lead, training/forecast periods) ...
# The 'forecast' period for MOS would use real-time GCM forecasts as input for X.
# pcpt.run('JJA_Rainfall_MOS_PerfectProg_GCM')
```

### c. Customizing CPT Settings

PyCPT allows setting many CPT command-line options directly as attributes.
Refer to the CPT User Manual for a full list of options.

```python
# Example: Setting specific missing value flags for X and Y
pcpt.missing_value_X = -999.0
pcpt.missing_value_Y = -99.0

# Example: Setting specific number of modes instead of a range
pcpt.cca_xmodes = 3
pcpt.cca_ymodes = 2

# Example: Requesting specific skill metrics
pcpt.skill_metrics = ['pearson', 'spearman', 'rmse', 'roc_area_below', 'roc_area_above']
```

### d. Forecast Verification Options

CPT performs verification as part of its process. PyCPT allows configuring this.

```python
# Skill scores are calculated based on the cross-validation mode.
# You can specify which scores to output.
# pcpt.goodness_index = ['corr', 'rmse', 'mad'] # Example, check CPT manual for exact names

# For probabilistic forecasts, ROC area, reliability diagrams, etc., are common.
# These are often part of standard CPT output if probabilistic forecasts are generated.
```

## 8. Data Formats for CPT

CPT primarily uses:

- **CPT Text Format**: Simple ASCII files.
  - For station data: Often columns for Station ID, Lat, Lon, Year, Month (or dekad/pentad), Value.
  - For gridded data (less common in text): Can be structured with grid point indices or lat/lon and values.
- **NetCDF**: Increasingly common, especially for gridded data. CPT expects specific conventions for coordinate names (e.g., `X`, `Y`, `T`) and variable names. PyCPT might help in preparing data or CPT might have utilities for converting common NetCDF formats.
- **IRI Data Library Format**: CPT can directly read data from the IRI Data Library if URLs are provided.

Data preparation is often a significant part of using CPT/PyCPT.

## 9. Strengths of PyCPT

- **Leverages CPT's Robustness**: Builds upon the well-tested statistical methods and forecast verification capabilities of CPT.
- **Automation and Scripting**: Makes complex or repetitive forecasting tasks manageable.
- **Python Ecosystem Integration**: Allows for powerful pre- and post-processing.
- **Reproducibility**: Enhances the transparency and reproducibility of seasonal forecasting.
- **Flexibility**: Provides fine-grained control over CPT parameters.

## 10. Limitations and Considerations

- **CPT Dependency**: Requires a working CPT installation and familiarity with CPT concepts.
- **Learning Curve**: Understanding CPT's statistical methods and data format requirements is essential.
- **Data Preparation**: Input data must be in a format CPT understands. This can be time-consuming.
- **Error Handling**: Debugging can sometimes be challenging as it involves both Python and CPT's own error messages.
- **Focus**: Primarily designed for seasonal forecasting using statistical methods. While it supports MOS for GCMs, it's not a GCM itself.

## 11. Further Resources

- **CPT Website and Documentation**: [https://iri.columbia.edu/resources/software-tools/cpt/](https://iri.columbia.edu/resources/software-tools/cpt/) (Download CPT, User Manuals)
- **PyCPT GitHub Repository (IRI)**: Search for "PyCPT IRI GitHub" for the source code, issue tracker, and potentially more examples or documentation. (e.g., [https://github.com/iri-pycpt/pycpt](https://github.com/iri-pycpt/pycpt) - check for the official link)
- **IRI Training Materials**: IRI often conducts training workshops on CPT and PyCPT, materials from which might be available online.

PyCPT is a valuable tool for those looking to operationalize or conduct extensive research using the Climate Predictability Tool, bridging the gap between CPT's powerful Fortran core and the versatile Python environment.
