# Polars Data Analysis Tutorial: High-Performance Data Manipulation

This tutorial is designed for Python experts who are new to data analysis using the Polars library. Polars is a blazingly fast DataFrame library implemented in Rust, using Apache Arrow Columnar Format as its memory model. It's gaining popularity as a high-performance alternative or complement to Pandas, especially for larger datasets.

## 1. Introduction to Polars

### a. What is Polars?

Polars is an open-source DataFrame library designed for speed and efficiency. It leverages Rust's performance and safety features and Apache Arrow for memory efficiency and interoperability. Polars provides a powerful expression API that allows for complex queries to be optimized and parallelized.

### b. Why Use Polars?

- **Performance**: Significantly faster than Pandas for many operations, especially on larger datasets, due to its Rust backend, multi-threading capabilities, and efficient memory management.
- **Lazy Evaluation**: Polars can operate in "lazy" mode, where operations are not executed immediately but are instead added to a query plan. This allows Polars to optimize the entire query before execution, often leading to substantial speedups and reduced memory usage.
- **Expressive API**: Its expression-based API is powerful and flexible, allowing for complex data transformations in a concise way.
- **Apache Arrow Integration**: Uses Arrow for its in-memory representation, which facilitates zero-copy data sharing with other Arrow-compatible systems (e.g., Spark, Dask, other databases).
- **Memory Efficiency**: Generally more memory-efficient than Pandas.
- **Growing Ecosystem**: While newer than Pandas, its community and feature set are rapidly expanding.

## 2. Core Polars Data Structures

Similar to Pandas, Polars has two primary data structures:

### a. `Series`

A Polars `Series` is a one-dimensional array of data of a single data type.

```python
import polars as pl
import numpy as np # For np.nan

# Creating a Series
s_polars = pl.Series("a", [1, 2, 3, None, 5]) # 'None' is Polars' way of representing nulls
print("--- Polars Series ---")
print(s_polars)
# Output:
# shape: (5,)
# Series: 'a' [i64]
# [
#         1
#         2
#         3
#         null
#         5
# ]

# Creating a Series with a specific data type
s_float = pl.Series("b", [1.0, 2.5, 3.0], dtype=pl.Float32)
print("\n--- Polars Series with Float32 ---")
print(s_float)
```

### b. `DataFrame`

A Polars `DataFrame` is a two-dimensional, tabular data structure composed of one or more `Series` of the same length. Columns must have names, and each column is a `Series`.

```python
# Creating a DataFrame from a dictionary of lists/Series
data_dict = {
    "integer_col": [1, 2, 3, 4],
    "string_col": ["apple", "banana", "cherry", "date"],
    "float_col": [0.5, 1.5, 2.5, 3.5]
}
df_polars = pl.DataFrame(data_dict)
print("\n--- Polars DataFrame ---")
print(df_polars)
# Output:
# shape: (4, 3)
# ┌─────────────┬────────────┬───────────┐
# │ integer_col ┆ string_col ┆ float_col │
# │ ---         ┆ ---        ┆ ---       │
# │ i64         ┆ str        ┆ f64       │
# ╞═════════════╪════════════╪═══════════╡
# │ 1           ┆ apple      ┆ 0.5       │
# │ 2           ┆ banana     ┆ 1.5       │
# │ 3           ┆ cherry     ┆ 2.5       │
# │ 4           ┆ date       ┆ 3.5       │
# └─────────────┴────────────┴───────────┘

# Creating a DataFrame with schema definition
df_with_schema = pl.DataFrame(
    {
        "id": [101, 102, 103],
        "value": [5.5, 6.6, 7.7]
    },
    schema={"id": pl.Int32, "value": pl.Float64}
)
print("\n--- DataFrame with Schema ---")
print(df_with_schema)
```

## 3. Basic Operations

### a. Loading and Saving Data

Polars supports various file formats, with optimized readers for CSV and Parquet.

```python
# --- Writing to CSV (Example) ---
# df_polars.write_csv("my_polars_dataframe.csv")
# print("\nPolars DataFrame saved to my_polars_dataframe.csv")

# --- Reading from CSV (Example) ---
# Create a dummy CSV file for reading (same as Pandas example)
dummy_csv_content = "name,age,city\nAlice,30,New York\nBob,24,Los Angeles\nCharlie,22,Chicago"
with open('dummy_polars_data.csv', 'w') as f:
    f.write(dummy_csv_content)

df_from_csv_pl = pl.read_csv('dummy_polars_data.csv')
print("\n--- Polars DataFrame from CSV ---")
print(df_from_csv_pl)

# Reading Parquet (often much faster for large data)
# df_polars.write_parquet("my_polars_dataframe.parquet")
# df_from_parquet_pl = pl.read_parquet("my_polars_dataframe.parquet")
```

### b. Inspecting Data

```python
print("\n--- Inspecting Polars DataFrame (df_from_csv_pl) ---")
print("Shape of DataFrame:")
print(df_from_csv_pl.shape) # (rows, columns)

print("\nFirst 3 rows (head):")
print(df_from_csv_pl.head(3))

print("\nLast 2 rows (tail):")
print(df_from_csv_pl.tail(2))

print("\nSchema (data types and column names):")
print(df_from_csv_pl.schema)

print("\nDescriptive statistics (describe):")
print(df_from_csv_pl.describe())
```

### c. Selection and Filtering (Expression API)

Polars primarily uses an **Expression API** for selections, filtering, and transformations. Expressions are powerful and allow for optimized execution, especially in lazy mode.

- **Selecting columns**: `df.select(pl.col("column_name"))` or `df.select(["col1", "col2"])`
- **Filtering rows**: `df.filter(pl.col("column_name") > value)`
- `pl.col(name)`: Refers to a column.
- `pl.all()`: Refers to all columns.
- `pl.lit(value)`: Creates a literal value expression.

```python
print("\n--- Selection and Filtering (df_from_csv_pl) ---")
print("Selecting 'name' column:")
print(df_from_csv_pl.select(pl.col("name")))
# Or: print(df_from_csv_pl["name"]) # Returns a Series

print("\nSelecting 'name' and 'city' columns:")
print(df_from_csv_pl.select(["name", "city"]))

print("\nFiltering rows where age > 24:")
print(df_from_csv_pl.filter(pl.col("age") > 24))

print("\nSelecting 'name' for people older than 24 and from New York:")
print(
    df_from_csv_pl.filter(
        (pl.col("age") > 24) & (pl.col("city") == "New York")
    ).select("name")
)
```

## 4. Data Cleaning and Preparation (using Expressions)

### a. Handling Missing/Null Values

Polars uses `null` for missing values.

```python
data_missing_pl = {
    "colA": [1.0, None, 3.0, 4.0, None],
    "colB": ["x", "y", None, "z", "y"]
}
df_missing_pl = pl.DataFrame(data_missing_pl)
print("\n--- Polars DataFrame with Missing Data ---")
print(df_missing_pl)

print("\nCheck for null values (is_null):")
print(df_missing_pl.select(pl.all().is_null()))

print("\nSum of null values per column:")
print(df_missing_pl.null_count())

# Filling null values
df_filled_A_pl = df_missing_pl.with_columns(
    pl.col("colA").fill_null(pl.col("colA").mean()).alias("colA_filled_mean")
)
print("\nDataFrame with colA nulls filled with mean:")
print(df_filled_A_pl)

df_filled_B_pl = df_missing_pl.with_columns(
    pl.col("colB").fill_null("unknown").alias("colB_filled_unknown")
)
print("\nDataFrame with colB nulls filled with 'unknown':")
print(df_filled_B_pl)

# Dropping rows with any null values
df_dropped_rows_pl = df_missing_pl.drop_nulls()
print("\nDataFrame with rows containing nulls dropped:")
print(df_dropped_rows_pl)
```

### b. Data Type Casting

Use the `.cast()` expression.

```python
df_types_pl = pl.DataFrame({"A": ["1", "2", "3"], "B": [10.0, 20.5, 30.1]})
print("\n--- Original Data Types (Polars) ---")
print(df_types_pl.schema)

df_casted_pl = df_types_pl.with_columns(
    pl.col("A").cast(pl.Int64),
    pl.col("B").cast(pl.Float32)
)
print("\n--- Data Types After Casting (Polars) ---")
print(df_casted_pl.schema)
```

### c. Creating New Columns / Modifying Existing Ones (`with_columns`)

`with_columns` is used to add or transform columns using expressions.

```python
df_transform_pl = pl.DataFrame({'values': [10, 20, 30, 40]})

df_transformed_pl = df_transform_pl.with_columns([
    (pl.col("values") + 5).alias("values_plus_5"),
    (pl.col("values") ** 2).alias("values_squared"),
    pl.when(pl.col("values") > 25)
      .then(pl.lit("High"))
      .otherwise(pl.lit("Low"))
      .alias("value_category")
])
print("\n--- Transformations with with_columns ---")
print(df_transformed_pl)
```

### d. String Operations

String operations are available via the `.str` namespace within expressions.

```python
df_string_pl = pl.Series("text_data", ['apple pie', 'banana bread', 'cherry cake'])
print("\n--- Polars String Operations ---")

print("Contains 'apple':")
print(df_string_pl.str.contains("apple"))

print("\nUppercase:")
print(df_string_pl.str.to_uppercase())

print("\nSplit by space (returns a list Series):")
print(df_string_pl.str.split(" "))
```

## 5. Grouping and Aggregation (`group_by` and `agg`)

Polars' `group_by` is also expression-based and very powerful.

```python
data_group_pl = {'Team': ['A', 'B', 'A', 'B', 'A', 'C'],
                 'Player': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
                 'Points': [10, 12, 8, 15, 12, 9],
                 'Assists': [5, 7, 3, 8, 6, 4]}
df_group_pl = pl.DataFrame(data_group_pl)
print("\n--- Polars DataFrame for Grouping ---")
print(df_group_pl)

# Group by 'Team' and calculate sum of 'Points'
team_points_sum_pl = df_group_pl.group_by("Team").agg(
    pl.sum("Points").alias("total_team_points")
)
print("\nSum of Points by Team (Polars):")
print(team_points_sum_pl)

# Group by 'Team' and calculate multiple aggregations
team_stats_pl = df_group_pl.group_by("Team").agg([
    pl.sum("Points").alias("total_points"),
    pl.mean("Assists").alias("avg_assists"),
    pl.count("Player").alias("num_players") # pl.count() counts non-null values in the default column
])
print("\nMultiple Aggregations by Team (Polars):")
print(team_stats_pl)
```

## 6. Joining DataFrames

Polars supports various join types (`inner`, `left`, `outer`, `semi`, `anti`).

```python
df1_pl = pl.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2_pl = pl.DataFrame({'key': ['B', 'C', 'D'], 'value2': [4, 5, 6]})

print("\n--- Joining DataFrames (Polars) ---")
# Inner join
joined_inner_pl = df1_pl.join(df2_pl, on="key", how="inner")
print("Inner Join:")
print(joined_inner_pl)

# Left join
joined_left_pl = df1_pl.join(df2_pl, on="key", how="left")
print("\nLeft Join:")
print(joined_left_pl)
```

## 7. Lazy Evaluation

One of Polars' most powerful features is its lazy execution engine. You define a sequence of operations (a query plan), and Polars optimizes it before actually computing the result.

```python
# Create a lazy DataFrame
lf_from_csv_pl = pl.scan_csv('dummy_polars_data.csv') # scan_csv starts a lazy query

# Define a query (no computation happens yet)
lazy_query = (
    lf_from_csv_pl
    .filter(pl.col("age") > 23)
    .with_columns(
        (pl.col("age") * 2).alias("age_doubled")
    )
    .select(["name", "city", "age_doubled"])
    .sort("age_doubled", descending=True)
)

print("\n--- Lazy Query Plan ---")
print(lazy_query.explain(optimized=True)) # Show the optimized query plan

# Execute the query and get the result
print("\n--- Result of Lazy Query ---")
result_lazy = lazy_query.collect()
print(result_lazy)
```

Using `scan_csv` or `scan_parquet` instead of `read_csv`/`read_parquet` initiates a lazy computation. The `.collect()` method executes the plan.

## 8. Time Series Functionality

Polars has growing support for time series operations, often leveraging its powerful expression API and window functions.

```python
# Create a time series DataFrame
rng = pl.date_range(low=np.datetime64("2023-01-01"), high=np.datetime64("2023-01-10"), interval="1d", eager=True)
df_ts_pl = pl.DataFrame({
    "time": rng,
    "value": np.random.rand(len(rng))
})
print("\n--- Polars Time Series DataFrame ---")
print(df_ts_pl)

# Rolling window calculations (e.g., 3-day rolling mean)
# Ensure the DataFrame is sorted by time if not already
df_ts_pl_sorted = df_ts_pl.sort("time")
df_ts_with_rolling = df_ts_pl_sorted.with_columns(
    pl.col("value").rolling_mean(window_size=3).alias("value_rolling_mean_3d")
)
print("\nTime Series with 3-day Rolling Mean:")
print(df_ts_with_rolling)

# Grouping by time periods (e.g., weekly sum)
# Requires 'time' column to be of Date/Datetime type
df_ts_weekly = df_ts_pl_sorted.group_by_dynamic(
    index_column="time",
    every="1w", # Group by 1 week
    # period="1w", # Duration of the window, if different from 'every'
    # offset="-3d" # Start day of the week, e.g. Monday
).agg(
    pl.sum("value").alias("weekly_sum_value")
)
print("\nWeekly Sum of Values:")
print(df_ts_weekly)
```

## 9. Preparing Data for Modeling (Conceptual Link)

Polars can be used to prepare data for machine learning, similar to Pandas. The key difference is often the syntax and the potential for performance gains with lazy evaluation.

1.  **Load & Clean Data**: Use Polars for efficient loading and cleaning.
2.  **Feature Engineering**: Use the expression API (`with_columns`) to create new features.
3.  **Preprocessing**:
    - **Encoding Categorical**: Polars can do dummy/one-hot encoding: `df.to_dummies(columns=["cat_col"])`.
    - **Scaling Numerical**: While Polars doesn't have built-in scalers like Scikit-learn, you can implement scaling using expressions or convert to NumPy/Pandas for scaling with Scikit-learn.
4.  **Converting to NumPy/PyTorch Tensors**:
    - `df_features.to_numpy()` converts a Polars DataFrame to a NumPy array.
    - `torch.from_numpy(df_features.to_numpy())` for PyTorch.

```python
# Conceptual: Preparing data for Scikit-learn/PyTorch
# Assume df_final_pl is your cleaned Polars DataFrame
# X_polars = df_final_pl.select(['feature1', 'feature2', 'encoded_cat_feature'])
# y_polars = df_final_pl.select('target_variable')

# X_numpy = X_polars.to_numpy()
# y_numpy = y_polars.to_numpy().ravel() # .ravel() if y is a single column for sklearn

# Now X_numpy and y_numpy can be used with Scikit-learn or converted to PyTorch Tensors
```

## 10. Conclusion

Polars offers a compelling alternative to Pandas for data manipulation in Python, especially when performance and memory efficiency are critical. Its expressive API and lazy evaluation capabilities allow for optimized and parallelized execution of complex queries. While the API is different from Pandas, Python experts will find its syntax intuitive and powerful once familiarized. As the library matures, its integration with the broader data science ecosystem continues to grow, making it an increasingly valuable tool.
