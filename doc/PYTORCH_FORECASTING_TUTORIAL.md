# PyTorch Time Series Forecasting Tutorial: From Zero to a Simple Model

This tutorial is designed for Python experts who are new to machine learning and modeling, specifically focusing on time series forecasting using PyTorch. We'll build a simple model to predict future values in a sequence based on past values.

## 1. Introduction to Forecasting and Modeling Concepts

### a. What is Time Series Forecasting?

A **time series** is a sequence of data points collected over time (e.g., daily stock prices, hourly temperature readings, monthly sales figures). **Time series forecasting** is the process of using a model to predict future values based on previously observed values.

### b. The Core Idea: Learning from the Past

The fundamental assumption is that past patterns in the data can provide insights into future behavior. Our goal is to "teach" a model to recognize these patterns.

### c. What is a "Model" in this Context?

In machine learning, a model is essentially a mathematical function that takes some input data and produces an output (a prediction). The process of "training" a model involves adjusting its internal parameters (often called weights and biases) so that its outputs for known past data are as close as possible to the actual historical outcomes.

### d. Supervised Learning for Forecasting

We can frame time series forecasting as a **supervised learning** problem. We create input-output pairs from our historical time series:

- **Input (X)**: A sequence of past observations (e.g., values from time `t-k` to `t-1`). This is often called a "window" or "look-back period."
- **Output (y)**: The value we want to predict (e.g., value at time `t`).

The model learns a mapping `f` such that `y ≈ f(X)`.

## 2. Data Preparation for Time Series Forecasting

This is a crucial step. Neural networks require data in a specific numerical format.

### a. Generating Sample Data

For simplicity, let's create a synthetic time series: a sine wave with some noise.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple sine wave time series
time = np.arange(0, 200, 0.1)
amplitude = np.sin(time) + np.random.normal(0, 0.15, len(time)) # Sine wave + noise

plt.figure(figsize=(12, 4))
plt.plot(time, amplitude)
plt.title("Synthetic Time Series (Sine Wave with Noise)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
# plt.show() # Uncomment if running locally to display
print("Sample data generated.")
```

### b. Creating Sequences (Sliding Window)

We need to transform our 1D time series into input sequences (X) and corresponding target values (y).

```python
import torch

def create_sequences(data, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 50  # Use last 50 points to predict the next point
X, y = create_sequences(amplitude, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).float().unsqueeze(1) # Add feature dimension for y

print(f"Shape of X_tensor: {X_tensor.shape}") # (num_samples, sequence_length)
print(f"Shape of y_tensor: {y_tensor.shape}") # (num_samples, 1)
```

Here, each sample in `X_tensor` is a sequence of 50 past values, and the corresponding entry in `y_tensor` is the single value that immediately followed that sequence.

### c. Splitting Data (Train and Test)

It's vital to evaluate the model on data it hasn't seen during training. For time series, the split must be chronological.

```python
# Split data chronologically
train_size = int(len(X_tensor) * 0.8)
test_size = len(X_tensor) - train_size

X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

print(f"Training set size: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set size: X_test {X_test.shape}, y_test {y_test.shape}")
```

### d. Scaling/Normalization

Neural networks often perform better when input features are scaled to a similar range (e.g., 0 to 1 or mean 0, std 1). We'll use Min-Max scaling. **Important**: Fit the scaler _only_ on the training data, then use it to transform both training and test data to prevent data leakage from the test set.

```python
from sklearn.preprocessing import MinMaxScaler

# Scale features (X)
# Reshape X_train and X_test to 2D for scaler: (num_samples * sequence_length, 1)
# then reshape back. Or scale each sequence if features within sequence have different scales.
# For simplicity here, we treat the whole sequence as features to be scaled together,
# which might not be ideal for all problems but works for this univariate example.
# A more common approach for sequences is to scale the original 1D series first.

# Let's scale the original 1D amplitude series first for better practice
amplitude_scaled = amplitude.copy().reshape(-1, 1) # Reshape for scaler
scaler = MinMaxScaler(feature_range=(-1, 1)) # Scale to [-1, 1]
amplitude_scaled = scaler.fit_transform(amplitude_scaled).flatten() # Flatten back to 1D

# Re-create sequences from scaled data
X_scaled, y_scaled = create_sequences(amplitude_scaled, sequence_length)
X_scaled_tensor = torch.from_numpy(X_scaled).float()
y_scaled_tensor = torch.from_numpy(y_scaled).float().unsqueeze(1)

# Split scaled data
X_train_scaled, X_test_scaled = X_scaled_tensor[:train_size], X_scaled_tensor[train_size:]
y_train_scaled, y_test_scaled = y_scaled_tensor[:train_size], y_scaled_tensor[train_size:]

print(f"X_train_scaled shape: {X_train_scaled.shape}")
```

## 3. Building a Simple Neural Network Model with PyTorch

We'll start with a simple Multi-Layer Perceptron (MLP) that takes the sequence of `sequence_length` past values and predicts the next value.

### a. Defining the Model Class

In PyTorch, models are typically defined as classes inheriting from `torch.nn.Module`.

```python
import torch.nn as nn

class SimpleMLPForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLPForecaster, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size) # Input layer to hidden layer
        self.relu = nn.ReLU()                         # Activation function
        self.fc2 = nn.Linear(hidden_size, output_size)# Hidden layer to output layer

    def forward(self, x):
        # Define the forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
input_size = sequence_length # We feed the whole sequence as features
hidden_size = 64             # Number of neurons in the hidden layer
output_size = 1              # We predict one value

model = SimpleMLPForecaster(input_size, hidden_size, output_size)
print("\nModel Architecture:")
print(model)
```

- `nn.Linear(in_features, out_features)`: A fully connected layer.
- `nn.ReLU()`: Rectified Linear Unit, a common activation function that introduces non-linearity.

## 4. Training the Model

Training involves feeding the training data to the model, calculating how "wrong" its predictions are (loss), and adjusting its parameters to reduce this error.

### a. Defining Loss Function and Optimizer

- **Loss Function**: Measures the difference between the model's predictions and the actual target values. For regression, Mean Squared Error (MSE) is common.
- **Optimizer**: Algorithm that adjusts the model's parameters (weights) to minimize the loss. Adam is a popular choice.

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # lr is learning rate
```

### b. The Training Loop

```python
num_epochs = 200  # Number of times to iterate over the entire training dataset
batch_size = 16   # Process data in mini-batches

# For batching, we can use PyTorch's DataLoader
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# shuffle=True is generally good, but for time series, if order within batch matters, consider False.
# For this MLP on fixed windows, shuffle=True is fine.

print("\nStarting Training...")
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        # 1. Forward pass: Compute predicted y by passing x to the model
        outputs = model(batch_X)

        # 2. Compute loss
        loss = criterion(outputs, batch_y)
        epoch_loss += loss.item() * batch_X.size(0) # Accumulate loss

        # 3. Backward pass and optimize
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights

    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    if (epoch + 1) % 20 == 0: # Print progress every 20 epochs
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}')

print("Training finished.")
```

- `model.train()`: Sets the model to training mode (important if using layers like Dropout or BatchNorm, though not critical for this simple MLP).
- `optimizer.zero_grad()`: Clears old gradients before computing new ones.
- `loss.backward()`: Computes the gradient of the loss with respect to model parameters.
- `optimizer.step()`: Updates model parameters based on the computed gradients.

## 5. Making Predictions (Forecasting)

Now, use the trained model to predict on the test set.

```python
model.eval() # Set model to evaluation mode
test_predictions_scaled = []
with torch.no_grad(): # Disable gradient calculations for inference
    for i in range(len(X_test_scaled)):
        input_seq = X_test_scaled[i:i+1] # Get one sequence, keep batch dimension
        prediction_scaled = model(input_seq)
        test_predictions_scaled.append(prediction_scaled.item())

# Convert predictions list to numpy array
test_predictions_scaled = np.array(test_predictions_scaled).reshape(-1, 1)

# Inverse transform the scaled predictions to get them in the original data scale
test_predictions = scaler.inverse_transform(test_predictions_scaled)

# Also inverse transform the actual test values for comparison
y_test_original = scaler.inverse_transform(y_test_scaled.numpy()) # y_test_scaled was a tensor

print("\nPredictions made on the test set.")
```

- `model.eval()`: Sets the model to evaluation mode.
- `torch.no_grad()`: Disables gradient computation, saving memory and speed during inference.
- **Inverse Transform**: Crucial to convert predictions back to the original scale if you scaled your data.

## 6. Evaluating the Model

### a. Metrics

Calculate metrics like Mean Squared Error (MSE) or Root Mean Squared Error (RMSE).

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test_original, test_predictions)
rmse = np.sqrt(mse)

print(f"\n--- Model Evaluation on Test Set ---")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
```

### b. Visualization

Plotting actual vs. predicted values is very insightful.

```python
plt.figure(figsize=(12, 6))
plt.plot(time, amplitude, label='Original Full Data', alpha=0.7)

# Plot training data portion used to create y_test_original
# y_test_original corresponds to amplitude values starting after the training sequences end
# and after the sequence_length look-back for the first test point.
# The first y_test_original value corresponds to amplitude[train_size + sequence_length]
test_start_index_original_series = train_size + sequence_length

plt.plot(time[test_start_index_original_series : test_start_index_original_series + len(y_test_original)],
         y_test_original, label='Actual Test Values', color='orange')
plt.plot(time[test_start_index_original_series : test_start_index_original_series + len(test_predictions)],
         test_predictions, label='Predicted Test Values', color='red', linestyle='--')

plt.title("Time Series Forecasting: Actual vs. Predicted")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
# plt.show() # Uncomment if running locally
print("Plot generated.")
```

## 7. Putting It All Together (Full Script)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- 1. Generate Sample Data ---
time_np = np.arange(0, 200, 0.1)
amplitude_np = np.sin(time_np) + np.random.normal(0, 0.15, len(time_np))

# --- 2. Data Preparation ---
# a. Scale original 1D series
amplitude_reshaped = amplitude_np.copy().reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
amplitude_scaled_np = scaler.fit_transform(amplitude_reshaped).flatten()

# b. Create sequences
sequence_length = 50
def create_sequences_np(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:(i + seq_len)])
        ys.append(data[i + seq_len])
    return np.array(xs), np.array(ys)

X_np_scaled, y_np_scaled = create_sequences_np(amplitude_scaled_np, sequence_length)

# c. Convert to PyTorch tensors
X_torch = torch.from_numpy(X_np_scaled).float()
y_torch = torch.from_numpy(y_np_scaled).float().unsqueeze(1)

# d. Split data
train_size_idx = int(len(X_torch) * 0.8)
X_train, X_test = X_torch[:train_size_idx], X_torch[train_size_idx:]
y_train, y_test = y_torch[:train_size_idx], y_torch[train_size_idx:]

# --- 3. Build Model ---
class SimpleMLPForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLPForecaster, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

input_dim_model = sequence_length
hidden_dim_model = 64
output_dim_model = 1
model_instance = SimpleMLPForecaster(input_dim_model, hidden_dim_model, output_dim_model)

# --- 4. Training ---
criterion_train = nn.MSELoss()
optimizer_train = torch.optim.Adam(model_instance.parameters(), lr=0.001)
num_epochs_train = 200
batch_size_train = 16

train_dataset_torch = TensorDataset(X_train, y_train)
train_loader_torch = DataLoader(train_dataset_torch, batch_size=batch_size_train, shuffle=True)

print("\nStarting Full Script Training...")
for epoch_idx in range(num_epochs_train):
    model_instance.train()
    for batch_X_train, batch_y_train in train_loader_torch:
        outputs_train = model_instance(batch_X_train)
        loss_train = criterion_train(outputs_train, batch_y_train)
        optimizer_train.zero_grad()
        loss_train.backward()
        optimizer_train.step()
    if (epoch_idx + 1) % 20 == 0:
        print(f'Epoch [{epoch_idx+1}/{num_epochs_train}], Loss: {loss_train.item():.6f}')
print("Full Script Training finished.")

# --- 5. Making Predictions ---
model_instance.eval()
test_preds_scaled_list = []
with torch.no_grad():
    for i in range(len(X_test)):
        input_seq_test = X_test[i:i+1]
        pred_scaled_test = model_instance(input_seq_test)
        test_preds_scaled_list.append(pred_scaled_test.item())

test_predictions_scaled_np = np.array(test_preds_scaled_list).reshape(-1, 1)
test_predictions_np = scaler.inverse_transform(test_predictions_scaled_np)
y_test_original_np = scaler.inverse_transform(y_test.numpy())

# --- 6. Evaluating ---
mse_eval = mean_squared_error(y_test_original_np, test_predictions_np)
rmse_eval = np.sqrt(mse_eval)
print(f"\n--- Full Script Model Evaluation ---")
print(f"Test MSE: {mse_eval:.4f}")
print(f"Test RMSE: {rmse_eval:.4f}")

# --- 7. Visualization ---
plt.figure(figsize=(14, 7))
plt.plot(time_np, amplitude_np, label='Original Full Data', alpha=0.5, color='blue')

# Determine the start index for plotting test actuals and predictions on the original time axis
test_plot_start_idx = train_size_idx + sequence_length
time_test_plot = time_np[test_plot_start_idx : test_plot_start_idx + len(y_test_original_np)]

plt.plot(time_test_plot, y_test_original_np, label='Actual Test Values', color='green')
plt.plot(time_test_plot, test_predictions_np, label='Predicted Test Values', color='red', linestyle='--')

plt.title("Full Script: Time Series Forecasting")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
# plt.show() # Uncomment if running locally
print("Full script plot generated.")
```

## 8. Next Steps and Further Learning

This tutorial covered a very simple MLP for forecasting. To improve and handle more complex time series:

- **Recurrent Neural Networks (RNNs)**: Specifically LSTMs (Long Short-Term Memory) or GRUs (Gated Recurrent Units) are designed to handle sequential data and learn long-term dependencies.
- **Convolutional Neural Networks (CNNs)**: 1D CNNs can also be effective for sequence data, often used in conjunction with RNNs.
- **Attention Mechanisms and Transformers**: State-of-the-art for many sequence modeling tasks.
- **Multivariate Forecasting**: Predicting multiple time series or using multiple input time series as predictors.
- **Probabilistic Forecasting**: Predicting a distribution of possible future outcomes instead of a single point estimate.
- **Hyperparameter Tuning**: Experimenting with `sequence_length`, number of layers, hidden units, learning rate, batch size, epochs, etc.
- **More Sophisticated Evaluation**: Using metrics like MAE (Mean Absolute Error), MAPE (Mean Absolute Percentage Error), and comparing against baseline models.
- **Handling Seasonality and Trend**: Explicitly modeling or removing these components before feeding data to a neural network can sometimes improve performance.

This foundation should give you a good start in exploring the exciting field of time series forecasting with PyTorch!
