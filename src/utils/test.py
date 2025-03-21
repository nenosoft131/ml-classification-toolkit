# Importing the MinMaxScaler class from scikit-learn
from sklearn.preprocessing import MinMaxScaler
import numpy as np  # Import NumPy for working with arrays

# Sample data
# Data can be a list or a NumPy array with multiple features or samples
data = [1, 2, 3, 4, 5]

# Convert the data to a 2D array (scikit-learn expects data in a 2D format)
# Each row represents a sample, and each column represents a feature
data_reshaped = np.array(data).reshape(-1, 1)

# Create an instance of MinMaxScaler
# MinMaxScaler scales data to a specified range, default is [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scaler to the data and transform it
scaled_data = scaler.fit_transform(data_reshaped)

# Display the original data and scaled data
print("Original Data:")
print(data_reshaped)
print("\nScaled Data (after MinMax scaling):")
print(scaled_data)

# To demonstrate how to inverse the transformation and get back the original data
original_data = scaler.inverse_transform(scaled_data)
print("\nData After Inverse Transformation (back to original values):")
print(original_data)

# Example: Scaling a new value using the fitted scaler
new_value = np.array([[6]])  # New data must also be in 2D format
scaled_new_value = scaler.transform(new_value)
print("\nNew Value (scaled):")
print(scaled_new_value)
