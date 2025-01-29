from sklearn.preprocessing import MinMaxScaler

# Sample data
data = [1, 2, 3, 4, 5]

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform([data])

# Print the scaled data
print(scaled_data)