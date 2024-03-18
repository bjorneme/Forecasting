from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

# Class for preparing dataset
class DataPreparation:
    def __init__(self, filepath, area_number=1, n_lags=24):
        self.filepath = filepath
        self.area_number = area_number
        self.n_lags = n_lags

    def prepare_data(self):
        # Step 1: Load the data
        data = pd.read_csv('consumption_and_temperatures.csv')
        print(data.shape)

        # Step 2: Add hour, day, month
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['month'] = data['timestamp'].dt.month
        data['day'] = data['timestamp'].dt.day
        data['hour'] = data['timestamp'].dt.hour

        # Step 3: Prepare features and target for the training data
        temperature = f'NO{self.area_number}_temperature'
        consumption = f'NO{self.area_number}_consumption'
        features = data[['month', 'day', 'hour', temperature]]
        target = data[consumption].values.reshape(-1, 1)

        # Step 4: Scale features and target
        scaler_features = MinMaxScaler(feature_range=(0, 1))
        scaled_features = scaler_features.fit_transform(features)

        scaler_target = MinMaxScaler(feature_range=(0, 1))
        scaled_target = scaler_target.fit_transform(target)

        # Step 5: Create sequence. Reserve last 24 hours for testing
        X_train, y_train = self.create_sequences(scaled_features[:-self.n_lags], scaled_target[:-self.n_lags]) 

        # Step 6: Use the n_lags last hours as input to predict the next n_lags hours
        X_test = scaled_features[-self.n_lags:]
        y_test = scaled_target[-self.n_lags:] 

        # Split into training and validation set
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        print(X_train_tensor.shape)
        print(y_train_tensor.shape)
        print(X_val_tensor.shape)
        print(y_val_tensor.shape)
        print(X_test_tensor.shape)
        print(y_test_tensor.shape)
 
        return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
    
    def create_sequences(self, input_data, target_data, forecast_horizon=24):
        X, y = [], []

        # Iterate over the traning dataset
        for i in range(len(input_data) - self.n_lags - forecast_horizon + 1):
            # Extract a sequence of n_lags
            train_input = input_data[i:i+self.n_lags]
            # Extract the target sequence of forecast_horizon
            train_target = target_data[i+self.n_lags:i+self.n_lags+forecast_horizon]

            X.append(train_input)
            y.append(train_target.flatten())

        return np.array(X), np.array(y)
