import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

# Class for preparing dataset
class DataPreparation:
    def __init__(self, filepath, area_number=1, lags=24):
        self.filepath = filepath
        self.area_number = area_number
        self.lags = lags

    def prepare_data(self):
        # Load data
        data = self.load_data(self.filepath)
        
        # Preprocess data
        preprocessed_data = self.preprocess_data(data)
        
        # Feature Engineering + Splitting
        X_train, y_train, X_test, y_test = self.feature_engineering(preprocessed_data)
        
        # Return prepared data
        return X_train, y_train, X_test, y_test

    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def preprocess_data(self, data):
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])

        return data

    
    def feature_engineering(self, data):

        # Adding time-based features
        data['month'] = data['timestamp'].dt.month
        data['day'] = data['timestamp'].dt.day
        data['hour'] = data['timestamp'].dt.hour

        # Select Features
        temperature_column = f'NO{self.area_number}_temperature'
        consumption_column = f'NO{self.area_number}_consumption'
        features = data[['month', 'day', 'hour', temperature_column, consumption_column]]

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_features = scaler.fit_transform(features)

        # Create lags
        X, y = self.add_lag_features(scaled_features, self.lags)

        # Split and convert to tensors
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=24, shuffle=False)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        # Return splittet dataset
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def add_lag_features(self, scaled_features, lags):
        # Add lagged features to the DataFrame.
        X, y = [], []
        for i in range(len(scaled_features) - lags):
            seq_x = scaled_features[i:i+lags, :-1]
            seq_y = scaled_features[i+lags, -1]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

        

