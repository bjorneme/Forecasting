from matplotlib import pyplot as plt
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
        self.scaler = MinMaxScaler(feature_range=(-1,1))

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
        self.features = data[['month', 'day', 'hour', temperature_column, consumption_column]]

        # Normalize the data
        self.scaled_features = self.scaler.fit_transform(self.features)

        # Create lags
        X, y = self.add_lag_features(self.scaled_features, self.lags)

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
    
    def visualize(self, predictions):
        # Inversely scale the predictions
        actual_predictions = self.scaler.inverse_transform(np.hstack((np.zeros((len(predictions), 4)), np.array(predictions).reshape(-1, 1))))[:, -1]

        # Plotting actual vs predicted
        plt.figure(figsize=(10,6))
        # Extract the actual 'NO1_consumption' values for the last part of the dataset for plotting
        actual_consumption = self.scaler.inverse_transform(self.scaled_features)[-len(predictions):, -1]
        plt.plot(actual_consumption, label='Actual Consumption')
        plt.plot(actual_predictions, label='Predicted Consumption')
        plt.legend()
        plt.show()

    def visualize_learning_progress(self,loss_history):
        # Plot both the loss
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label='Training Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def visulaize_data(self):

        # Plotting electricity consumption and temperature trends for NO1
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot consumption
        consumption_col = f'NO{self.area_number}_consumption'
        ax1.plot(self.features[consumption_col], label=f'Area {self.area_number} Consumption')
        ax1.set_ylabel('Consumption')
        ax1.legend()
    
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # Plot temperature
        temperature_col = f'NO{self.area_number}_temperature'
        ax2.plot(self.features[temperature_col], label=f'Area {self.area_number} Temperature', color='red')
        ax2.set_ylabel('Temperature')
        ax2.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title(f'NO{self.area_number} Consumption and Temperature Trends Over Time')
        plt.show()  
        

