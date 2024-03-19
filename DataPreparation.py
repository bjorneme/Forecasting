from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import matplotlib.dates as mdates

# Class for preparing dataset
class DataPreparation:
    def __init__(self, filepath, area_number=1, forcast_range = 24, num_lags=24):

        # Initialize necessary parameters
        self.filepath = filepath
        self.area_number = area_number
        self.forcast_range = forcast_range
        self.n_lags = num_lags
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self):
        # Step 1: Load the data
        data = pd.read_csv('consumption_and_temperatures.csv')
        self.data = data # Saved for plotting

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
        scaled_features = self.scaler.fit_transform(features)
        scaled_target = self.scaler.fit_transform(target)

        # Step 5: Create sequence. Reserve last 24 hours for testing
        X_train, y_train = self.create_sequences(scaled_features[:-self.forcast_range], scaled_target[:-self.forcast_range]) 

        # Step 6: Use the n_lags last hours as input to predict the next forcast_range hours
        X_test = scaled_features[-self.forcast_range:]
        y_test = scaled_target[-self.forcast_range:] 

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
    
    def create_sequences(self, input_data, target_data):
        X, y = [], []

        # Iterate over the traning dataset
        for i in range(len(input_data) - self.n_lags - self.forcast_range + 1):
            # Extract a sequence of n_lags
            train_input = input_data[i:i+self.n_lags]
            # Extract the target sequence of forecast_horizon
            train_target = target_data[i+self.n_lags:i+self.n_lags+self.forcast_range]

            X.append(train_input)
            y.append(train_target.flatten())

        return np.array(X), np.array(y)

    def visualize_dataset(self):
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))  # 2 rows, 1 column

        # Original Data
        timestamp = self.data['timestamp']
        consumption_col = f'NO{self.area_number}_consumption'
        temperature_col = f'NO{self.area_number}_temperature'

        # Plot original data
        ax1 = axs[0]
        ax1.plot(timestamp, self.data[consumption_col], label='Consumption', color='blue')
        ax1.set_ylabel('Consumption', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(timestamp, self.data[temperature_col], label='Temperature', color='red')
        ax2.set_ylabel('Temperature', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title('Original Consumption and Temperature')

        # Normalize Data
        normalized_data = self.scaler.fit_transform(self.data[[consumption_col, temperature_col]])

        # Plot normalized data
        ax1n = axs[1]
        ax1n.plot(timestamp, normalized_data[:, 0], label='Normalized Consumption', color='blue')
        ax1n.set_ylabel('Normalized Consumption', color='blue')
        ax1n.tick_params(axis='y', labelcolor='blue')

        ax2n = ax1n.twinx()
        ax2n.plot(timestamp, normalized_data[:, 1], label='Normalized Temperature', color='red')
        ax2n.set_ylabel('Normalized Temperature', color='red')
        ax2n.tick_params(axis='y', labelcolor='red')

        ax1n.legend(loc='upper left')
        ax2n.legend(loc='upper right')
        ax1n.set_title('Normalized Consumption and Temperature')

        # Formatting
        for ax in [ax1, ax1n]:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        fig.autofmt_xdate()  # Auto-format date labels for better readability
        plt.tight_layout()
        plt.show()