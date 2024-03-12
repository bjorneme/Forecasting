import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Class for preparing dataset
class DataPreparation:
    def __init__(self, filepath, area_number, lags=24):
        self.filepath = filepath
        self.area_number = area_number
        self.lags = lags
        self.features = []
        self.target = f'NO{self.area_number}_consumption'
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
        # Convert timestamp to datetime and set as index
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        return data

    
    def feature_engineering(self, data):
        # Add lag features before any splitting
        data = self.add_lag_features(data)

        # Adding time-based features
        data['hour_of_day'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month

        # Now, split the data into training and testing
        test_df = data.iloc[-24:].dropna()
        train_df = data.iloc[:-24].dropna()


        # Specify features and target variable
        features = [f'NO{self.area_number}_temperature', 'hour_of_day', 'day_of_week', 'month'] + [f'lag_{i}' for i in range(1, self.lags + 1)]
        target = f'NO{self.area_number}_consumption'

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(-1,1))
        X_train = scaler.fit_transform(train_df[features])
        y_train = train_df[target].values
        self.y_train = y_train
        X_test = scaler.transform(test_df[features])
        y_test = test_df[target].values

        # Return processed training and testing data
        return X_train, y_train, X_test, y_test
    
    def add_lag_features(self, data):
        # Add lagged features to the DataFrame.
        target = f'NO{self.area_number}_consumption'
        for i in range(1, self.lags + 1):
            data[f'lag_{i}'] = data[target].shift(i)
        return data

