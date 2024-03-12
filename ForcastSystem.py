import joblib
import pandas as pd
import matplotlib.pyplot as plt

from DataPreparation import DataPreparation


class ForcastingSystem:
    def __init__(self, filepath, area_number, model=None, model_filepath=None):
        self.data_preparation = DataPreparation(filepath, area_number)
        self.model = model

        # Load a pretrained model
        self.model_filepath = model_filepath
        if self.model_filepath:
            self.load_model()

    def run_system(self):
        # Step 1: Prepare the data
        X_train, y_train, X_test, y_test = self.data_preparation.prepare_data()

        # Step 2: Train the model
        # TODO

        # Step 3: Save the trained model to a file
        self.save_model()
    
    def save_model(self):
        # Saves the trained model
        if self.model_filepath:
            joblib.dump(self.model, self.model_filepath)
            print(f"Model saved to {self.model_filepath}")

    def load_model(self):
        # Loads a pre-trained model
        try:
            self.model = joblib.load(self.model_filepath)
            print(f"Model loaded from {self.model_filepath}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {self.model_filepath}. Starting training from scratch.")