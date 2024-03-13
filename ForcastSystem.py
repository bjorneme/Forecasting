import joblib
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

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
        self.train_model(X_train, y_train)

        # Step 3: Evaluate the model
        # TODO

        # Step 3: Save the trained model to a file
        self.save_model()

    def train_model(self, X_train, y_train):

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        epochs = 10
        for epoch in range(epochs):
            self.model.train()
            for seq, labels in train_loader:
                optimizer.zero_grad()
                y_pred = self.model(seq)
                loss = loss_function(y_pred.squeeze(), labels)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    
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