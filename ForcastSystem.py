import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from DataPreparation import DataPreparation


class ForcastingSystem:
    def __init__(self, filepath, area_number, model, num_epochs, learning_rate, model_filepath=None):
        self.data_preparation = DataPreparation(filepath, area_number)
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Load a pretrained model
        self.model_filepath = model_filepath
        if self.model_filepath:
            self.load_model()

    def run_system(self):
        # Step 1: Prepare the data
        X_train, y_train, X_test, y_test = self.data_preparation.prepare_data()

        self.data_preparation.visulaize_data()

        # Step 2: Train the model
        self.train_model(X_train, y_train)

        # Step 3: Evaluate the model
        self.evaluate_model(X_test, y_test)

        # Step 4: Save the trained model to a file
        self.save_model()

    def train_model(self, X_train, y_train):
        # Initialize a list to store loss
        self.loss_history = []
        
        # Create dataset from the training data
        train_dataset = TensorDataset(X_train, y_train)
        
        # Create data loader for batch processing and shuffling
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        # Define loss function as MSE
        loss_function = nn.MSELoss()

        # Initialize the optimizer with model parameters and learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate)

        # Iterate over the dataset for a defined number of epochs
        for epoch in range(self.num_epochs):
            self.loss_history_epoch = []
            
            # Set the model to training mode (enables dropout, batchnorm updates)
            self.model.train()

            # Loop over batches of data from the data loader
            for input_features, target in train_loader:

                # Reset gradients to zero before backpropagatio
                optimizer.zero_grad()

                # Forward pass
                y_pred = self.model(input_features)

                # Calculate the loss 
                loss = loss_function(y_pred.squeeze(), target)
                
                # Backward pass
                loss.backward()

                # Update model parameters based on gradients
                optimizer.step()

                # Append the loss of this batch
                self.loss_history_epoch.append(loss.item())

            # Append average loss of the epoch. Used for visualization
            avg_loss = sum(self.loss_history_epoch)/len(train_loader)
            self.loss_history.append(avg_loss)


            print(f'Epoch {epoch+1}, Loss: {avg_loss}')


        # Plot the loss history
        self.data_preparation.visualize_learning_progress(self.loss_history)

    def evaluate_model(self, X_test, y_test):

        # Create dataset from the test data
        train_dataset = TensorDataset(X_test, y_test)
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=1)

        # Set the model to evaluation mode
        self.model.eval()

        # Initialize a list to store predictions
        predictions = []

        # Disable gradient computation for evaluation
        with torch.no_grad():

            # Iterate over individual data points in test dataset
            for input_features, _ in train_loader:
                
                # Calculate the models predictions
                y_test_pred = self.model(input_features)
                
                # Append the predictions to the prediction list
                predictions.append(y_test_pred.numpy().flatten()[0])

        # Visualize the predictions vs. actual consumption
        self.data_preparation.visualize(predictions)

    def save_model(self):
        # Saves the trained model
        if self.model_filepath:
            torch.save(self.model.state_dict(), self.model_filepath)
            print(f"Model saved to {self.model_filepath}")

    def load_model(self):
        # Loads a pre-trained model
        try:
            self.model.load_state_dict(torch.load(self.model_filepath))
            self.model.eval()
            print(f"Model loaded from {self.model_filepath}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {self.model_filepath}. Starting training from scratch.")