from matplotlib import pyplot as plt
import numpy as np
from DataPreparation import DataPreparation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ForecastingSystem:
    def __init__(self, filepath, area_number, forcast_range, num_lags, model, num_epochs = 10, learning_rate = 0.001, model_filepath=None):
        
        # Initialize necessary parameters
        self.data_preparation = DataPreparation(filepath, area_number, forcast_range, num_lags)
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Load a pretrained model
        self.model_filepath = model_filepath
        if self.model_filepath:
            self.load_model(self.model_filepath)

    def run_system(self):
        # Step 1: Prepare the data
        X_train, y_train,  X_val, y_val, X_test, y_test = self.data_preparation.prepare_data()
        self.X_test = X_test
        self.y_test = y_test
        
        # Step 2: Train the model
        self.train_model(X_train, y_train, X_val, y_val)

        # Step 3: Evaluate the model
        # self.evaluate_model(X_test, y_test)

        # Step 4: Save the trained model to a file
        self.save_model()

    def train_model(self, X_train, y_train, X_val, y_val):

        # Initialize a list to store training loss
        self.train_loss_history = []
        # Initialize a list to store validation loss
        self.val_loss_history = []
        
        # Create dataset from the training data
        train_dataset = TensorDataset(X_train, y_train)
        
        # Create data loader for batch processing and shuffling
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Define loss function as MSE
        loss_function = nn.MSELoss()

        # Initialize the optimizer with model parameters and learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-5)
        
        # Set the model to training mod
        self.model.train()

        # Iterate over the dataset for a defined number of epochs
        for epoch in range(self.num_epochs):
            self.loss_history_epoch = []

            # Loop over batches of data from the data loader
            for X_batch, y_batch in train_loader:
                
                # Reset gradients to zero before backpropagation
                optimizer.zero_grad()

                # Forward pass
                output = self.model(X_batch)

                # Calculate the loss
                loss = loss_function(output, y_batch)

                # Backward pass
                loss.backward()

                # Update model parameters based on gradients
                optimizer.step()

                # Append the loss of this batch
                self.loss_history_epoch.append(loss.item())

            # Append average loss of the epoch. Used for visualization
            total_train_loss = sum(self.loss_history_epoch)/len(train_loader)
            self.train_loss_history.append(total_train_loss)

            # Validate the model
            total_val_loss = self.validate_model(X_val, y_val)

            print(f'Epoch {epoch+1}, Training Loss: {total_train_loss} Validation Loss: {total_val_loss}')

    def validate_model(self,  X_val, y_val):

        # Create dataset from the validation data
        val_dataset = TensorDataset(X_val, y_val)

        # Create data loader for batch processing
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Define loss function as MSE
        loss_function = nn.MSELoss()

        # Set model to evaluation mode
        self.model.eval()

        total_loss = 0

        with torch.no_grad():
            # Loop over batches of data from the data loader
            for X_batch, y_batch in val_loader:

                # Forward pass
                output = self.model(X_batch)

                # Calculate the loss
                loss = loss_function(output, y_batch)

                total_loss += loss.item()

        # Append total loss from validation. Used for visualization
        total_val_loss = total_loss/len(val_loader)
        self.val_loss_history.append(total_val_loss)
        
        return total_val_loss
    
    def evaluate_model(self, X_test, y_test):
        # Set to evaluation mode
        self.model.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Predicting the next 24 hours
            y_pred = self.model(X_test.unsqueeze(0))

        return y_pred

    def save_model(self):
        # Saves the trained model
        if self.model_filepath:
            torch.save(self.model.state_dict(), self.model_filepath)
            print(f"Model saved: {self.model_filepath}")

    def load_model(self, model_filepath):
        # Loads a pre-trained model
        try:
            self.model.load_state_dict(torch.load(model_filepath))
            self.model.eval()
            print(f"Model loaded: {model_filepath}")
        except FileNotFoundError:
            print(f"Pre-trained model not found: {model_filepath}.")

    def visualize_learning_progress(self):
        # Plot both the loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss', color='orange')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def visualize_dataset(self):
        self.data_preparation.visualize_dataset()

    def evaluate_and_plot_models(self, model_paths, models):
        # Create figure and axes for subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

        # Inverse transform the actual values for plotting
        actual_values = self.data_preparation.scaler.inverse_transform(self.y_test)

        # Plot actual values on the first subplot
        ax1.plot(actual_values, label='Actual', color='black', linewidth=2)
        ax1.set_title("Model Predictions vs Actual")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Value")
        ax1.legend()

        for model_name, model in models.items():
            self.model = model
            model_path = model_paths.get(model_name)
            if model_path:
                try:
                    self.load_model(model_path)
                    print(f"Model {model_name} loaded successfully from {model_path}.")
                except FileNotFoundError:
                    print(f"Model file not found at {model_path}. Skipping {model_name}.")
                    continue
                
                # Use the loaded model for evaluation
                predictions = self.evaluate_model(self.X_test, self.y_test)
                # Ensure predictions are reshaped appropriately for inverse transformation
                predictions_reshaped = predictions.reshape(-1, 1)
                predictions_inverse = self.data_preparation.scaler.inverse_transform(predictions_reshaped)

                # Plot the inverse-transformed predictions
                ax1.plot(predictions_inverse, label=f'{model_name} Predictions')

                # Calculate error
                error = np.abs(predictions_inverse.squeeze() - actual_values.squeeze())

                # Plot error on the second subplot
                ax2.plot(error, label=f'{model_name} Error')

        ax2.set_title("Error for Each Model")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Absolute Error")
        ax2.legend()

        # Show the plots
        plt.tight_layout()
        plt.show()