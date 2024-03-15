import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from DataPreparation import DataPreparation

class ForecastingSystem:
    def __init__(self, filepath, area_number, model, num_epochs, learning_rate, model_filepath=None):
        self.data_preparation = DataPreparation(filepath, area_number)
        self.model = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Load a pretrained model
        self.model_filepath = model_filepath
        if self.model_filepath:
            self.load_model(self.model_filepath)

    def run_system(self):
        # Step 1: Prepare the data
        X_train, y_train, X_val, y_val, X_test, y_test = self.data_preparation.prepare_data()
        self.X_test = X_test
        self.y_test = y_test
        
        # self.data_preparation.visulaize_data()

        # Step 2: Train the model
        self.train_model(X_train, y_train, X_val, y_val)

        # Step 3: Evaluate the model
        self.evaluate_model(X_test, y_test)

        # Step 4: Save the trained model to a file
        self.save_model()

    def train_model(self, X_train, y_train, X_val, y_val):
        # Initialize a list to store loss
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Create dataset from the training data
        train_dataset = TensorDataset(X_train, y_train)
        
        # Create data loader for batch processing and shuffling
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

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
            self.train_loss_history.append(avg_loss)
            # At the end of each epoch, validate the model
            val_loss = self.validate_model(X_val, y_val)
            self.val_loss_history.append(val_loss)
            print(f'Epoch {epoch+1}, Training Loss: {avg_loss} Validation Loss: {val_loss}')


        # Plot the loss history
        self.data_preparation.visualize_learning_progress(self.train_loss_history, self.val_loss_history)

    def validate_model(self, X_val, y_val):
        # Similar to the evaluate_model method but returns the average loss on the validation dataset
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Set model to evaluation mode
        self.model.eval()
        
        total_loss = 0
        count = 0
        with torch.no_grad():
            for input_features, target in val_loader:
                y_pred = self.model(input_features)
                loss = nn.MSELoss()(y_pred.squeeze(), target)
                total_loss += loss.item()
                count += 1
        
        return total_loss / count

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
        # self.data_preparation.visualize(predictions)

        # Return the populated list of predictions
        return predictions

    def save_model(self):
        # Saves the trained model
        if self.model_filepath:
            torch.save(self.model.state_dict(), self.model_filepath)
            print(f"Model saved to {self.model_filepath}")

    def load_model(self, model_filepath):
        # Loads a pre-trained model
        try:
            self.model.load_state_dict(torch.load(model_filepath))
            self.model.eval()
            print(f"Model loaded from {model_filepath}")
        except FileNotFoundError:
            print(f"No pre-trained model found at {model_filepath}. Starting training from scratch.")

    def evaluate_and_plot_models(self, model_paths, models):
        # Create figure and axes for subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

        # Plot actual values on the first subplot
        ax1.plot(self.y_test, label='Actual', color='black', linewidth=2)
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
                ax1.plot(predictions, label=f'{model_name} Predictions')

                # Calculate error
                error = np.abs(predictions - self.y_test.squeeze().numpy())  # Adjust if necessary

                # Plot error on the second subplot
                ax2.plot(error, label=f'{model_name} Error')

        ax2.set_title("Error for Each Model")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Absolute Error")
        ax2.legend()

        # Show the plots
        plt.tight_layout()
        plt.show()
