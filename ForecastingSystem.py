from matplotlib import pyplot as plt
from DataPreparation import DataPreparation

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ForecastingSystem:
    def __init__(self, filepath, area_number, model, num_epochs = 10, learning_rate = 0.001, model_filepath=None):
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
        X_train, y_train,  X_val, y_val, X_test, y_test = self.data_preparation.prepare_data()
        
        # Step 2: Train the model
        self.train_model(X_train, y_train, X_val, y_val)

        # Step 3: Evaluate the model
        self.evaluate(X_test, y_test)

        # Step 4: Save the trained model to a file
        self.save_model()

    def train_model(self, X_train, y_train, X_val, y_val):
        # Conceptual addition to train_model method:
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 2

        # Initialize a list to store training loss
        self.train_loss_history = []
        
        # Create dataset from the training data
        train_dataset = TensorDataset(X_train, y_train)
        
        # Create data loader for batch processing and shuffling
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        # Define loss function as MSE
        loss_function = nn.MSELoss()

        # Initialize the optimizer with model parameters and learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=1e-5)

        # After initializing the optimizer in the train_model method:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        
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
                scheduler.step()

                # Append the loss of this batch
                self.loss_history_epoch.append(loss.item())

            # Append average loss of the epoch. Used for visualization
            avg_loss = sum(self.loss_history_epoch)/len(train_loader)
            self.train_loss_history.append(avg_loss)

            # Validate the model
            val_loss = self.validate_model(X_val, y_val)

            print(f'Epoch {epoch+1}, Training Loss: {avg_loss} Validation Loss: {val_loss}')

            # At the end of each epoch, check if validation loss has improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset counter if validation loss improves
            else:
                patience_counter += 1
            if patience_counter > patience:
                print("Stopping early due to increasing validation loss.")
                break  # Break out of the loop if patience limit exceeded

    def validate_model(self,  X_val, y_val):

        # Initialize a list to store validation loss
        self.val_loss_history = []

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
        self.val_loss_history.append(total_loss)
        
        return total_loss
    
    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            # Predicting the next 24 hours
            y_pred = self.model(X_test.unsqueeze(0))

        # Plot predicted vs. actual values
        self.plot_actual_vs_predicted(y_test, y_pred[0])

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
            print(f"No pre-trained model found at {model_filepath}.")

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

    def plot_actual_vs_predicted(self, actual, predicted):
        # Plotting predicted vs. actual values
        plt.figure(figsize=(10, 6))
        plt.plot(predicted, label='Predicted', color='blue', marker='o')
        plt.plot(actual, label='Actual', color='red', marker='x')
        plt.title('Predicted vs. Actual Consumption')
        plt.xlabel('Hour')
        plt.ylabel('Normalized Consumption')
        plt.legend()
        plt.grid(True)
        plt.show()

        

