import pandas as pd
from DataPreparation import DataPreparation


class ForcastingSystem:
    def __init__(self, filepath, model=None):
        self.data_preparation = DataPreparation(filepath)
        self.model = model

    def run_system(self):
        # Step 1: Prepare the data
        data = self.data_preparation.prepare_data()

        return data

    def visualize_results(self):
        pass