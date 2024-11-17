import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config

    def load_environmental_data(self):
        """Load rainfall, solar, and temperature data."""
        dataset_rainfall = pd.read_csv(f'{self.config.DATA_DIR}/avg_rainfall_dataset.csv')
        dataset_solar = pd.read_csv(f'{self.config.DATA_DIR}/avg_solar_dataset.csv')
        dataset_temp = pd.read_csv(f'{self.config.DATA_DIR}/avg_max_temp_dataset.csv')
        return dataset_rainfall, dataset_solar, dataset_temp

    def load_gene_data(self):
        """Load gene data."""
        return pd.read_csv(f'{self.config.DATA_DIR}/final_cleaned_merge_imputed.csv')