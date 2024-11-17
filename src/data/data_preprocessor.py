import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()

    def preprocess_environmental_data(self, rainfall_data, solar_data, temp_data):
        """Preprocess environmental data."""
        # Common preprocessing steps
        for dataset in [rainfall_data, solar_data, temp_data]:
            dataset.drop(columns=['Variety', 'Reason', 'Location', 'ZS49PlHt', 
                                'HrvPlHt', 'ZS91 (no days)', 'GrYld(kg/ha)'], 
                        inplace=True)
            dataset['Study'] = dataset['Study'].fillna('No Treatment')
            dataset.dropna(inplace=True)
        
        # Reshape data for LSTM
        rainfall_reshaped = rainfall_data.values.reshape(-1, 5, 1)
        solar_reshaped = solar_data.values.reshape(-1, 5, 1)
        temp_reshaped = temp_data.values.reshape(-1, 5, 1)
        
        return np.concatenate((rainfall_reshaped, solar_reshaped, temp_reshaped), 
                            axis=2)

    def preprocess_gene_data(self, merged_df):
        """Preprocess gene data."""
        dataset_zs49 = merged_df.copy()
        dataset_zs49.drop(columns=['ZS49PlHt', 'GrYld(kg/ha)'], inplace=True)
        dataset_zs49.dropna(subset=['ZS49 (no days)'], inplace=True)
        
        X = dataset_zs49.iloc[:, 6:]
        y = dataset_zs49['ZS49 (no days)']
        
        return X, y