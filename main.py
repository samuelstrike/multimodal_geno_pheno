# main.py

import os
from config import Config
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.models.intermediate_fusion import intermediate_fusion
from src.train import train_intermediate_fusion, train_late_fusion
from src.utils.performance_metrics import print_final_metrics

def main():
    # Initialize config
    config = Config()
    
    # Create necessary directories
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data_loader = DataLoader(config)
    preprocessor = DataPreprocessor()
    
    rainfall_data, solar_data, temp_data = data_loader.load_environmental_data()
    merged_df = data_loader.load_gene_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    env_data = preprocessor.preprocess_environmental_data(
        rainfall_data, solar_data, temp_data
    )
    gene_data, target = preprocessor.preprocess_gene_data(merged_df)
    study_data = merged_df['Study'].values
    
    # Train Intermediate Fusion Model
    print("\nTraining Intermediate Fusion Model...")
    int_fusion_model = intermediate_fusion(
        input_lstm=(5, 3),
        input_cnn=(gene_data.shape[1], 1),
        input_study=(1,)
    )
    
    int_fusion_metrics = train_intermediate_fusion(
        model=int_fusion_model,
        env_data=env_data,
        gene_data=gene_data,
        study_data=study_data,
        target=target,
        config=config
    )
    
    # Train Late Fusion Model
    print("\nTraining Late Fusion Model...")
    late_fusion_metrics = train_late_fusion(
        env_data=env_data,
        gene_data=gene_data,
        study_data=study_data,
        target=target,
        config=config
    )
    
    # Print final results
    print("\nIntermediate Fusion Results:")
    print("=" * 50)
    print_final_metrics(int_fusion_metrics['train'], int_fusion_metrics['test'])
    
    print("\nLate Fusion Results:")
    print("=" * 50)
    print_final_metrics(late_fusion_metrics['train'], late_fusion_metrics['test'])

if __name__ == "__main__":
    main()