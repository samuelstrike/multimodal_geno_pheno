# config.py

class Config:
    """Configuration settings for the project."""
    
    # Data paths
    DATA_DIR = "data"
    MODEL_DIR = "models"
    LOGS_DIR = "logs"
    RESULTS_DIR = "results"

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    PATIENCE = 10
    N_SPLITS = 3
    RANDOM_STATE = 42
    VALIDATION_SPLIT = 0.2

    # Model parameters
    LSTM_UNITS = [223, 265]
    DENSE_UNITS = [136, 128]
    DROPOUT_RATES = [0.2, 0.2, 0.3]
    
    # RandomForest parameters
    RF_PARAMS = {
        'n_estimators': 604,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 9,
        'n_jobs': -1,
        'random_state': 42
    }
    
    # Lasso parameter
    LASSO_ALPHA = 0.003
