# train.py

from sklearn.model_selection import KFold
import numpy as np
from src.utils.performance_metrics import calculate_fold_metrics, print_final_metrics
from src.models.late_fusion import LateFusionModel

def train_intermediate_fusion(model, env_data, gene_data, study_data, target, config):
    """Train intermediate fusion model."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    metrics = {'train': [], 'test': []}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(env_data), 1):
        print(f"\nIntermediate Fusion - Fold {fold}/3")
        
        # Split data
        X_train_env = env_data[train_idx]
        X_test_env = env_data[test_idx]
        X_train_gene = gene_data.iloc[train_idx]
        X_test_gene = gene_data.iloc[test_idx]
        X_study_train = study_data[train_idx]
        X_study_test = study_data[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]
        
        # Train model
        history = model.fit(
            [X_train_env, X_train_gene, X_study_train],
            y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_split=0.2,
            verbose=1
        )
        
        # Get predictions
        train_preds = model.predict([X_train_env, X_train_gene, X_study_train])
        test_preds = model.predict([X_test_env, X_test_gene, X_study_test])
        
        # Calculate metrics
        train_fold_metrics = calculate_fold_metrics(y_train, train_preds)
        test_fold_metrics = calculate_fold_metrics(y_test, test_preds)
        
        metrics['train'].append(train_fold_metrics)
        metrics['test'].append(test_fold_metrics)
        
        print(f"\nFold {fold} Results:")
        print(f"Train MSE: {train_fold_metrics['mse']:.2f}")
        print(f"Test MSE: {test_fold_metrics['mse']:.2f}")
    
    return metrics

def train_late_fusion(env_data, gene_data, study_data, target, config):
    """Train late fusion model."""
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    metrics = {'train': [], 'test': []}
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(env_data), 1):
        print(f"\nLate Fusion - Fold {fold}/3")
        
        # Split data
        X_train_env = env_data[train_idx]
        X_test_env = env_data[test_idx]
        X_train_gene = gene_data.iloc[train_idx]
        X_test_gene = gene_data.iloc[test_idx]
        X_study_train = study_data[train_idx]
        X_study_test = study_data[test_idx]
        y_train = target[train_idx]
        y_test = target[test_idx]
        
        # Initialize and train late fusion model
        model = LateFusionModel(config)
        model.build_lstm_model()
        model.build_rf_model()
        
        history = model.train(
            X_train_lstm=X_train_env,
            X_train_rf=X_train_gene,
            X_study_train=X_study_train,
            y_train=y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE
        )
        
        # Get predictions
        train_preds = model.predict(X_train_env, X_train_gene, X_study_train)
        test_preds = model.predict(X_test_env, X_test_gene, X_study_test)
        
        # Calculate metrics
        train_fold_metrics = calculate_fold_metrics(y_train, train_preds)
        test_fold_metrics = calculate_fold_metrics(y_test, test_preds)
        
        metrics['train'].append(train_fold_metrics)
        metrics['test'].append(test_fold_metrics)
        
        print(f"\nFold {fold} Results:")
        print(f"Train MSE: {train_fold_metrics['mse']:.2f}")
        print(f"Test MSE: {test_fold_metrics['mse']:.2f}")
    
    return metrics