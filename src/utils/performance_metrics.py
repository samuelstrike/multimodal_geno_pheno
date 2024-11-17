# src/utils/performance_metrics.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_fold_metrics(y_true, y_pred):
    """Calculate metrics for a single fold."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'r2': r2
    }

def print_final_metrics(train_metrics, test_metrics):
    """Print final metrics for both training and test sets."""
    print("\nFinal Results:")
    print("=" * 50)
    
    for split, metrics in [("Training", train_metrics), ("Test", test_metrics)]:
        print(f"\n{split} Metrics:")
        print("-" * 20)
        print(f"MSE: {np.mean(metrics['mse']):.2f} ± {np.std(metrics['mse']):.2f}")
        print(f"RMSE: {np.sqrt(np.mean(metrics['mse'])):.2f}")
        print(f"MAE: {np.mean(metrics['mae']):.2f} ± {np.std(metrics['mae']):.2f}")
        print(f"R²: {np.mean(metrics['r2']):.4f} ± {np.std(metrics['r2']):.4f}")