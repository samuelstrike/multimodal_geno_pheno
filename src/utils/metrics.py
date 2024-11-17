import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class PerformanceMetrics:
    def __init__(self):
        self.train_mse = []
        self.train_mae = []
        self.train_r2 = []
        self.test_mse = []
        self.test_mae = []
        self.test_r2 = []

    def calculate_metrics(self, y_true_train, y_pred_train, y_true_test, y_pred_test):
        """Calculate performance metrics for both training and test sets."""
        # Training metrics
        self.train_mse.append(mean_squared_error(y_true_train, y_pred_train))
        self.train_mae.append(mean_absolute_error(y_true_train, y_pred_train))
        self.train_r2.append(r2_score(y_true_train, y_pred_train))

        # Test metrics
        self.test_mse.append(mean_squared_error(y_true_test, y_pred_test))
        self.test_mae.append(mean_absolute_error(y_true_test, y_pred_test))
        self.test_r2.append(r2_score(y_true_test, y_pred_test))

    def get_summary(self):
        """Get summary of all metrics."""
        summary = {
            'train': {
                'mse': np.mean(self.train_mse),
                'rmse': np.sqrt(np.mean(self.train_mse)),
                'mae': np.mean(self.train_mae),
                'r2': np.mean(self.train_r2),
                'mse_std': np.std(self.train_mse)
            },
            'test': {
                'mse': np.mean(self.test_mse),
                'rmse': np.sqrt(np.mean(self.test_mse)),
                'mae': np.mean(self.test_mae),
                'r2': np.mean(self.test_r2),
                'mse_std': np.std(self.test_mse)
            }
        }
        return summary

    def print_summary(self):
        """Print formatted summary of all metrics."""
        summary = self.get_summary()
        
        print("Training Metrics:")
        print(f"MSE: {summary['train']['mse']:.2f}")
        print(f"RMSE: {summary['train']['rmse']:.2f}")
        print(f"MAE: {summary['train']['mae']:.2f}")
        print(f"R2: {summary['train']['r2']:.2f}")
        print(f"Standard deviation of Train MSE: {summary['train']['mse_std']:.2f}\n")

        print("Test Metrics:")
        print(f"MSE: {summary['test']['mse']:.2f}")
        print(f"RMSE: {summary['test']['rmse']:.2f}")
        print(f"MAE: {summary['test']['mae']:.2f}")
        print(f"R2: {summary['test']['r2']:.2f}")
        print(f"Standard deviation of Test MSE: {summary['test']['mse_std']:.2f}")