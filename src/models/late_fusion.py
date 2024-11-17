# src/models/late_fusion.py
"""Late fusion model implementation combining LSTM and Random Forest predictions using Lasso."""

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    LSTM, Dense, Input, Concatenate, Dropout, 
    BatchNormalization, Normalization, Embedding, Flatten
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
import tensorflow as tf

class LateFusionModel:
    def __init__(self, config):
        """
        Initialize late fusion model with configuration.
        
        Args:
            config (dict): Configuration parameters for the model
        """
        self.config = config
        self.lstm_model = None
        self.rf_model = None
        self.lasso_model = None
        self.scaler = MinMaxScaler()
        
    def build_lstm_model(self, input_shape=(4, 3)):
        """
        Build LSTM part of the model.
        
        Args:
            input_shape (tuple): Shape of input data (time_steps, features)
        
        Returns:
            Model: Compiled Keras model
        """
        # LSTM inputs
        lstm_input = Input(shape=input_shape)
        study_input = Input(shape=(1,))
        
        # Study embedding
        study_embedded = Embedding(input_dim=6, output_dim=3, input_length=1)(study_input)
        study_embedded = Flatten()(study_embedded)
        
        # LSTM layers
        lstm_out = LSTM(223, return_sequences=True)(lstm_input)
        dropout1 = Dropout(0.2)(lstm_out)
        lstm_out = LSTM(265, return_sequences=False)(dropout1)
        dropout2 = Dropout(0.2)(lstm_out)
        
        # Combine LSTM and study
        concat = Concatenate()([dropout2, study_embedded])
        
        # Normalize
        normalizer = Normalization()
        normalized_concat = normalizer(concat)
        
        # Dense layers
        dense_out = Dense(136, activation='relu')(normalized_concat)
        dropout3 = Dropout(0.3)(dense_out)
        final_output = Dense(1, activation='relu')(dropout3)
        
        # Create model
        model = Model(inputs=[lstm_input, study_input], outputs=final_output)
        
        # Compile model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
        self.lstm_model = model
        return model
    
    def build_rf_model(self):
        """Build Random Forest model."""
        self.rf_model = RandomForestRegressor(
            n_estimators=604,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=9,
            n_jobs=-1,
            random_state=42
        )
        
    def preprocess_data(self, X_lstm, X_study, is_training=True):
        """
        Preprocess data for LSTM model.
        
        Args:
            X_lstm: LSTM input data
            X_study: Study input data
            is_training: Whether this is training data
            
        Returns:
            Preprocessed data
        """
        if is_training:
            X_scaled_lstm = self.scaler.fit_transform(
                X_lstm.reshape(-1, X_lstm.shape[-1])
            ).reshape(X_lstm.shape)
        else:
            X_scaled_lstm = self.scaler.transform(
                X_lstm.reshape(-1, X_lstm.shape[-1])
            ).reshape(X_lstm.shape)
            
        return X_scaled_lstm, X_study
    
    def train(self, X_train_lstm, X_train_rf, X_study_train, y_train, 
              validation_data=None, epochs=100, batch_size=32):
        """
        Train both LSTM and RF models, then combine with Lasso.
        
        Args:
            X_train_lstm: Training data for LSTM
            X_train_rf: Training data for Random Forest
            X_study_train: Study data
            y_train: Target values
            validation_data: Validation data tuple
            epochs: Number of epochs for LSTM
            batch_size: Batch size for LSTM
        """
        # Preprocess data
        X_train_scaled_lstm, X_study_train = self.preprocess_data(
            X_train_lstm, X_study_train
        )
        
        # Train LSTM
        history = self.lstm_model.fit(
            [X_train_scaled_lstm, X_study_train],
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        
        # Train Random Forest
        self.rf_model.fit(X_train_rf, y_train)
        
        # Get predictions from both models
        lstm_preds = self.lstm_model.predict([X_train_scaled_lstm, X_study_train])
        rf_preds = self.rf_model.predict(X_train_rf)
        
        # Combine predictions
        combined_preds = np.column_stack((lstm_preds, rf_preds))
        
        # Train Lasso
        self.lasso_model = Lasso(alpha=0.003)
        self.lasso_model.fit(combined_preds, y_train)
        
        return history
    
    def predict(self, X_lstm, X_rf, X_study):
        """
        Make predictions using the trained models.
        
        Args:
            X_lstm: LSTM input data
            X_rf: Random Forest input data
            X_study: Study input data
            
        Returns:
            Final predictions from ensemble
        """
        # Preprocess data
        X_scaled_lstm, X_study = self.preprocess_data(
            X_lstm, X_study, is_training=False
        )
        
        # Get predictions from both models
        lstm_preds = self.lstm_model.predict([X_scaled_lstm, X_study])
        rf_preds = self.rf_model.predict(X_rf)
        
        # Combine predictions
        combined_preds = np.column_stack((lstm_preds, rf_preds))
        
        # Get final predictions
        return self.lasso_model.predict(combined_preds)