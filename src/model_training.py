"""
Model Training Module
Trains and evaluates ML models for lap time and pit stop prediction.
INCLUDES: XGBoost (Regression) + LSTM (Deep Learning)
FIXED: Auto-feature engineering for missing columns.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from typing import Dict, Tuple

class F1ModelTrainer:
    """Trains ML models for F1 predictions including Deep Learning"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                     n_estimators: int = 100, max_depth: int = 6,
                     learning_rate: float = 0.1) -> xgb.XGBRegressor:
        """Train XGBoost model"""
        print("\n=== Training XGBoost ===")
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"Model trained with {n_estimators} estimators")
        return model

    # --- LSTM DEEP LEARNING MODULE ---
    def prepare_lstm_data(self, df, feature_cols, target_col, look_back=5):
        """
        Prepares sequences for LSTM (Time Series).
        Creates a sliding window of data.
        """
        # Verify columns exist before processing
        available_cols = [c for c in feature_cols if c in df.columns]
        if len(available_cols) != len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            raise KeyError(f"Missing columns for LSTM: {missing}")
            
        print(f"Preparing LSTM sequences with features: {available_cols}")
            
        data = df[available_cols + [target_col]].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(data_scaled) - look_back):
            # Sequence of 'look_back' laps
            X.append(data_scaled[i:(i + look_back), :-1])
            # Target: The NEXT lap time
            y.append(data_scaled[i + look_back, -1])
            
        return np.array(X), np.array(y), scaler, len(available_cols)

    def train_lstm(self, X_train, y_train, look_back, n_features):
        """
        Train Long Short-Term Memory (LSTM) Neural Network.
        """
        print(f"\n=== Training LSTM (Deep Learning) with input shape ({look_back}, {n_features}) ===")
        
        model = Sequential()
        # Layer 1: LSTM with memory
        model.add(LSTM(50, return_sequences=True, input_shape=(look_back, n_features)))
        model.add(Dropout(0.2))
        
        # Layer 2: LSTM to capture deeper patterns
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output Layer
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
        
        print("LSTM Training Complete.")
        return model
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """Evaluate model performance"""
        print(f"\n=== Evaluating {model_name} ===")
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def save_model(self, model, model_name: str, output_dir: str = './data/models'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(model, Sequential):
            model.save(os.path.join(output_dir, f'{model_name}.keras')) # New Keras format
            print(f"Deep Learning Model saved to {output_dir}/{model_name}.keras")
        else:
            filepath = os.path.join(output_dir, f'{model_name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {filepath}")

def engineer_features(df):
    """
    CRITICAL STEP: Fixes missing columns for the student project.
    Creates 'TireAge' and 'FuelLoad' if they don't exist.
    """
    print("\n=== Engineering Features ===")
    
    # 1. Fix Tire Age (FastF1 uses 'TyreLife')
    if 'TyreLife' in df.columns and 'TireAge' not in df.columns:
        print("Mapped 'TyreLife' -> 'TireAge'")
        df['TireAge'] = df['TyreLife']
    elif 'TireAge' not in df.columns:
        print("⚠️ 'TyreLife' missing. Creating synthetic Tire Age.")
        # Synthetic: Reset count every time 'Compound' changes or Driver changes
        df['TireAge'] = df.groupby(['Driver', 'Stint'])['LapNumber'].rank(method='first')

    # 2. Fix Fuel Load (Proxy Variable)
    if 'FuelLoad' not in df.columns:
        print("Created 'FuelLoad' proxy (TotalLaps - CurrentLap)")
        # Assuming standard race is approx 50-60 laps. 
        # Fuel burns linearly, so Inverse of LapNumber acts as Fuel Weight.
        df['FuelLoad'] = 57 - df['LapNumber'] 
        # Clip negative values just in case
        df['FuelLoad'] = df['FuelLoad'].apply(lambda x: max(x, 0))
        
    # 3. Ensure numeric types
    numeric_cols = ['LapNumber', 'TireAge', 'FuelLoad', 'LapTimeSeconds']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df.dropna(subset=numeric_cols)

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('./data/processed/processed_laps.csv')
        print(f"Loaded {len(df)} laps.")
    except FileNotFoundError:
        print("Error: csv file not found. Please run data_collection.py first.")
        return

    # 2. Engineer Features (Fixes the KeyError)
    df = engineer_features(df)
    
    trainer = F1ModelTrainer()

    # ==========================================
    # A. XGBoost Training (Standard ML)
    # ==========================================
    print("\n--- Preparing XGBoost Data ---")
    from data_preprocessing import F1DataPreprocessor
    try:
        preprocessor = F1DataPreprocessor()
        X, y = preprocessor.prepare_training_data(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xgb_model = trainer.train_xgboost(X_train, y_train)
        trainer.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        trainer.save_model(xgb_model, "xgboost_model")
    except Exception as e:
        print(f"Skipping XGBoost (Check preprocessor): {e}")

    # ==========================================
    # B. LSTM Training (Deep Learning)
    # ==========================================
    print("\n--- Preparing LSTM Data ---")
    
    # Sort to ensure time sequence is correct
    df_sorted = df.sort_values(['Driver', 'LapNumber'])
    
    # Define Features explicitly (Now guaranteed to exist)
    features = ['LapNumber', 'TireAge', 'FuelLoad'] 
    target = 'LapTimeSeconds'
    look_back = 5
    
    try:
        X_lstm, y_lstm, scaler, n_features = trainer.prepare_lstm_data(df_sorted, features, target, look_back)
        
        # Split LSTM
        train_size = int(len(X_lstm) * 0.8)
        X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
        y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]
        
        # Train
        lstm_model = trainer.train_lstm(X_train_lstm, y_train_lstm, look_back, n_features)
        
        # Save
        trainer.save_model(lstm_model, "lstm_deep_learning_model")
        
        # Save Scaler (Important for frontend inference)
        with open('./data/models/lstm_scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
    except KeyError as e:
        print(f"CRITICAL ERROR: {e}")
        print("Available columns in dataframe:", df.columns.tolist())

    print("\n=== All Models Trained Successfully ===")

if __name__ == "__main__":
    main()