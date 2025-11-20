"""
Model Training Module
Trains and evaluates ML models for lap time and pit stop prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import os
from typing import Dict, Tuple


class F1ModelTrainer:
    """Trains ML models for F1 predictions"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.metrics = {}
        
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           n_estimators: int = 100, max_depth: int = 20) -> RandomForestRegressor:
        """Train Random Forest model"""
        print("\n=== Training Random Forest ===")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        print(f"Model trained with {n_estimators} trees")
        return model
    
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
    
    def train_gradient_boosting(self, X_train: pd.DataFrame, y_train: pd.Series,
                               n_estimators: int = 100, max_depth: int = 5,
                               learning_rate: float = 0.1) -> GradientBoostingRegressor:
        """Train Gradient Boosting model"""
        print("\n=== Training Gradient Boosting ===")
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            random_state=self.random_state
        )
        
        model.fit(X_train, y_train)
        
        print(f"Model trained with {n_estimators} estimators")
        return model
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n=== Evaluating {model_name} ===")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                            cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        print("\n=== Cross-Validation ===")
        
        # Negative MAE for cross_val_score (it maximizes)
        cv_scores = cross_val_score(model, X, y, cv=cv, 
                                    scoring='neg_mean_absolute_error', n_jobs=-1)
        
        mae_scores = -cv_scores
        
        print(f"CV MAE scores: {mae_scores}")
        print(f"Mean CV MAE: {mae_scores.mean():.4f} (+/- {mae_scores.std():.4f})")
        
        return {
            'cv_mae_mean': mae_scores.mean(),
            'cv_mae_std': mae_scores.std()
        }
    
    def get_feature_importance(self, model, feature_names: list, top_n: int = 10) -> pd.DataFrame:
        """Extract feature importance"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n=== Top {top_n} Most Important Features ===")
            print(importance_df.head(top_n))
            
            return importance_df
        else:
            print("Model does not support feature importance")
            return None
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Train all models and return results
        
        Returns:
            Dictionary with models and metrics
        """
        results = {}
        
        # Train Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
        results['random_forest'] = {'model': rf_model, 'metrics': rf_metrics}
        
        # Train XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        results['xgboost'] = {'model': xgb_model, 'metrics': xgb_metrics}
        
        # Train Gradient Boosting
        gb_model = self.train_gradient_boosting(X_train, y_train)
        gb_metrics = self.evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")
        results['gradient_boosting'] = {'model': gb_model, 'metrics': gb_metrics}
        
        # Find best model
        best_model_name = min(results.keys(), 
                             key=lambda k: results[k]['metrics']['mae'])
        print(f"\n=== Best Model: {best_model_name} ===")
        
        results['best_model'] = best_model_name
        
        return results
    
    def save_model(self, model, model_name: str, output_dir: str = './data/models'):
        """Save trained model"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{model_name}.pkl')
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, models_dir: str = './data/models'):
        """Load trained model"""
        filepath = os.path.join(models_dir, f'{model_name}.pkl')
        
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
        print(f"Model loaded from {filepath}")
        return model


class PitStopPredictor:
    """Specialized predictor for optimal pit stop timing"""
    
    def __init__(self, lap_time_model, degradation_model=None):
        self.lap_time_model = lap_time_model
        self.degradation_model = degradation_model
        
    def predict_optimal_pit_lap(self, race_data: pd.DataFrame, 
                               total_laps: int, current_lap: int) -> int:
        """
        Predict optimal pit stop lap
        
        Args:
            race_data: Current race conditions
            total_laps: Total race laps
            current_lap: Current lap number
            
        Returns:
            Recommended pit lap
        """
        # Simulate remaining laps
        projected_times = []
        
        for pit_lap in range(current_lap + 5, total_laps - 5):
            # Calculate expected race time with pit at this lap
            total_time = self._simulate_pit_strategy(race_data, pit_lap, total_laps)
            projected_times.append((pit_lap, total_time))
        
        # Find lap with minimum total time
        optimal_lap = min(projected_times, key=lambda x: x[1])[0]
        
        return optimal_lap
    
    def _simulate_pit_strategy(self, race_data: pd.DataFrame, 
                               pit_lap: int, total_laps: int,
                               pit_loss: float = 25.0) -> float:
        """Simulate total race time with pit at specified lap"""
        # Simplified simulation - in reality, use lap time model
        # This is a placeholder for the full simulation logic
        
        stint1_laps = pit_lap
        stint2_laps = total_laps - pit_lap
        
        # Estimate stint times (simplified)
        stint1_time = stint1_laps * 90  # Base lap time
        stint2_time = stint2_laps * 88  # Fresher tires
        
        total_time = stint1_time + stint2_time + pit_loss
        
        return total_time


def main():
    """Example usage"""
    # Load processed data
    df = pd.read_csv('./data/processed/processed_laps.csv')
    
    # Prepare features
    from data_preprocessing import F1DataPreprocessor
    
    preprocessor = F1DataPreprocessor()
    X, y = preprocessor.prepare_training_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    trainer = F1ModelTrainer()
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    trainer.save_model(best_model, best_model_name)
    
    # Feature importance
    trainer.get_feature_importance(best_model, X.columns.tolist())
    
    print("\n=== Model Training Complete ===")


if __name__ == "__main__":
    main()