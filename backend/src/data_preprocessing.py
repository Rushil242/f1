"""
Data Preprocessing Module
Cleans data and engineers features for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class F1DataPreprocessor:
    """Preprocesses F1 data and engineers features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def clean_lap_data(self, laps_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean lap data: handle missing values, outliers
        
        Args:
            laps_df: Raw laps DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        print("\n=== Cleaning Lap Data ===")
        
        df = laps_df.copy()
        initial_rows = len(df)
        
        # Convert time columns to seconds
        if 'LapTime' in df.columns:
            df['LapTimeSeconds'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
        
        if 'Sector1Time' in df.columns:
            df['Sector1TimeSeconds'] = pd.to_timedelta(df['Sector1Time']).dt.total_seconds()
        if 'Sector2Time' in df.columns:
            df['Sector2TimeSeconds'] = pd.to_timedelta(df['Sector2Time']).dt.total_seconds()
        if 'Sector3Time' in df.columns:
            df['Sector3TimeSeconds'] = pd.to_timedelta(df['Sector3Time']).dt.total_seconds()
            
        # Remove invalid laps (too slow or too fast)
        if 'LapTimeSeconds' in df.columns:
            df = df[df['LapTimeSeconds'] > 0]
            # Remove laps > 2 minutes (likely invalid)
            df = df[df['LapTimeSeconds'] < 120]
            
        # Remove laps with pit stops for pure pace analysis
        if 'PitOutTime' in df.columns and 'PitInTime' in df.columns:
            df['IsPitLap'] = (~df['PitOutTime'].isna()) | (~df['PitInTime'].isna())
        else:
            df['IsPitLap'] = False
            
        # Fill missing tire compounds
        if 'Compound' in df.columns:
            df['Compound'] = df['Compound'].fillna('UNKNOWN')
            
        print(f"Removed {initial_rows - len(df)} invalid laps")
        print(f"Remaining laps: {len(df)}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for ML models
        
        Features:
        - Tire degradation rate
        - Stint characteristics
        - Lap pace trends
        - Track position effects
        - Weather impact
        """
        print("\n=== Engineering Features ===")
        
        df = df.copy()
        
        # 1. Stint-level features
        df = df.sort_values(['Year', 'RaceNumber', 'Driver', 'LapNumber'])
        
        # Create stint number
        df['StintNumber'] = df.groupby(['Year', 'RaceNumber', 'Driver'])['IsPitLap'].cumsum()
        
        # Lap in stint
        df['LapInStint'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber']).cumcount() + 1
        
        # 2. Tire degradation rate
        # Calculate rolling average lap time change
        df['LapTimeDelta'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'])['LapTimeSeconds'].diff()
        
        # Degradation rate per lap (seconds lost per lap)
        print("   → Calculating tire degradation rate...")
        grouped = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'], group_keys=False)
        df['TireDegradationRate'] = grouped['LapTimeDelta'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
        
        # 3. Pace metrics
        # Average pace in stint
        df['AvgStintPace'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'])['LapTimeSeconds'].transform('mean')
        
        # Best lap in stint
        df['BestStintLap'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'])['LapTimeSeconds'].transform('min')
        
        # Pace loss from best lap
        df['PaceLossFromBest'] = df['LapTimeSeconds'] - df['BestStintLap']
        
        # 4. Track position features
        if 'Position' in df.columns:
            df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            df['PositionChange'] = df.groupby(['Year', 'RaceNumber', 'Driver'])['Position'].diff()
        
        # 5. Compound encoding
        if 'Compound' in df.columns:
            # Create binary features for each compound
            df['IsSoft'] = (df['Compound'] == 'SOFT').astype(int)
            df['IsMedium'] = (df['Compound'] == 'MEDIUM').astype(int)
            df['IsHard'] = (df['Compound'] == 'HARD').astype(int)
            
        # 6. Fuel load proxy (lap number)
        df['FuelLoadProxy'] = df.groupby(['Year', 'RaceNumber'])['LapNumber'].transform('max') - df['LapNumber']
        
        # 7. Race progress
        df['RaceProgress'] = df['LapNumber'] / df.groupby(['Year', 'RaceNumber'])['LapNumber'].transform('max')
        
        # 8. Weather features (if available)
        if 'TrackTemp' in df.columns:
            df['TrackTemp'] = pd.to_numeric(df['TrackTemp'], errors='coerce')
        if 'AirTemp' in df.columns:
            df['AirTemp'] = pd.to_numeric(df['AirTemp'], errors='coerce')
            
        print(f"Engineered {len([c for c in df.columns if c not in df.columns])} new features")
        
        return df
    
    def create_pitstop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specifically for pit stop prediction
        """
        print("\n=== Creating Pit Stop Features ===")
        
        df = df.copy()
        
        # Stint length
        df['StintLength'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'])['LapNumber'].transform('count')
        
        # Time since last pit
        df['LapsSinceLastPit'] = df['LapInStint']
        
        # Estimated tire life remaining (normalized)
        df['TireLifeRemaining'] = 1 - (df['LapInStint'] / df['StintLength'])
        
        # Cumulative degradation
        df['CumulativeDegradation'] = df.groupby(['Year', 'RaceNumber', 'Driver', 'StintNumber'])['TireDegradationRate'].cumsum()
        
        # Pit window indicator
        total_laps = df.groupby(['Year', 'RaceNumber'])['LapNumber'].transform('max')
        df['InPitWindow'] = ((df['LapNumber'] > total_laps * 0.2) & 
                             (df['LapNumber'] < total_laps * 0.8)).astype(int)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, target_col: str = 'LapTimeSeconds') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for ML training
        
        Args:
            df: Preprocessed DataFrame
            target_col: Target variable name
            
        Returns:
            X (features), y (target)
        """
        print("\n=== Preparing Training Data ===")
        
        # Select feature columns
        feature_cols = [
            'LapNumber', 'LapInStint', 'StintNumber',
            'TireDegradationRate', 'AvgStintPace', 'PaceLossFromBest',
            'IsSoft', 'IsMedium', 'IsHard',
            'FuelLoadProxy', 'RaceProgress',
            'StintLength', 'LapsSinceLastPit', 'TireLifeRemaining',
            'CumulativeDegradation', 'InPitWindow'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Remove rows with missing target
        df_clean = df[df[target_col].notna()].copy()
        
        # Remove rows with any missing features
        df_clean = df_clean.dropna(subset=available_features)
        
        X = df_clean[available_features]
        y = df_clean[target_col]
        
        print(f"Training data shape: {X.shape}")
        print(f"Features used: {len(available_features)}")
        print(f"Samples: {len(X)}")
        
        return X, y
    
    def normalize_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Normalize numerical features"""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
            
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)


def main():
    """Example usage"""
    # Load raw data
    laps_df = pd.read_csv('./data/raw/laps_data.csv')
    
    # Initialize preprocessor
    preprocessor = F1DataPreprocessor()
    
    # Clean and engineer features
    cleaned_df = preprocessor.clean_lap_data(laps_df)
    featured_df = preprocessor.engineer_features(cleaned_df)
    pitstop_df = preprocessor.create_pitstop_features(featured_df)
    
    # Prepare training data
    X, y = preprocessor.prepare_training_data(pitstop_df)
    
    # Save processed data
    pitstop_df.to_csv('./data/processed/processed_laps.csv', index=False)
    
    print("\n=== Preprocessing Complete ===")
    print(f"Processed data saved with {len(pitstop_df)} laps")


if __name__ == "__main__":
    main()