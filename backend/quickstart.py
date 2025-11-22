"""
Quick Start Script for F1 Pit Stop Optimizer
This version handles common errors gracefully
"""

import os
import sys
import yaml
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import F1DataCollector
from data_preprocessing import F1DataPreprocessor
from model_training import F1ModelTrainer
from optimization_engine import F1StrategyOptimizer
from visualization import F1Visualizer


def setup_directories():
    """Create necessary directories"""
    dirs = [
        'data/raw',
        'data/processed',
        'data/models',
        'output/visualizations'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Directories created")


def collect_data():
    """Step 1: Collect race data"""
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    
    collector = F1DataCollector(cache_dir='./data/raw/cache')
    
    # Only collect races 1-13 for 2024 (races with available data)
    print("Collecting 2024 season data (races 1-13)...")
    
    try:
        data = collector.collect_all_data(
            years=[2024],
            races=list(range(1, 14))  # Races 1-13
        )
        
        # Save data
        collector.save_data(data['laps'], 'laps_data.csv')
        collector.save_data(data['ergast'], 'ergast_data.csv')
        
        print(f"\n✓ Collected {len(data['laps'])} laps from {data['laps']['RaceNumber'].nunique()} races")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during data collection: {e}")
        return False


def preprocess_data():
    """Step 2: Preprocess and engineer features"""
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    try:
        # Load raw data
        laps_df = pd.read_csv('./data/raw/laps_data.csv')
        
        # Initialize preprocessor
        preprocessor = F1DataPreprocessor()
        
        # Clean and engineer features
        print("Cleaning data...")
        cleaned_df = preprocessor.clean_lap_data(laps_df)
        
        print("Engineering features...")
        featured_df = preprocessor.engineer_features(cleaned_df)
        pitstop_df = preprocessor.create_pitstop_features(featured_df)
        
        # Save processed data
        output_path = './data/processed/processed_laps.csv'
        pitstop_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Processed {len(pitstop_df)} laps")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_models():
    """Step 3: Train ML models"""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Load processed data
        df = pd.read_csv('./data/processed/processed_laps.csv')
        
        # Prepare training data
        preprocessor = F1DataPreprocessor()
        X, y = preprocessor.prepare_training_data(df)
        
        print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        trainer = F1ModelTrainer(random_state=42)
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save best model
        best_model_name = results['best_model']
        best_model = results[best_model_name]['model']
        trainer.save_model(best_model, 'best_model')
        
        # Save feature importance
        importance_df = trainer.get_feature_importance(best_model, X.columns.tolist())
        if importance_df is not None:
            importance_df.to_csv('./data/processed/feature_importance.csv', index=False)
        
        print(f"\n✓ Best model: {best_model_name}")
        print(f"   MAE: {results[best_model_name]['metrics']['mae']:.4f}")
        print(f"   R²: {results[best_model_name]['metrics']['r2']:.4f}")
        
        return best_model
        
    except Exception as e:
        print(f"\n✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return None


def optimize_strategy(model):
    """Step 4: Optimize pit stop strategy"""
    print("\n" + "="*70)
    print("STEP 4: STRATEGY OPTIMIZATION")
    print("="*70)
    
    try:
        # Initialize optimizer
        optimizer = F1StrategyOptimizer(model, pit_loss_seconds=25.0)
        
        # Define race parameters (average F1 race)
        race_params = {
            'total_laps': 58,
            'base_lap_time': 85.0,
            'fuel_effect': 0.05,
            'max_stops': 2,
            'min_stint': 10
        }
        
        print("Simulating strategies...")
        strategies = optimizer.optimize_strategy(race_params, n_strategies=100)
        
        # Compare top strategies
        comparison_df = optimizer.compare_strategies(strategies, top_n=10)
        comparison_df.to_csv('./data/processed/strategy_comparison.csv', index=False)
        
        print(f"\n✓ Evaluated {len(strategies)} strategies")
        print(f"\nBest Strategy:")
        print(f"  Total Time: {strategies[0].total_time:.2f}s")
        print(f"  Pit Stops: {strategies[0].pit_laps}")
        print(f"  Tires: {' → '.join(strategies[0].tire_sequence)}")
        
        return strategies
        
    except Exception as e:
        print(f"\n✗ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_visualizations():
    """Step 5: Create visualizations"""
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    try:
        # Load data
        df = pd.read_csv('./data/processed/processed_laps.csv')
        strategies_df = pd.read_csv('./data/processed/strategy_comparison.csv')
        
        # Initialize visualizer
        viz = F1Visualizer()
        
        print("Generating plots...")
        
        # Create visualizations
        viz.plot_tire_degradation(df, save_path='./output/visualizations/tire_degradation.png')
        viz.plot_lap_time_evolution(df, save_path='./output/visualizations/lap_times.png')
        viz.plot_strategy_comparison(strategies_df, save_path='./output/visualizations/strategy_comparison.png')
        viz.plot_pit_window_analysis(df, save_path='./output/visualizations/pit_windows.png')
        
        # Feature importance
        if os.path.exists('./data/processed/feature_importance.csv'):
            importance_df = pd.read_csv('./data/processed/feature_importance.csv')
            viz.plot_feature_importance(importance_df, save_path='./output/visualizations/feature_importance.png')
        
        # Interactive dashboard
        viz.create_interactive_strategy_dashboard(
            df, strategies_df,
            save_path='./output/visualizations/dashboard.html'
        )
        
        print("\n✓ Visualizations created in output/visualizations/")
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete pipeline"""
    print("\n" + "="*70)
    print("F1 PIT STOP STRATEGY OPTIMIZER - QUICK START")
    print("="*70)
    
    # Setup
    setup_directories()
    
    # Step 1: Data Collection
    if not collect_data():
        print("\n⚠ Data collection failed. Check if:")
        print("  - You have internet connection")
        print("  - FastF1 API is accessible")
        print("  - The race numbers are correct")
        return
    
    # Step 2: Preprocessing
    if not preprocess_data():
        print("\n⚠ Preprocessing failed. Check data quality.")
        return
    
    # Step 3: Model Training
    model = train_models()
    if model is None:
        print("\n⚠ Model training failed.")
        return
    
    # Step 4: Strategy Optimization
    strategies = optimize_strategy(model)
    if strategies is None:
        print("\n⚠ Optimization failed.")
        return
    
    # Step 5: Visualization
    create_visualizations()
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults:")
    print(f"  - Processed data: data/processed/")
    print(f"  - Trained models: data/models/")
    print(f"  - Visualizations: output/visualizations/")
    print(f"  - Dashboard: output/visualizations/dashboard.html")


if __name__ == "__main__":
    main()