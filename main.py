"""
Main Execution Script
Runs the complete F1 pit stop optimization pipeline
"""

import os
import yaml
import argparse
from datetime import datetime

# Import project modules
from src.data_collection import F1DataCollector
from src.data_preprocessing import F1DataPreprocessor
from src.model_training import F1ModelTrainer, PitStopPredictor
from src.optimization_engine import F1StrategyOptimizer, RaceStrategy
from src.visualization import F1Visualizer


def load_config(config_path: str = './config/config.yaml') -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Create necessary directories"""
    dirs = [
        config['paths']['raw_data'],
        config['paths']['processed_data'],
        config['paths']['models'],
        config['paths']['visualizations']
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def run_data_collection(config: dict):
    """Step 1: Collect data from APIs"""
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    
    collector = F1DataCollector(
        cache_dir=os.path.join(config['paths']['raw_data'], 'cache')
    )
    
    # Collect data
    years = config['data_collection']['years']
    races = config['data_collection']['races']
    
    data = collector.collect_all_data(years=years, races=races)
    
    # Save data
    collector.save_data(data['laps'], 'laps_data.csv', config['paths']['raw_data'])
    collector.save_data(data['ergast'], 'ergast_data.csv', config['paths']['raw_data'])
    
    print("\n✓ Data collection complete")
    return data


def run_preprocessing(config: dict):
    """Step 2: Preprocess and engineer features"""
    print("\n" + "="*70)
    print("STEP 2: DATA PREPROCESSING")
    print("="*70)
    
    # Load raw data
    import pandas as pd
    laps_path = os.path.join(config['paths']['raw_data'], 'laps_data.csv')
    laps_df = pd.read_csv(laps_path)
    
    # Initialize preprocessor
    preprocessor = F1DataPreprocessor()
    
    # Clean data
    cleaned_df = preprocessor.clean_lap_data(laps_df)
    
    # Engineer features
    featured_df = preprocessor.engineer_features(cleaned_df)
    pitstop_df = preprocessor.create_pitstop_features(featured_df)
    
    # Save processed data
    output_path = os.path.join(config['paths']['processed_data'], 'processed_laps.csv')
    pitstop_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Preprocessing complete: {len(pitstop_df)} laps processed")
    return pitstop_df


def run_model_training(config: dict):
    """Step 3: Train ML models"""
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load processed data
    processed_path = os.path.join(config['paths']['processed_data'], 'processed_laps.csv')
    df = pd.read_csv(processed_path)
    
    # Prepare training data
    preprocessor = F1DataPreprocessor()
    X, y = preprocessor.prepare_training_data(df)
    
    # Split data
    test_size = config['models']['test_size']
    random_state = config['models']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train models
    trainer = F1ModelTrainer(random_state=random_state)
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Save best model
    best_model_name = results['best_model']
    best_model = results[best_model_name]['model']
    trainer.save_model(best_model, 'best_model', config['paths']['models'])
    
    # Get feature importance
    importance_df = trainer.get_feature_importance(best_model, X.columns.tolist())
    if importance_df is not None:
        importance_path = os.path.join(config['paths']['processed_data'], 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
    
    print(f"\n✓ Model training complete: {best_model_name} saved")
    return best_model, results


def run_optimization(config: dict, model):
    """Step 4: Optimize pit stop strategy"""
    print("\n" + "="*70)
    print("STEP 4: STRATEGY OPTIMIZATION")
    print("="*70)
    
    # Initialize optimizer
    pit_loss = config['optimization']['pit_time_loss']
    optimizer = F1StrategyOptimizer(model, pit_loss_seconds=pit_loss)
    
    # Define race parameters (example: Australian GP)
    race_params = {
        'total_laps': 58,
        'base_lap_time': 82.0,
        'fuel_effect': 0.05,
        'max_stops': config['optimization']['max_pit_stops'],
        'min_stint': config['optimization']['min_stint_length']
    }
    
    # Optimize strategy
    n_strategies = config['optimization']['simulation_strategies']
    strategies = optimizer.optimize_strategy(race_params, n_strategies=n_strategies)
    
    # Compare top strategies
    comparison_df = optimizer.compare_strategies(strategies, top_n=10)
    
    # Save results
    comparison_path = os.path.join(config['paths']['processed_data'], 'strategy_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"\n✓ Strategy optimization complete: {len(strategies)} strategies evaluated")
    return strategies, comparison_df


def run_visualization(config: dict):
    """Step 5: Create visualizations"""
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    import pandas as pd
    
    # Load data
    processed_path = os.path.join(config['paths']['processed_data'], 'processed_laps.csv')
    df = pd.read_csv(processed_path)
    
    comparison_path = os.path.join(config['paths']['processed_data'], 'strategy_comparison.csv')
    strategies_df = pd.read_csv(comparison_path)
    
    importance_path = os.path.join(config['paths']['processed_data'], 'feature_importance.csv')
    
    # Initialize visualizer
    viz = F1Visualizer(
        style=config['visualization']['style'],
        figsize=tuple(config['visualization']['figure_size'])
    )
    
    # Create visualizations directory
    viz_dir = config['paths']['visualizations']
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    viz.plot_tire_degradation(df, save_path=os.path.join(viz_dir, 'tire_degradation.png'))
    viz.plot_lap_time_evolution(df, save_path=os.path.join(viz_dir, 'lap_times.png'))
    viz.plot_strategy_comparison(strategies_df, save_path=os.path.join(viz_dir, 'strategy_comparison.png'))
    viz.plot_pit_window_analysis(df, save_path=os.path.join(viz_dir, 'pit_windows.png'))
    
    # Feature importance
    if os.path.exists(importance_path):
        importance_df = pd.read_csv(importance_path)
        viz.plot_feature_importance(importance_df, save_path=os.path.join(viz_dir, 'feature_importance.png'))
    
    # Interactive dashboard
    viz.create_interactive_strategy_dashboard(
        df, strategies_df, 
        save_path=os.path.join(viz_dir, 'dashboard.html')
    )
    
    print(f"\n✓ Visualization complete: saved to {viz_dir}")


def generate_final_report(config: dict, strategies: list, comparison_df):
    """Generate final strategy report"""
    print("\n" + "="*70)
    print("FINAL REPORT")
    print("="*70)
    
    viz = F1Visualizer()
    
    best_strategy = strategies[0]
    
    report_path = os.path.join(config['paths']['visualizations'], 'strategy_report.txt')
    viz.generate_strategy_report(best_strategy, comparison_df, save_path=report_path)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='F1 Pit Stop Strategy Optimizer')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection (use existing data)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training (use existing model)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print("\n" + "="*70)
    print("F1 PIT STOP STRATEGY OPTIMIZATION SYSTEM")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    setup_directories(config)
    
    try:
        # Step 1: Data Collection
        if not args.skip_collection:
            run_data_collection(config)
        else:
            print("\n⊳ Skipping data collection (using existing data)")
        
        # Step 2: Preprocessing
        run_preprocessing(config)
        
        # Step 3: Model Training
        if not args.skip_training:
            model, results = run_model_training(config)
        else:
            print("\n⊳ Skipping model training (loading existing model)")
            trainer = F1ModelTrainer()
            model = trainer.load_model('best_model', config['paths']['models'])
        
        # Step 4: Strategy Optimization
        strategies, comparison_df = run_optimization(config, model)
        
        # Step 5: Visualization
        run_visualization(config)
        
        # Final Report
        generate_final_report(config, strategies, comparison_df)
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nResults saved to: {config['paths']['visualizations']}")
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())