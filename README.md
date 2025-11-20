# F1 Pit Stop Strategy Optimizer

An end-to-end machine learning system that predicts and optimizes Formula 1 pit stop strategies using real race data.

## 🏎️ Project Overview

This system analyzes historical F1 race data to predict optimal pit stop timing and tire strategies. It combines machine learning models with race simulation to recommend strategies that minimize total race time.

### Key Features

- **Data Collection**: Automated fetching from FastF1 and Ergast APIs
- **Feature Engineering**: 15+ engineered features including tire degradation, stint analysis, and race progression
- **ML Models**: Random Forest, XGBoost, and Gradient Boosting for lap time prediction
- **Strategy Optimization**: Simulation-based optimization with genetic algorithm support
- **Visualizations**: Interactive dashboards and comprehensive race analysis plots

## 📊 System Architecture

```
Data Collection → Preprocessing → Model Training → Optimization → Visualization
     ↓                ↓                ↓               ↓              ↓
  FastF1/          Feature         RF/XGB/GB      Simulate      Charts/
  Ergast          Engineering      Models         Strategies    Dashboard
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd f1-pitstop-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Edit `config/config.yaml` to customize:
- Years and races to analyze
- Model hyperparameters
- Optimization settings
- Visualization preferences

### Running the Full Pipeline

```bash
# Run complete pipeline
python main.py

# Skip data collection (use existing data)
python main.py --skip-collection

# Skip training (use existing model)
python main.py --skip-training
```

### Running Individual Modules

```bash
# Data collection only
python src/data_collection.py

# Model training only
python src/model_training.py

# Optimization only
python src/optimization_engine.py

# Visualization only
python src/visualization.py
```

## 📁 Project Structure

```
f1-pitstop-optimizer/
│
├── data/
│   ├── raw/                    # Raw API data
│   ├── processed/              # Cleaned & feature-engineered data
│   └── models/                 # Trained ML models
│
├── src/
│   ├── data_collection.py      # FastF1 & Ergast integration
│   ├── data_preprocessing.py   # Cleaning & feature engineering
│   ├── model_training.py       # ML model training
│   ├── optimization_engine.py  # Strategy optimization
│   └── visualization.py        # Plotting & dashboards
│
├── config/
│   └── config.yaml             # Configuration parameters
│
├── output/
│   └── visualizations/         # Generated plots & reports
│
├── main.py                     # Main execution script
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🔧 Technical Details

### Data Sources

**FastF1 API**
- Lap-by-lap timing data
- Tire compounds and stint information
- Track conditions and weather
- Telemetry data

**Ergast API**
- Race metadata and results
- Driver and constructor information
- Historical standings

### Feature Engineering

The system creates 15+ features including:

- **Tire Features**: Degradation rate, compound type, tire age
- **Stint Features**: Lap in stint, stint length, stint number
- **Pace Features**: Average pace, pace loss from best lap
- **Race Features**: Fuel load proxy, race progress, pit window indicators
- **Position Features**: Track position changes

### ML Models

**Random Forest Regressor**
- Ensemble of decision trees
- Handles non-linear relationships
- Feature importance analysis

**XGBoost**
- Gradient boosting framework
- Optimized performance
- Handles missing data well

**Gradient Boosting**
- Sequential ensemble method
- Strong predictive accuracy

### Optimization Strategies

**Brute Force Simulation**
- Evaluates multiple strategy combinations
- Simulates complete race with each strategy
- Selects minimum total time

**Genetic Algorithm** (Optional)
- Population-based optimization
- Crossover and mutation operations
- Converges to optimal solution

## 📈 Output

The system generates:

1. **Processed Data Files**
   - `processed_laps.csv`: Full featured dataset
   - `strategy_comparison.csv`: Top 10 strategies
   - `feature_importance.csv`: Model feature rankings

2. **Visualizations**
   - Tire degradation curves
   - Lap time evolution plots
   - Strategy comparison charts
   - Pit window analysis
   - Feature importance plots
   - Interactive HTML dashboard

3. **Reports**
   - `strategy_report.txt`: Optimal strategy recommendation
   - Stint breakdown
   - Alternative strategies

## 🎯 Example Output

```
╔══════════════════════════════════════════════════════════════╗
║          F1 PIT STOP STRATEGY OPTIMIZATION REPORT           ║
╚══════════════════════════════════════════════════════════════╝

OPTIMAL STRATEGY
─────────────────────────────────────────────────────────────
Total Race Time:     4845.23 seconds
Number of Pit Stops: 2
Pit Laps:            18, 38
Tire Sequence:       SOFT → MEDIUM → HARD

STINT BREAKDOWN
─────────────────────────────────────────────────────────────
Stint 1: Laps 1-18 (18 laps) on SOFT tires
Stint 2: Laps 19-38 (20 laps) on MEDIUM tires
Stint 3: Laps 39-58 (20 laps) on HARD tires
```

## 🔬 Model Performance

Typical performance metrics:
- **MAE**: 0.5-1.0 seconds per lap
- **RMSE**: 0.8-1.5 seconds
- **R²**: 0.85-0.95

## 🛠️ Customization

### Adding New Features

Edit `src/data_preprocessing.py`:

```python
def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Add your custom features here
    df['MyFeature'] = ...
    return df
```

### Custom Optimization Logic

Edit `src/optimization_engine.py`:

```python
def custom_optimization(self, race_params: Dict):
    # Implement your optimization logic
    pass
```

### Adding Visualizations

Edit `src/visualization.py`:

```python
def plot_custom_analysis(self, df: pd.DataFrame):
    # Create custom plots
    pass
```

## 📝 Future Enhancements

- [ ] Real-time strategy updates during races
- [ ] Weather impact modeling
- [ ] Safety car probability predictions
- [ ] Multi-driver strategic interactions
- [ ] Deep learning models (LSTM for time series)
- [ ] API endpoint for strategy queries
- [ ] Web dashboard with Streamlit/Dash

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is for educational purposes. F1 data is accessed through official APIs with appropriate attribution.

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

- **FastF1**: For providing comprehensive F1 data API
- **Ergast**: For historical F1 database
- **scikit-learn**: For ML framework
- **XGBoost**: For gradient boosting implementation

---

**Built with ❤️ for F1 fans and data scientists**