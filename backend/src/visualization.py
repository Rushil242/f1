"""
Visualization Module
Creates charts and dashboards for F1 strategy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from typing import List, Dict


class F1Visualizer:
    """Creates visualizations for F1 pit stop analysis"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: tuple = (12, 8)):
        """Initialize visualizer with style settings"""
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        sns.set_palette("husl")
        
        # F1 tire colors
        self.tire_colors = {
            'SOFT': '#FF0000',      # Red
            'MEDIUM': '#FFD700',    # Yellow
            'HARD': '#FFFFFF',      # White
            'INTERMEDIATE': '#00FF00',  # Green
            'WET': '#0000FF'        # Blue
        }
        
    def plot_tire_degradation(self, df: pd.DataFrame, driver: str = None, 
                             save_path: str = None):
        """
        Plot tire degradation curves
        
        Args:
            df: DataFrame with lap data
            driver: Specific driver to plot (None = all)
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Filter data
        if driver:
            plot_df = df[df['Driver'] == driver].copy()
            title = f'Tire Degradation - {driver}'
        else:
            plot_df = df.copy()
            title = 'Tire Degradation - All Drivers'
        
        # Plot by compound
        for compound in plot_df['Compound'].unique():
            compound_df = plot_df[plot_df['Compound'] == compound]
            
            # Group by stint and calculate average degradation
            degradation = compound_df.groupby('LapInStint')['TireDegradationRate'].mean()
            
            ax.plot(degradation.index, degradation.values, 
                   marker='o', label=compound, 
                   color=self.tire_colors.get(compound, 'gray'),
                   linewidth=2, markersize=4)
        
        ax.set_xlabel('Lap in Stint', fontsize=12)
        ax.set_ylabel('Degradation Rate (s/lap)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def plot_lap_time_evolution(self, df: pd.DataFrame, drivers: List[str] = None,
                               save_path: str = None):
        """
        Plot lap time evolution during race
        
        Args:
            df: DataFrame with lap data
            drivers: List of drivers to plot
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if drivers is None:
            drivers = df['Driver'].unique()[:5]  # Top 5 drivers
        
        for driver in drivers:
            driver_df = df[df['Driver'] == driver].sort_values('LapNumber')
            
            ax.plot(driver_df['LapNumber'], driver_df['LapTimeSeconds'],
                   marker='o', label=driver, linewidth=2, markersize=3, alpha=0.7)
        
        ax.set_xlabel('Lap Number', fontsize=12)
        ax.set_ylabel('Lap Time (seconds)', fontsize=12)
        ax.set_title('Lap Time Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def plot_strategy_comparison(self, strategies_df: pd.DataFrame, save_path: str = None):
        """
        Plot comparison of different strategies
        
        Args:
            strategies_df: DataFrame with strategy comparison
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Total time comparison
        ax1.barh(strategies_df['rank'], strategies_df['total_time'], 
                color='steelblue', alpha=0.7)
        ax1.set_xlabel('Total Race Time (seconds)', fontsize=12)
        ax1.set_ylabel('Strategy Rank', fontsize=12)
        ax1.set_title('Total Race Time by Strategy', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Time delta from best
        ax2.bar(strategies_df['rank'], strategies_df['time_delta'], 
               color='coral', alpha=0.7)
        ax2.set_xlabel('Strategy Rank', fontsize=12)
        ax2.set_ylabel('Time Delta from Best (seconds)', fontsize=12)
        ax2.set_title('Time Loss vs Optimal Strategy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def plot_pit_window_analysis(self, df: pd.DataFrame, save_path: str = None):
        """
        Analyze and plot optimal pit windows
        
        Args:
            df: DataFrame with lap data
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate average lap time by lap number
        avg_times = df.groupby('LapNumber')['LapTimeSeconds'].agg(['mean', 'std'])
        
        ax.plot(avg_times.index, avg_times['mean'], 
               color='darkblue', linewidth=2, label='Average Lap Time')
        ax.fill_between(avg_times.index, 
                       avg_times['mean'] - avg_times['std'],
                       avg_times['mean'] + avg_times['std'],
                       alpha=0.3, color='lightblue')
        
        # Mark common pit windows
        total_laps = df['LapNumber'].max()
        early_window = (int(total_laps * 0.2), int(total_laps * 0.35))
        mid_window = (int(total_laps * 0.45), int(total_laps * 0.60))
        
        ax.axvspan(early_window[0], early_window[1], alpha=0.2, color='green', 
                  label='Early Pit Window')
        ax.axvspan(mid_window[0], mid_window[1], alpha=0.2, color='orange', 
                  label='Mid Pit Window')
        
        ax.set_xlabel('Lap Number', fontsize=12)
        ax.set_ylabel('Lap Time (seconds)', fontsize=12)
        ax.set_title('Optimal Pit Windows Analysis', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15,
                               save_path: str = None):
        """
        Plot feature importance from model
        
        Args:
            importance_df: DataFrame with feature importance
            top_n: Number of top features to show
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        top_features = importance_df.head(top_n).sort_values('importance')
        
        ax.barh(top_features['feature'], top_features['importance'], 
               color='teal', alpha=0.7)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Top Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
    def create_interactive_strategy_dashboard(self, df: pd.DataFrame, 
                                             strategies_df: pd.DataFrame,
                                             save_path: str = None):
        """
        Create interactive Plotly dashboard
        
        Args:
            df: Lap data
            strategies_df: Strategy comparison data
            save_path: Path to save HTML
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Tire Degradation', 'Strategy Comparison',
                          'Lap Time Distribution', 'Pit Window Analysis'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                   [{'type': 'box'}, {'type': 'scatter'}]]
        )
        
        # 1. Tire Degradation
        for compound in df['Compound'].unique():
            compound_df = df[df['Compound'] == compound]
            degradation = compound_df.groupby('LapInStint')['TireDegradationRate'].mean()
            
            fig.add_trace(
                go.Scatter(x=degradation.index, y=degradation.values,
                          mode='lines+markers', name=compound,
                          marker=dict(color=self.tire_colors.get(compound, 'gray'))),
                row=1, col=1
            )
        
        # 2. Strategy Comparison
        fig.add_trace(
            go.Bar(x=strategies_df['rank'], y=strategies_df['total_time'],
                  name='Total Time', marker_color='steelblue'),
            row=1, col=2
        )
        
        # 3. Lap Time Distribution by Compound
        for compound in df['Compound'].unique():
            compound_df = df[df['Compound'] == compound]
            fig.add_trace(
                go.Box(y=compound_df['LapTimeSeconds'], name=compound,
                      marker_color=self.tire_colors.get(compound, 'gray')),
                row=2, col=1
            )
        
        # 4. Pit Window Analysis
        avg_times = df.groupby('LapNumber')['LapTimeSeconds'].mean()
        fig.add_trace(
            go.Scatter(x=avg_times.index, y=avg_times.values,
                      mode='lines', name='Avg Lap Time',
                      line=dict(color='darkblue', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="F1 Pit Stop Strategy Dashboard",
            title_font_size=20
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved interactive dashboard: {save_path}")
        
        fig.show()
        
    def generate_strategy_report(self, best_strategy, comparison_df: pd.DataFrame,
                                save_path: str = None):
        """
        Generate a text report of the optimal strategy
        
        Args:
            best_strategy: Best RaceStrategy object
            comparison_df: Strategy comparison DataFrame
            save_path: Path to save report
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          F1 PIT STOP STRATEGY OPTIMIZATION REPORT           ║
╚══════════════════════════════════════════════════════════════╝

OPTIMAL STRATEGY
─────────────────────────────────────────────────────────────
Total Race Time:     {best_strategy.total_time:.2f} seconds
Number of Pit Stops: {len(best_strategy.pit_laps)}
Pit Laps:            {', '.join(map(str, best_strategy.pit_laps))}
Tire Sequence:       {' → '.join(best_strategy.tire_sequence)}

STINT BREAKDOWN
─────────────────────────────────────────────────────────────
"""
        
        for stint in best_strategy.stints:
            report += f"Stint {stint['stint_number']}: "
            report += f"Laps {stint['start_lap']}-{stint['end_lap']} "
            report += f"({stint['length']} laps) on {stint['compound']} tires\n"
        
        report += f"""
ALTERNATIVE STRATEGIES
─────────────────────────────────────────────────────────────
Top 5 Alternative Strategies:

"""
        
        for _, row in comparison_df.head(5).iterrows():
            report += f"{int(row['rank'])}. Time: {row['total_time']:.2f}s "
            report += f"(+{row['time_delta']:.2f}s) | "
            report += f"{int(row['n_stops'])} stops | "
            report += f"{row['tire_sequence']}\n"
        
        report += "\n" + "="*65 + "\n"
        
        print(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
             f.write(report)
            print(f"Report saved to: {save_path}")
        
        return report


def main():
    """Example usage"""
    # Load data
    df = pd.read_csv('./data/processed/processed_laps.csv')
    strategies_df = pd.read_csv('./data/processed/strategy_comparison.csv')
    
    # Initialize visualizer
    viz = F1Visualizer()
    
    # Create output directory
    os.makedirs('./output/visualizations', exist_ok=True)
    
    # Generate plots
    viz.plot_tire_degradation(df, save_path='./output/visualizations/tire_degradation.png')
    viz.plot_lap_time_evolution(df, save_path='./output/visualizations/lap_times.png')
    viz.plot_strategy_comparison(strategies_df, save_path='./output/visualizations/strategy_comparison.png')
    viz.plot_pit_window_analysis(df, save_path='./output/visualizations/pit_windows.png')
    
    # Create interactive dashboard
    viz.create_interactive_strategy_dashboard(df, strategies_df, 
                                             save_path='./output/visualizations/dashboard.html')
    
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()