"""
Optimization Engine Module
Simulates and optimizes F1 pit stop strategies
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class RaceStrategy:
    """Represents a pit stop strategy"""
    
    def __init__(self, pit_laps: List[int], tire_sequence: List[str], total_time: float = None):
        self.pit_laps = pit_laps
        self.tire_sequence = tire_sequence
        self.total_time = total_time
        self.stints = self._calculate_stints()
        
    def _calculate_stints(self) -> List[Dict]:
        """Calculate stint details"""
        stints = []
        start_lap = 1
        
        for i, pit_lap in enumerate(self.pit_laps):
            stints.append({
                'stint_number': i + 1,
                'start_lap': start_lap,
                'end_lap': pit_lap,
                'length': pit_lap - start_lap + 1,
                'compound': self.tire_sequence[i]
            })
            start_lap = pit_lap + 1
            
        return stints
    
    def __repr__(self):
        return f"Strategy(pits={self.pit_laps}, tires={self.tire_sequence}, time={self.total_time:.2f}s)"


class F1StrategyOptimizer:
    """Optimizes pit stop strategies using simulation"""
    
    def __init__(self, lap_time_model, pit_loss_seconds: float = 25.0):
        """
        Initialize optimizer
        
        Args:
            lap_time_model: Trained ML model for lap time prediction
            pit_loss_seconds: Time loss per pit stop
        """
        self.lap_time_model = lap_time_model
        self.pit_loss = pit_loss_seconds
        
        # Tire compound characteristics
        self.tire_degradation = {
            'SOFT': 0.08,      # Seconds lost per lap
            'MEDIUM': 0.04,
            'HARD': 0.02
        }
        
        self.tire_base_pace = {
            'SOFT': 0.0,       # Fastest (baseline)
            'MEDIUM': 0.3,     # 0.3s slower
            'HARD': 0.6        # 0.6s slower
        }
        
    def simulate_race(self, strategy: RaceStrategy, race_params: Dict) -> float:
        """
        Simulate a complete race with given strategy
        
        Args:
            strategy: RaceStrategy object
            race_params: Dictionary with race parameters
            
        Returns:
            Total race time in seconds
        """
        total_laps = race_params['total_laps']
        base_lap_time = race_params['base_lap_time']
        fuel_effect = race_params.get('fuel_effect', 0.05)  # Seconds per lap for fuel
        
        total_time = 0.0
        
        # Simulate each stint
        for i, stint in enumerate(strategy.stints):
            compound = stint['compound']
            stint_length = stint['length']
            
            # Calculate stint time
            for lap in range(stint_length):
                # Base time for this compound
                lap_time = base_lap_time + self.tire_base_pace[compound]
                
                # Add tire degradation
                lap_time += self.tire_degradation[compound] * lap
                
                # Add fuel effect (decreasing weight)
                remaining_laps = total_laps - (stint['start_lap'] + lap)
                lap_time -= fuel_effect * (1 - remaining_laps / total_laps)
                
                total_time += lap_time
            
            # Add pit stop time (except after last stint)
            if i < len(strategy.stints) - 1:
                total_time += self.pit_loss
                
        strategy.total_time = total_time
        return total_time
    
    def generate_strategies(self, total_laps: int = 70, max_stops: int = 2, 
                          min_stint_length: int = 5) -> List[RaceStrategy]:
        """Generate realistic F1 pit strategies (1-2 stops, valid tire rules)"""
        strategies = []
        tires = ['SOFT', 'MEDIUM', 'HARD']
        
        # 1-STOP STRATEGIES (most common in modern F1)
        for pit_lap in range(12, min(total_laps - 10, 45)):  # Realistic pit window
            for start_tire in tires:
                for end_tire in tires:
                    if start_tire == end_tire:
                        continue  # Must change compounds
                    if end_tire == 'SOFT' and pit_lap > 20:
                        continue  # No soft for long stints
                    
                    # Validate stint lengths
                    if pit_lap < min_stint_length or total_laps - pit_lap < min_stint_length:
                        continue
                        
                    strategies.append(RaceStrategy(
                        pit_laps=[pit_lap],
                        tire_sequence=[start_tire, end_tire]
                    ))
        
        # 2-STOP STRATEGIES (backup option)
        if max_stops >= 2:
            for pit1 in range(10, min(total_laps - 20, 30)):
                for pit2 in range(pit1 + 10, min(total_laps - 10, 50)):
                    for t1 in tires:
                        for t2 in tires:
                            for t3 in tires:
                                if t1 == t2 or t2 == t3:
                                    continue  # Different compounds
                                if t3 == 'SOFT':
                                    continue  # Avoid soft at end
                                
                                # Validate stint lengths
                                if (pit1 < min_stint_length or 
                                    pit2 - pit1 < min_stint_length or 
                                    total_laps - pit2 < min_stint_length):
                                    continue
                                    
                                strategies.append(RaceStrategy(
                                    pit_laps=[pit1, pit2],
                                    tire_sequence=[t1, t2, t3]
                                ))
        
        print(f"Generated {len(strategies)} realistic strategies")
        return strategies[:1000]  # Top 1000 for efficiency
    
    def optimize_strategy(self, race_params: Dict, n_strategies: int = 50) -> List[RaceStrategy]:
        """
        Find optimal pit stop strategy
        
        Args:
            race_params: Race parameters
            n_strategies: Number of strategies to evaluate
            
        Returns:
            List of strategies sorted by total time (best first)
        """
        print("\n=== Optimizing Pit Stop Strategy ===")
        
        total_laps = race_params['total_laps']
        
        # Generate strategies
        print("Generating strategies...")
        all_strategies = self.generate_strategies(
            total_laps, 
            max_stops=race_params.get('max_stops', 2),
            min_stint_length=race_params.get('min_stint', 5)
        )
        
        print(f"Generated {len(all_strategies)} possible strategies")
        
        # Limit number of strategies to evaluate
        if len(all_strategies) > n_strategies:
            strategies = list(np.random.choice(all_strategies, n_strategies, replace=False))
        else:
            strategies = all_strategies
        
        # Simulate each strategy
        print("Simulating strategies...")
        for strategy in strategies:
            self.simulate_race(strategy, race_params)
        
        # Sort by total time
        strategies = sorted(strategies, key=lambda s: s.total_time)
        
        print(f"\n=== Top 5 Strategies ===")
        for strategy in strategies[:5]:
            print(strategy)
        
        return strategies
    
    def compare_strategies(self, strategies: List[RaceStrategy], top_n: int = 5) -> pd.DataFrame:
        """
        Compare top strategies
        
        Returns:
            DataFrame with strategy comparison
        """
        comparison = []
        
        for i, strategy in enumerate(strategies[:top_n]):
            comparison.append({
                'rank': i + 1,
                'total_time': strategy.total_time,
                'n_stops': len(strategy.pit_laps),
                'pit_laps': str(strategy.pit_laps),
                'tire_sequence': ' → '.join(strategy.tire_sequence),
                'time_delta': strategy.total_time - strategies[0].total_time
            })
        
        return pd.DataFrame(comparison)


class GeneticOptimizer:
    """Genetic algorithm for strategy optimization"""
    
    def __init__(self, simulator: F1StrategyOptimizer, population_size: int = 50):
        self.simulator = simulator
        self.population_size = population_size
        
    def optimize(self, race_params: Dict, generations: int = 20) -> RaceStrategy:
        """
        Optimize using genetic algorithm
        
        Args:
            race_params: Race parameters
            generations: Number of generations
            
        Returns:
            Best strategy found
        """
        print("\n=== Genetic Algorithm Optimization ===")
        
        # Initialize population
        population = self.simulator.generate_strategies(
            race_params['total_laps'],
            max_stops=2,
            min_stint_length=10
        )[:self.population_size]
        
        # Evaluate initial population
        for strategy in population:
            self.simulator.simulate_race(strategy, race_params)
        
        # Evolution loop
        for gen in range(generations):
            # Selection
            population = sorted(population, key=lambda s: s.total_time)
            elite = population[:10]  # Keep top 10
            
            # Crossover and mutation
            offspring = self._crossover(elite, race_params)
            
            population = elite + offspring[:self.population_size - 10]
            
            if gen % 5 == 0:
                print(f"Generation {gen}: Best time = {population[0].total_time:.2f}s")
        
        best_strategy = population[0]
        print(f"\nBest strategy: {best_strategy}")
        
        return best_strategy
    
    def _crossover(self, elite: List[RaceStrategy], race_params: Dict) -> List[RaceStrategy]:
        """Create offspring from elite strategies"""
        offspring = []
        
        for _ in range(40):
            # Select two parents
            parent1, parent2 = np.random.choice(elite, 2, replace=False)
            
            # Crossover: mix pit laps and tire choices
            pit_laps = sorted(set(parent1.pit_laps + parent2.pit_laps))
            if len(pit_laps) > len(parent1.pit_laps):
                pit_laps = pit_laps[:len(parent1.pit_laps)]
            
            tire_seq = [np.random.choice([parent1.tire_sequence[i % len(parent1.tire_sequence)],
                                          parent2.tire_sequence[i % len(parent2.tire_sequence)]])
                       for i in range(len(pit_laps) + 1)]
            
            child = RaceStrategy(pit_laps, tire_seq)
            self.simulator.simulate_race(child, race_params)
            offspring.append(child)
        
        return offspring


def main():
    """Example usage"""
    from model_training import F1ModelTrainer
    
    # Load model (in practice, load pre-trained model)
    trainer = F1ModelTrainer()
    # model = trainer.load_model('best_model')
    model = None  # Placeholder
    
    # Initialize optimizer
    optimizer = F1StrategyOptimizer(model, pit_loss_seconds=25.0)
    
    # Define race parameters
    race_params = {
        'total_laps': 50,
        'base_lap_time': 85.0,  # seconds
        'fuel_effect': 0.05,
        'max_stops': 2,
        'min_stint': 10
    }
    
    # Optimize strategy
    strategies = optimizer.optimize_strategy(race_params, n_strategies=100)
    
    # Compare top strategies
    comparison_df = optimizer.compare_strategies(strategies, top_n=10)
    print("\n=== Strategy Comparison ===")
    print(comparison_df)
    
    # Save results
    comparison_df.to_csv('./data/processed/strategy_comparison.csv', index=False)
    
    print("\n=== Optimization Complete ===")


if __name__ == "__main__":
    main()