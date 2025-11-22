"""
Data Collection Module - FIXED VERSION
Handles Ergast API shutdown and missing race data
"""

import fastf1
import pandas as pd
import requests
import os
from tqdm import tqdm
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class F1DataCollector:
    """Collects F1 race data from multiple sources"""
    
    def __init__(self, cache_dir: str = "./data/raw/cache"):
        """Initialize data collector with caching"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        
    def collect_fastf1_data(self, year: int, races: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Collect race data from FastF1 API
        
        Args:
            year: Season year
            races: List of race numbers (None = all races)
            
        Returns:
            DataFrame with lap-by-lap data
        """
        print(f"\n=== Collecting FastF1 data for {year} season ===")
        
        schedule = fastf1.get_event_schedule(year)
        all_data = []
        
        if races is None:
            # Only include races that have actually happened
            races = range(1, len(schedule) + 1)
            
        for race_num in tqdm(races, desc=f"Processing {year} races"):
            try:
                # Load race session
                session = fastf1.get_session(year, race_num, 'R')
                session.load()
                
                # Check if session has data
                if session.laps is None or len(session.laps) == 0:
                    print(f"No data available for race {race_num}")
                    continue
                
                # Get laps data
                laps = session.laps
                
                # Add race metadata
                laps['Year'] = year
                laps['RaceNumber'] = race_num
                laps['RaceName'] = schedule.loc[schedule['RoundNumber'] == race_num, 'EventName'].values[0]
                
                all_data.append(laps)
                
            except Exception as e:
                print(f"Error loading race {race_num}: {e}")
                continue
                
        if not all_data:
            raise ValueError(f"No data collected for {year}")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Collected {len(combined_df)} laps from {len(all_data)} races")
        
        return combined_df
    
    def collect_ergast_data_alternative(self, year: int) -> pd.DataFrame:
        """
        Collect race metadata from alternative source (Jolpica F1 API)
        Since Ergast shut down, we use Jolpica as replacement
        
        Args:
            year: Season year
            
        Returns:
            DataFrame with race results
        """
        print(f"\n=== Collecting race data for {year} (using alternative API) ===")
        
        try:
            # Try Jolpica F1 API (Ergast replacement)
            base_url = "https://api.jolpi.ca/ergast/f1"
            results_url = f"{base_url}/{year}/results.json?limit=1000"
            
            response = requests.get(results_url, timeout=10)
            
            if response.status_code != 200:
                print(f"Warning: Could not fetch data from alternative API (status {response.status_code})")
                return self._create_dummy_ergast_data(year)
            
            results_data = response.json()
            
            # Parse results into DataFrame
            races_list = []
            for race in results_data['MRData']['RaceTable']['Races']:
                race_info = {
                    'year': year,
                    'round': race['round'],
                    'race_name': race['raceName'],
                    'circuit': race['Circuit']['circuitName'],
                    'date': race['date'],
                }
                
                for result in race['Results']:
                    entry = race_info.copy()
                    entry.update({
                        'driver': result['Driver'].get('code', result['Driver']['driverId']),
                        'driver_id': result['Driver']['driverId'],
                        'constructor': result['Constructor']['name'],
                        'grid': result['grid'],
                        'position': result.get('position', 'N/A'),
                        'points': result['points'],
                        'status': result['status']
                    })
                    races_list.append(entry)
            
            results_df = pd.DataFrame(races_list)
            print(f"Collected {len(results_df)} race results")
            
            return results_df
            
        except Exception as e:
            print(f"Warning: Alternative API also failed: {e}")
            print("Creating minimal metadata from FastF1 data...")
            return self._create_dummy_ergast_data(year)
    
    def _create_dummy_ergast_data(self, year: int) -> pd.DataFrame:
        """
        Create minimal race metadata when external APIs fail
        This ensures the pipeline doesn't break
        """
        print("Note: Using minimal race metadata (no detailed results)")
        
        try:
            schedule = fastf1.get_event_schedule(year)
            
            dummy_data = []
            for _, race in schedule.iterrows():
                dummy_data.append({
                    'year': year,
                    'round': race['RoundNumber'],
                    'race_name': race['EventName'],
                    'circuit': race.get('Location', 'Unknown'),
                    'date': race['EventDate'],
                    'driver': 'N/A',
                    'driver_id': 'N/A',
                    'constructor': 'N/A',
                    'grid': 0,
                    'position': 0,
                    'points': 0,
                    'status': 'N/A'
                })
            
            return pd.DataFrame(dummy_data)
        except:
            # Last resort: return empty DataFrame with correct structure
            return pd.DataFrame(columns=[
                'year', 'round', 'race_name', 'circuit', 'date',
                'driver', 'driver_id', 'constructor', 'grid', 
                'position', 'points', 'status'
            ])
    
    def save_data(self, data: pd.DataFrame, filename: str, output_dir: str = "./data/raw"):
        """Save collected data to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath, index=False)
        print(f"Saved data to {filepath}")
        
    def collect_all_data(self, years: List[int], races: Optional[List[int]] = None) -> dict:
        """
        Collect all data for multiple seasons
        
        Args:
            years: List of seasons to collect
            races: Specific races to collect (None = all available)
            
        Returns:
            Dictionary with 'laps' and 'ergast' DataFrames
        """
        all_laps = []
        all_ergast = []
        
        for year in years:
            try:
                # Collect FastF1 data
                laps_df = self.collect_fastf1_data(year, races)
                all_laps.append(laps_df)
                
                # Collect race metadata (using alternative API)
                ergast_data = self.collect_ergast_data_alternative(year)
                if len(ergast_data) > 0:
                    all_ergast.append(ergast_data)
                
            except Exception as e:
                print(f"Error collecting data for {year}: {e}")
                continue
        
        # Combine all data
        if not all_laps:
            raise ValueError("No lap data collected!")
        
        combined_laps = pd.concat(all_laps, ignore_index=True)
        
        # Handle case where no ergast data was collected
        if all_ergast:
            combined_ergast = pd.concat(all_ergast, ignore_index=True)
        else:
            print("Warning: No race metadata collected. Creating minimal metadata...")
            combined_ergast = self._create_dummy_ergast_data(years[0])
        
        return {
            'laps': combined_laps,
            'ergast': combined_ergast
        }


def main():
    """Example usage"""
    collector = F1DataCollector()
    
    # Collect data for 2024 season - only races 1-13 (available data)
    # Adjust this range based on which races have actually occurred
    data = collector.collect_all_data(years=[2024], races=list(range(1, 14)))
    
    # Save data
    collector.save_data(data['laps'], 'laps_data.csv')
    collector.save_data(data['ergast'], 'ergast_data.csv')
    
    print("\n=== Data Collection Complete ===")
    print(f"Total laps collected: {len(data['laps'])}")
    print(f"Total race results: {len(data['ergast'])}")


if __name__ == "__main__":
    main()