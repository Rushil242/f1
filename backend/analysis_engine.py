import fastf1
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Enable FastF1 Cache (Speed up repeated requests)
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
fastf1.Cache.enable_cache('f1_cache')

class RaceAnalytics:
    def __init__(self, year=2023, race='Bahrain', session='R'):
        print(f"Loading Session: {year} {race} {session}...")
        self.session = fastf1.get_session(year, race, session)
        self.session.load()
        self.laps = self.session.laps
        print("Session Loaded.")

        # Load AI Models
        self.load_models()

    def load_models(self):
        """Load the trained LSTM model and Scaler with Robust Path Finding"""
        try:
            # 1. Get the absolute path of THIS file (analysis_engine.py)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # 2. Go up one level to the project root (f1/)
            project_root = os.path.dirname(current_dir)
            
            # 3. Construct the full path to the model
            model_path = os.path.join(project_root, 'backend','data', 'models', 'lstm_deep_learning_model.keras')
            scaler_path = os.path.join(project_root, 'backend','data', 'models', 'lstm_scaler.pkl')

            print(f"🔍 DEBUG: Attempting to load model from: {model_path}")

            # 4. Verify file existence before loading
            if not os.path.exists(model_path):
                print(f"❌ CRITICAL ERROR: Model file not found at {model_path}")
                print("Did you run 'python src/model_training.py'?")
                self.lstm_model = None
                return

            # 5. Load
            self.lstm_model = tf.keras.models.load_model(model_path)
            
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                print(f"⚠️ Warning: Scaler not found at {scaler_path}")
                
            print("✅ LSTM Model & Scaler Loaded Successfully")
            
        except Exception as e:
            print(f"⚠️ Model Loading Failed: {e}")
            self.lstm_model = None

    def get_track_dominance(self, driver_1, driver_2):
        """
        FEATURE 1: SPATIAL DOMINANCE
        Compares speed at 100 points along the track.
        """
        # Get fastest laps
        l1 = self.laps.pick_driver(driver_1).pick_fastest()
        l2 = self.laps.pick_driver(driver_2).pick_fastest()

        # Get telemetry
        t1 = l1.get_telemetry().add_distance()
        t2 = l2.get_telemetry().add_distance()

        # Create common distance axis (interpolation)
        max_dist = max(t1['Distance'].max(), t2['Distance'].max())
        x_new = np.linspace(0, max_dist, 100) # 100 mini-sectors

        # Interpolate speed, X, Y
        s1 = np.interp(x_new, t1['Distance'], t1['Speed'])
        s2 = np.interp(x_new, t2['Distance'], t2['Speed'])
        
        # Track coordinates for the map
        x_coords = np.interp(x_new, t1['Distance'], t1['X'])
        y_coords = np.interp(x_new, t1['Distance'], t1['Y'])

        dominance_data = []
        for i in range(len(x_new)):
            color = '#FF1801' if s1[i] > s2[i] else '#00D2BE' # Red vs Teal
            dominance_data.append({
                "x": x_coords[i],
                "y": y_coords[i],
                "color": color,
                "driver_faster": driver_1 if s1[i] > s2[i] else driver_2
            })

        return dominance_data

    def get_telemetry_ghost(self, driver_1, driver_2):
        """
        FEATURE 2: TELEMETRY GHOST
        Returns aligned telemetry for comparing throttle/brake inputs.
        """
        l1 = self.laps.pick_driver(driver_1).pick_fastest()
        l2 = self.laps.pick_driver(driver_2).pick_fastest()

        t1 = l1.get_telemetry().add_distance()
        t2 = l2.get_telemetry().add_distance()

        # Downsample for frontend performance (every 10th point)
        def process_tel(t, driver):
            return {
                "driver": driver,
                "distance": t['Distance'].tolist()[::5],
                "speed": t['Speed'].tolist()[::5],
                "throttle": t['Throttle'].tolist()[::5],
                "brake": t['Brake'].tolist()[::5]
            }

        return {
            "driver_1": process_tel(t1, driver_1),
            "driver_2": process_tel(t2, driver_2)
        }

    def predict_tire_cliff(self, driver):
        """
        FEATURE 3: LSTM TIRE SENTIENCE
        Predicts the next 5 laps. If lap time spikes, we flag a 'Cliff'.
        """
        if not self.lstm_model:
            return {"error": "AI Model not loaded"}

        # 1. Get recent history (Last 5 laps)
        driver_laps = self.laps.pick_driver(driver)
        # Filter out pit in/out laps for cleaner data
        clean_laps = driver_laps.dropna(subset=['LapTime'])
        
        if len(clean_laps) < 5:
            return {"status": "Insufficient Data"}

        recent_5 = clean_laps.iloc[-5:].copy()

        # 2. Feature Engineering (Must match training exactly!)
        # Calculate FuelLoad proxy
        total_laps = 57 # Approx for Bahrain
        recent_5['FuelLoad'] = total_laps - recent_5['LapNumber']
        
        # Calculate TireAge (If missing, assuming incremental stint)
        if 'TyreLife' in recent_5.columns:
             recent_5['TireAge'] = recent_5['TyreLife']
        else:
             recent_5['TireAge'] = range(10, 15) # Fallback

        # 3. Prepare Input Vector
        features = ['LapNumber', 'TireAge', 'FuelLoad']
        input_data = recent_5[features].values
        
        # Scale
        # Note: We need to pad with a dummy target to use the same scaler
        # Creating a temporary array with 4 columns (3 features + 1 target) to fit scaler
        dummy_target = np.zeros((5, 1))
        input_with_dummy = np.hstack((input_data, dummy_target))
        scaled_input = self.scaler.transform(input_with_dummy)
        
        # Extract just the features (first 3 cols) for LSTM input
        X_input = scaled_input[:, :-1].reshape(1, 5, 3) # (Batch, Lookback, Features)

        # 4. AI Prediction
        predicted_scaled_time = self.lstm_model.predict(X_input)[0][0]
        
        # Inverse transform to get seconds (Tricky part: we need to inverse a full row)
        dummy_row = np.zeros((1, 4))
        dummy_row[0, -1] = predicted_scaled_time
        predicted_seconds = self.scaler.inverse_transform(dummy_row)[0, -1]

        # 5. Logic: Is this a Cliff?
        # If predicted time is > 1.5s slower than average of last 3 laps
        avg_recent = recent_5['LapTime'].dt.total_seconds().mean()
        diff = predicted_seconds - avg_recent
        
        health_status = "HEALTHY"
        if diff > 1.5: health_status = "CRITICAL (CLIFF DETECTED)"
        elif diff > 0.5: health_status = "DEGRADING"

        # RETURN THIS UPDATED DICTIONARY
        return {
            "driver": driver,
            "predicted_next_lap": round(predicted_seconds, 3),
            "pace_drop_predicted": round(diff, 3),
            "tire_health_status": health_status,
            # Add a summary string for the LLM
            "dominance_summary": "Pace is stable" if diff < 0.5 else "Losing significant time in Sector 3"
        }