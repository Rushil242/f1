from flask import Flask, jsonify, request
from flask_cors import CORS
from analysis_engine import RaceAnalytics
from flask import send_file # Add this import
from voice_agent import RaceEngineerAgent # Import the new class
import os

app = Flask(__name__)
CORS(app) # Allow React to make requests

# Initialize Engine (Loads FastF1 data + Models)
# In a real app, we might load this dynamically, but for demo, we load Bahrain 2023
print("Initializing F1 Mission Control Engine...")
engine = RaceAnalytics(2023, 'Bahrain', 'R')

@app.route('/')
def home():
    return jsonify({"status": "F1 Mission Control Online", "model": "LSTM + XGBoost Active"})

@app.route('/api/dominance', methods=['GET'])
def get_dominance():
    """Feature 1: Track Map Dominance"""
    d1 = request.args.get('d1', 'VER')
    d2 = request.args.get('d2', 'LEC')
    data = engine.get_track_dominance(d1, d2)
    return jsonify(data)

@app.route('/api/ghost', methods=['GET'])
def get_ghost():
    """Feature 2: Telemetry Ghost Overlay"""
    d1 = request.args.get('d1', 'VER')
    d2 = request.args.get('d2', 'HAM')
    data = engine.get_telemetry_ghost(d1, d2)
    return jsonify(data)

@app.route('/api/tire-cliff', methods=['GET'])
def get_cliff():
    """Feature 3: AI Tire Sentience"""
    driver = request.args.get('driver', 'VER')
    try:
        data = engine.predict_tire_cliff(driver)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

GEMINI_API_KEY = "AIzaSyDs0QaB3WkzRUq8aD1iBr3DE8EkR1ulBHM" 

engine = RaceAnalytics(2023, 'Bahrain', 'R')
voice_agent = RaceEngineerAgent(GEMINI_API_KEY)


@app.route('/api/radio', methods=['GET'])
def race_engineer_radio():
    """
    Triggers the Voice Agent.
    1. Gets analysis from Engine.
    2. Sends to Gemini -> Text.
    3. Sends Text to TTS -> Audio File.
    4. Returns Audio File to frontend.
    """
    driver = request.args.get('driver', 'VER')
    
    # 1. Get the hard data
    try:
        data = engine.predict_tire_cliff(driver)
    except:
        data = {"driver": driver, "status": "No Data"}
        
    # 2. Generate Voice Response
    script_text, audio_path = voice_agent.analyze_and_speak(data)
    
    if audio_path and os.path.exists(audio_path):
        # Return the audio file directly
        return send_file(audio_path, mimetype="audio/mpeg")
    else:
        return jsonify({"error": "Radio failed", "script": script_text}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)