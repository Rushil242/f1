import google.generativeai as genai
from gtts import gTTS
import os
import time

class RaceEngineerAgent:
    def __init__(self, api_key):
        if not api_key:
            print("⚠️ WARNING: No Gemini API Key found. Voice Agent disabled.")
            self.model = None
            return
            
        genai.configure(api_key=api_key)
        # Use the STABLE model. 2.0 is experimental/preview and often causes errors.
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        
        # --- FIX: USE ABSOLUTE PATHS ---
        # Get the directory where this script (voice_agent.py) lives
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Create audio_cache folder right next to this script
        self.audio_dir = os.path.join(base_dir, 'audio_cache')
        
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)
            print(f"📁 Created audio cache at: {self.audio_dir}")

    def analyze_and_speak(self, context_data):
        if not self.model:
            return {"error": "No API Key"}, None

        # A. Generate Script
        script = self._generate_script(context_data)
        
        # B. Generate Audio
        audio_path = self._generate_audio(script)
        
        return script, audio_path

    def _generate_script(self, data):
        system_prompt = """
        You are an elite F1 Race Engineer (like GP or Bono). 
        Your driver is currently racing. 
        I will give you telemetry data. You must respond with a RADIO MESSAGE.
        
        Rules:
        1. Be concise, calm, and authoritative. (Max 2 sentences).
        2. If 'tire_health_status' is CRITICAL, command a "Box Box".
        3. Use F1 terminology (Delta, Deg, Box, Push, Lift and Coast).
        """
        
        user_message = f"""
        Telemetry:
        - Driver: {data.get('driver', 'Unknown')}
        - Predicted Next Lap: {data.get('predicted_next_lap', 'N/A')}
        - Pace Drop: {data.get('pace_drop_predicted', 0)}
        - Tire Health: {data.get('tire_health_status', 'Unknown')}
        """
        
        try:
            # Generate content
            response = self.model.generate_content([system_prompt, user_message])
            return response.text
        except Exception as e:
            print(f"🔴 Gemini API Error: {e}")
            return "Radio check. Systems offline."

    def _generate_audio(self, text):
        try:
            filename = f"radio_{int(time.time())}.mp3"
            # Save to the absolute path we calculated in __init__
            filepath = os.path.join(self.audio_dir, filename)
            
            tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
            tts.save(filepath)
            print(f"🔊 Audio saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"🔴 TTS Error: {e}")
            return None