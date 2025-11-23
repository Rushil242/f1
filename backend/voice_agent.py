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
        """
        UPGRADE: Generates a 45-60 second 'Strategic Briefing' 
        instead of a short burst.
        """
        
        system_prompt = """
        You are 'GP', the Race Engineer for Max Verstappen. 
        This is a STRATEGIC BRIEFING during a quiet moment in the race.
        
        Your goal is to speak for about 45-60 seconds. 
        Structure your response into these 4 distinct sections:
        
        1. **Current Status**: Summarize the current lap, tire age, and immediate pace.
        2. **Tire Analysis**: deeply analyze the tire degradation (health status). Explain if we are seeing graining or thermal deg.
        3. **The Threat**: Analyze the rival (Leclerc). Is he catching? What are his sector times doing?
        4. **The Decision**: Give the final recommendation (Extend stint, Box now, or Switch Plans).
        
        Tone: Professional, calm, highly technical, but conversational. 
        Use fillers like "Okay Max," "Looking at the data," "Copy that."
        """
        
        user_message = f"""
        Detailed Telemetry Report:
        - Driver: {data.get('driver', 'Unknown')}
        - Current Prediction: {data.get('predicted_next_lap', 'N/A')}s
        - Pace Delta: {data.get('pace_drop_predicted', 0)}s vs target.
        - Tire Model Status: {data.get('tire_health_status', 'Unknown')}
        - Sector Analysis: {data.get('dominance_summary', 'N/A')}
        """
        
        try:
            response = self.model.generate_content([system_prompt, user_message])
            return response.text
        except Exception as e:
            print(f"🔴 Gemini API Error: {e}")
            return "Telemetry link unstable. Stand by."

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