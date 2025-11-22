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
        # Using 'gemini-1.5-flash' for speed (Race engineers need to be fast)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Ensure audio output directory exists
        if not os.path.exists('backend/audio_cache'):
            os.makedirs('backend/audio_cache')

    def analyze_and_speak(self, context_data):
        """
        1. Takes complex JSON data.
        2. Uses Gemini to write a short radio script.
        3. Converts script to Audio.
        """
        if not self.model:
            return {"error": "No API Key"}, None

        # A. THE BRAIN (Gemini)
        script = self._generate_script(context_data)
        
        # B. THE VOICE (TTS)
        audio_path = self._generate_audio(script)
        
        return script, audio_path

    def _generate_script(self, data):
        """Uses Gemini to act as a Race Engineer"""
        
        system_prompt = """
        You are an elite F1 Race Engineer (like GP or Bono). 
        Your driver is currently racing. 
        I will give you telemetry data. You must respond with a RADIO MESSAGE.
        
        Rules:
        1. Be concise, calm, and authoritative. (Max 2 sentences).
        2. If 'tire_health_status' is CRITICAL, command a "Box Box".
        3. Use F1 terminology (Delta, Deg, Box, Push, Lift and Coast).
        4. Do NOT explain the math, just give the strategic instruction.
        """
        
        user_message = f"""
        Here is the live telemetry:
        - Driver: {data.get('driver', 'Unknown')}
        - Predicted Next Lap: {data.get('predicted_next_lap', 'N/A')}s
        - Pace Drop: {data.get('pace_drop_predicted', 0)}s
        - Tire Health: {data.get('tire_health_status', 'Unknown')}
        - Track Dominance: {data.get('dominance_summary', 'N/A')}
        """
        
        try:
            response = self.model.generate_content([system_prompt, user_message])
            return response.text
        except Exception as e:
            return "Radio check. Systems offline."

    def _generate_audio(self, text):
        """Converts text to MP3"""
        try:
            # We use a timestamp to avoid browser caching issues with the audio file
            filename = f"radio_{int(time.time())}.mp3"
            filepath = os.path.join('backend/audio_cache', filename)
            
            # gTTS (Google Text-to-Speech)
            # lang='en' (English), tld='co.uk' gives it a British accent (Classic F1 style)
            tts = gTTS(text=text, lang='en', tld='co.uk', slow=False)
            tts.save(filepath)
            
            return filepath
        except Exception as e:
            print(f"TTS Error: {e}")
            return None