import React, { useState } from 'react';
import { Mic, Play, Activity } from 'lucide-react';
import axios from 'axios';

const RadioControl = ({ driver }) => {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("RADIO SILENT");

  const handleRadioCheck = async () => {
    setLoading(true);
    setStatus("ESTABLISHING LINK...");
    
    try {
      // 1. Request Audio Blob from Backend
      const response = await axios.get(`/api/radio?driver=${driver}`, {
        responseType: 'blob' // Important for audio files
      });

      // 2. Play Audio
      const audioUrl = URL.createObjectURL(response.data);
      const audio = new Audio(audioUrl);
      setStatus("TRANSMITTING...");
      audio.play();
      
      audio.onended = () => {
        setStatus("RADIO SILENT");
        setLoading(false);
      };

    } catch (error) {
      console.error("Radio Error", error);
      setStatus("CONNECTION FAILED");
      setLoading(false);
    }
  };

  return (
    <div className="bg-mission-card p-6 rounded-xl border-l-4 border-mission-green shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-mono font-bold flex items-center gap-2">
          <Mic className="w-5 h-5 text-mission-green" />
          RACE ENGINEER (AI)
        </h2>
        <div className="flex items-center gap-2">
            {loading && <Activity className="w-5 h-5 text-mission-green animate-bounce" />}
            <span className={`text-xs font-mono px-2 py-1 rounded ${loading ? 'bg-mission-green text-black' : 'bg-gray-800 text-gray-500'}`}>
                {status}
            </span>
        </div>
      </div>

      <p className="text-gray-400 text-sm mb-4">
        AI analyzes LSTM tire predictions + Telemetry to generate strategic audio commands.
      </p>

      <button 
        onClick={handleRadioCheck}
        disabled={loading}
        className={`w-full py-4 rounded-lg font-bold text-lg tracking-widest transition-all
          ${loading 
            ? 'bg-gray-700 cursor-not-allowed opacity-50' 
            : 'bg-gradient-to-r from-mission-green to-emerald-600 hover:shadow-[0_0_20px_rgba(0,255,0,0.4)] text-black'
          }`}
      >
        {loading ? 'ANALYZING TELEMETRY...' : '🔊 OPEN RADIO CHANNEL'}
      </button>
    </div>
  );
};

export default RadioControl;