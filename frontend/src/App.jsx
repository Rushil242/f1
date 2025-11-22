import React, { useEffect, useState } from 'react';
import axios from 'axios';
import TrackMap from './TrackMap';
import RadioControl from './RadioControl';
import ThemeToggle from './ThemeToggle'; // <--- IMPORT
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Zap, Activity, Timer } from 'lucide-react';

function App() {
  const [dominanceData, setDominanceData] = useState(null);
  const [telemetryData, setTelemetryData] = useState([]);
  const [tireData, setTireData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const domRes = await axios.get('/api/dominance?d1=VER&d2=LEC');
        setDominanceData(domRes.data);

        const ghostRes = await axios.get('/api/ghost?d1=VER&d2=LEC');
        const d1 = ghostRes.data.driver_1;
        const d2 = ghostRes.data.driver_2;
        const mergedTel = d1.distance.map((dist, i) => ({
          distance: dist,
          speed1: d1.speed[i],
          speed2: d2.speed[i]
        }));
        setTelemetryData(mergedTel);

        const tireRes = await axios.get('/api/tire-cliff?driver=VER');
        setTireData(tireRes.data);
      } catch (error) {
        console.error("API Error:", error);
      }
    };
    fetchData();
  }, []);

  return (
    // REMOVED "bg-mission-bg text-white" (Handled by global CSS now)
    <div className="min-h-screen p-6 transition-colors duration-300">
      
      {/* Header */}
      <header className="mb-8 border-b border-mission-border pb-4 flex justify-between items-end">
        <div>
          <h1 className="text-4xl font-black tracking-tighter italic bg-clip-text text-transparent bg-gradient-to-r from-mission-red to-orange-500">
            F1 MISSION CONTROL
          </h1>
          <p className="text-gray-500 font-mono text-sm mt-1">
            POWERED BY LSTM NEURAL NETWORKS & FASTF1
          </p>
        </div>
        <div className="flex items-center gap-4">
            <ThemeToggle /> {/* <--- ADD TOGGLE HERE */}
            <div className="text-right">
            <div className="text-xs text-gray-500">SESSION</div>
            <div className="text-xl font-bold text-mission-text">BAHRAIN GP 2023</div>
            </div>
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        <div className="lg:col-span-2 space-y-6">
          <TrackMap data={dominanceData} />
          
          {/* Updated borders and text colors */}
          <div className="bg-mission-card p-4 rounded-xl border border-mission-border h-[300px]">
            <h3 className="text-sm font-bold text-gray-400 mb-4 flex items-center gap-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              TELEMETRY GHOST OVERLAY
            </h3>
            <ResponsiveContainer width="100%" height="80%">
              {telemetryData.length > 0 ? (
                <LineChart data={telemetryData}>
                  <XAxis dataKey="distance" hide />
                  <YAxis domain={['dataMin', 'dataMax']} hide />
                  <Tooltip contentStyle={{ backgroundColor: 'var(--mission-card)', borderColor: 'var(--mission-border)', color: 'var(--mission-text)' }} />
                  <Legend />
                  <Line type="monotone" dataKey="speed1" name="VER (Red)" stroke="#FF1801" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="speed2" name="LEC (Teal)" stroke="#00D2BE" dot={false} strokeWidth={2} strokeDasharray="3 3" />
                </LineChart>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500 animate-pulse">
                  Loading Telemetry...
                </div>
              )}
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-mission-card p-6 rounded-xl border border-mission-border">
             <h3 className="text-sm font-bold text-gray-400 mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              LSTM TIRE PREDICTION
            </h3>
            
            {tireData ? (
              <div className="text-center">
                <div className={`text-3xl font-black mb-2 ${tireData.tire_health_status.includes('CRITICAL') ? 'text-mission-red animate-pulse' : 'text-green-500'}`}>
                  {tireData.tire_health_status}
                </div>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div className="bg-mission-bg p-3 rounded border border-mission-border">
                    <div className="text-xs text-gray-500">PREDICTED LAP</div>
                    <div className="text-xl font-mono text-mission-text">{tireData.predicted_next_lap}s</div>
                  </div>
                  <div className="bg-mission-bg p-3 rounded border border-mission-border">
                    <div className="text-xs text-gray-500">PACE DROP</div>
                    <div className="text-xl font-mono text-mission-text">{tireData.pace_drop_predicted}s</div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-500 animate-pulse">Running Neural Network...</div>
            )}
          </div>

          <RadioControl driver="VER" />

          <div className="bg-gradient-to-br from-gray-800 to-black dark:from-gray-900 dark:to-black p-6 rounded-xl border border-mission-border text-white">
            <div className="flex items-center gap-3 mb-2">
              <Timer className="w-5 h-5 text-purple-500" />
              <span className="font-bold">SYSTEM LATENCY</span>
            </div>
            <div className="text-3xl font-mono">12ms</div>
          </div>

        </div>
      </div>
    </div>
  );
}

export default App;