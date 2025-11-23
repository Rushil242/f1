import React, { useEffect, useState } from 'react';
import axios from 'axios';
import TrackMap from '../TrackMap';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Zap, Activity, Thermometer, TrendingUp } from 'lucide-react';

const Dashboard = ({ driver }) => {
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

        const tireRes = await axios.get(`/api/tire-cliff?driver=${driver}`);
        setTireData(tireRes.data);
      } catch (error) { console.error("API Error:", error); }
    };
    fetchData();
  }, [driver]);

  return (
    <div className="p-6 md:p-8 animate-fade-in space-y-6 max-w-[1600px] mx-auto">
      
      {/* TOP ROW: MAP & TIRE STATS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 h-[500px]">
        
        {/* 1. TRACK MAP (Takes 2/3 width) */}
        <div className="lg:col-span-2 h-full">
            <TrackMap data={dominanceData} />
        </div>

        {/* 2. TIRE HEALTH (Takes 1/3 width) - EXPANDED METRICS */}
        <div className="glass-panel p-0 rounded-2xl flex flex-col border-t-4 border-green-500 h-full">
             {/* Header */}
             <div className="p-6 border-b border-white/10 flex justify-between items-center">
                <div className="flex items-center gap-2 text-green-400">
                    <Activity size={18} />
                    <span className="font-bold tracking-widest text-xs">TIRE MODEL (LSTM-V2)</span>
                </div>
                <div className="text-[10px] text-gray-500 font-mono bg-black/40 px-2 py-1 rounded border border-white/5">
                    AI CONFIDENCE: 94.2%
                </div>
             </div>
             
             {tireData ? (
               <div className="flex-1 p-6 flex flex-col gap-6">
                 {/* Main Status */}
                 <div className="text-center py-6 bg-gradient-to-b from-white/5 to-transparent rounded-xl border border-white/5 relative overflow-hidden">
                     <div className={`text-6xl font-black tracking-tighter ${tireData.tire_health_status.includes('CRITICAL') ? 'text-red-500 animate-pulse' : 'text-white'}`}>
                       {tireData.tire_health_status}
                     </div>
                     <div className="text-[10px] text-gray-400 font-mono mt-2 tracking-[0.2em] uppercase">Structural Integrity</div>
                     {/* Subtle background glow */}
                     <div className="absolute top-0 left-1/2 -translate-x-1/2 w-32 h-32 bg-green-500/10 blur-[50px] rounded-full pointer-events-none"></div>
                 </div>
                 
                 {/* Data Grid - Fills the space */}
                 <div className="grid grid-cols-2 gap-3">
                   <div className="bg-black/40 p-4 rounded border border-white/10 flex flex-col justify-between">
                     <div className="text-[9px] text-gray-500 font-bold mb-1 flex items-center gap-1"><TrendingUp size={10}/> PREDICTED LAP</div>
                     <div className="text-2xl font-mono text-white tracking-tighter">{tireData.predicted_next_lap}<span className="text-sm text-gray-600 ml-1">s</span></div>
                   </div>
                   <div className="bg-black/40 p-4 rounded border border-white/10 flex flex-col justify-between">
                     <div className="text-[9px] text-gray-500 font-bold mb-1 flex items-center gap-1"><Zap size={10}/> PACE DELTA</div>
                     <div className="text-2xl font-mono text-white tracking-tighter">{tireData.pace_drop_predicted}<span className="text-sm text-gray-600 ml-1">s</span></div>
                   </div>
                   <div className="bg-black/40 p-4 rounded border border-white/10 flex flex-col justify-between">
                     <div className="text-[9px] text-gray-500 font-bold mb-1 flex items-center gap-1"><Thermometer size={10}/> SURFACE TEMP</div>
                     <div className="text-2xl font-mono text-yellow-400 tracking-tighter">102<span className="text-sm text-gray-600 ml-1">°C</span></div>
                   </div>
                   <div className="bg-black/40 p-4 rounded border border-white/10 flex flex-col justify-between">
                     <div className="text-[9px] text-gray-500 font-bold mb-1 flex items-center gap-1"><Thermometer size={10}/> CORE TEMP</div>
                     <div className="text-2xl font-mono text-red-400 tracking-tighter">115<span className="text-sm text-gray-600 ml-1">°C</span></div>
                   </div>
                 </div>

                 {/* Tire Life Progress Bar */}
                 <div className="mt-auto">
                    <div className="flex justify-between text-[10px] font-bold text-gray-400 mb-2 uppercase tracking-wider">
                        <span>Est. Remaining Life</span>
                        <span className="text-white">12 Laps</span>
                    </div>
                    <div className="h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 w-[40%] rounded-full shadow-[0_0_10px_rgba(0,255,0,0.5)]"></div>
                    </div>
                 </div>
               </div>
             ) : (
               <div className="flex-1 flex items-center justify-center flex-col gap-4 text-gray-500 animate-pulse">
                   <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin"></div>
                   <span className="font-mono text-xs">Initializing Neural Network...</span>
               </div>
             )}
        </div>
      </div>

      {/* BOTTOM ROW: GHOST TELEMETRY (Full Width) */}
      <div className="glass-panel p-6 rounded-2xl h-[350px] border-l-4 border-yellow-500 relative overflow-hidden">
        {/* Background Grid */}
        <div className="absolute inset-0 opacity-10 pointer-events-none" 
             style={{backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '40px 40px'}}>
        </div>

        <div className="flex justify-between items-start mb-6 relative z-10">
            <div>
                <h3 className="font-black text-xl text-white italic tracking-tighter flex items-center gap-2">
                  TELEMETRY GHOST
                  <span className="text-xs font-normal not-italic text-gray-500 bg-black/50 px-2 py-0.5 rounded border border-white/10">SPEED TRACE</span>
                </h3>
                <p className="text-xs text-gray-500 font-mono mt-1">REAL-TIME INPUT COMPARISON (THROTTLE/BRAKE)</p>
            </div>
            
            <div className="flex items-center gap-6 bg-black/60 px-4 py-2 rounded-lg border border-white/10 backdrop-blur-sm">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-[#FF1801] rounded-sm shadow-[0_0_8px_#FF1801]"></div>
                    <span className="text-xs font-bold text-gray-300">VERSTAPPEN</span>
                </div>
                <div className="h-4 w-px bg-white/10"></div>
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-[#00D2BE] rounded-sm shadow-[0_0_8px_#00D2BE]"></div>
                    <span className="text-xs font-bold text-gray-300">LECLERC</span>
                </div>
            </div>
        </div>
        
        <div className="w-full h-[240px] relative z-10">
            <ResponsiveContainer width="100%" height="100%">
            <LineChart data={telemetryData}>
                <XAxis dataKey="distance" hide />
                <YAxis domain={['dataMin', 'dataMax']} hide />
                <Tooltip 
                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.9)', borderColor: '#333', color: '#fff', backdropFilter: 'blur(4px)' }} 
                    itemStyle={{ fontSize: '12px', fontFamily: 'monospace' }}
                    formatter={(value) => [`${value} km/h`, 'Speed']}
                    labelFormatter={() => ''}
                />
                <Line type="monotone" dataKey="speed1" stroke="#FF1801" dot={false} strokeWidth={3} activeDot={{ r: 6, fill: '#FF1801' }} />
                <Line type="monotone" dataKey="speed2" stroke="#00D2BE" dot={false} strokeWidth={3} strokeDasharray="4 4" activeDot={{ r: 6, fill: '#00D2BE' }} />
            </LineChart>
            </ResponsiveContainer>
        </div>
      </div>

    </div>
  );
};

export default Dashboard;