import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer 
} from 'recharts';
import { Microscope, TrendingUp, Zap, Target } from 'lucide-react';

const TelemetryLab = ({ driver }) => {
  const [radarData, setRadarData] = useState([]);
  const [forecastData, setForecastData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Fetch Technical Features
        const radarRes = await axios.get('/api/performance/radar?d1=VER&d2=LEC');
        setRadarData(radarRes.data);

        const forecastRes = await axios.get('/api/battle/forecast?d1=VER&d2=LEC');
        setForecastData(forecastRes.data);
      } catch (e) { console.error(e); }
    };
    fetchData();
  }, []);

  return (
    <div className="p-6 md:p-10 animate-fade-in space-y-8">
      
      {/* HEADER */}
      <div className="flex items-center justify-between border-b border-white/10 pb-6">
        <div>
            <h2 className="text-3xl font-black italic tracking-tighter text-white">
                TELEMETRY <span className="text-mission-red">LAB</span>
            </h2>
            <p className="text-sm text-gray-500 font-mono mt-1">
                ADVANCED PHYSICS ANALYSIS & PREDICTIVE MODELING
            </p>
        </div>
        <div className="flex gap-4">
            <div className="flex items-center gap-2 text-xs font-bold bg-[#FF1801]/10 px-3 py-1 rounded text-[#FF1801] border border-[#FF1801]/20">
                <div className="w-2 h-2 bg-[#FF1801] rounded-full"></div> VERSTAPPEN
            </div>
            <div className="flex items-center gap-2 text-xs font-bold bg-[#00D2BE]/10 px-3 py-1 rounded text-[#00D2BE] border border-[#00D2BE]/20">
                <div className="w-2 h-2 bg-[#00D2BE] rounded-full"></div> LECLERC
            </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* FEATURE 1: PERFORMANCE DNA RADAR */}
        <div className="glass-panel p-8 rounded-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
                <Microscope size={100} />
            </div>
            <div className="mb-6">
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <Zap className="text-yellow-400" size={20}/> 
                    PERFORMANCE DNA
                </h3>
                <p className="text-xs text-gray-400 font-mono mt-1">
                    CLUSTERING ALGORITHM ANALYSIS (SECTOR SEGMENTATION)
                </p>
            </div>
            
            <div className="h-[350px] w-full flex justify-center items-center">
                <ResponsiveContainer width="100%" height="100%">
                    <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                        <PolarGrid stroke="#333" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#888', fontSize: 10, fontWeight: 'bold' }} />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                        <Radar name="Verstappen" dataKey="driver_1" stroke="#FF1801" strokeWidth={3} fill="#FF1801" fillOpacity={0.4} />
                        <Radar name="Leclerc" dataKey="driver_2" stroke="#00D2BE" strokeWidth={3} fill="#00D2BE" fillOpacity={0.4} />
                        <Tooltip contentStyle={{ backgroundColor: '#000', borderColor: '#333' }} itemStyle={{fontSize: 12}} />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
            <div className="text-center text-xs text-gray-500 mt-2 font-mono">
                NORMALIZED PERFORMANCE INDEX (0-100)
            </div>
        </div>

        {/* FEATURE 2: PREDICTIVE GAP FORECAST */}
        <div className="glass-panel p-8 rounded-2xl relative overflow-hidden border-l-4 border-mission-red">
            <div className="absolute top-0 right-0 p-4 opacity-10">
                <TrendingUp size={100} />
            </div>
            <div className="mb-6">
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <Target className="text-mission-red" size={20}/> 
                    THE CATCH PREDICTOR
                </h3>
                <p className="text-xs text-gray-400 font-mono mt-1">
                    RECURSIVE XGBOOST GAP FORECAST (NEXT 15 LAPS)
                </p>
            </div>

            <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={forecastData}>
                        <defs>
                            <linearGradient id="colorGap" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#FF1801" stopOpacity={0.3}/>
                                <stop offset="95%" stopColor="#FF1801" stopOpacity={0}/>
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                        <XAxis dataKey="lap" stroke="#666" tick={{fontSize: 10}} />
                        <YAxis stroke="#666" label={{ value: 'Gap (s)', angle: -90, position: 'insideLeft', fill: '#666' }} />
                        <Tooltip 
                            contentStyle={{ backgroundColor: '#000', borderColor: '#333' }}
                            labelStyle={{ color: '#888' }}
                        />
                        {/* Reference Line for Overtake */}
                        <Area 
                            type="monotone" 
                            dataKey="gap" 
                            stroke="#FF1801" 
                            strokeWidth={3}
                            fillOpacity={1} 
                            fill="url(#colorGap)" 
                        />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
            <div className="mt-4 flex justify-between items-center bg-white/5 p-3 rounded border border-white/10">
                <div className="text-xs text-gray-400 font-bold">PREDICTED INTERCEPT</div>
                <div className="text-xl font-mono font-black text-white">
                    {forecastData.find(d => d.gap <= 0)?.lap || "NO CATCH"} <span className="text-xs text-gray-500 font-normal">LAPS</span>
                </div>
            </div>
        </div>

      </div>
    </div>
  );
};

export default TelemetryLab;