import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const StrategyPage = () => {
  const [data, setData] = useState([]);

  useEffect(() => {
    // Fetch the ML Projections
    axios.get('/api/strategy/projection?driver=VER').then(res => {
      // Transform data for Recharts (Merge the 3 arrays into one)
      // Assuming res.data[0]=Soft, [1]=Medium, [2]=Hard
      const soft = res.data[0];
      const medium = res.data[1];
      const hard = res.data[2];
      
      const merged = soft.map((point, i) => ({
        lap: point.lap,
        soft: point.time,
        medium: medium[i]?.time,
        hard: hard[i]?.time
      }));
      setData(merged);
    });
  }, []);

  return (
    <div className="p-8">
      <h2 className="text-3xl font-bold mb-6 text-mission-text">ML STRATEGY PROJECTION</h2>
      <div className="bg-mission-card p-6 rounded-xl border border-mission-border h-[500px]">
        <h3 className="text-gray-400 mb-4">PREDICTED PACE (NEXT 20 LAPS)</h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="lap" label={{ value: 'Lap Number', position: 'insideBottom', offset: -5 }} />
            <YAxis domain={['auto', 'auto']} label={{ value: 'Lap Time (s)', angle: -90, position: 'insideLeft' }} />
            <Tooltip contentStyle={{ backgroundColor: '#111' }} />
            <Legend verticalAlign="top" height={36}/>
            <Line type="monotone" dataKey="soft" stroke="#FF1801" name="Soft Compound" strokeWidth={3} />
            <Line type="monotone" dataKey="medium" stroke="#FFD700" name="Medium Compound" strokeWidth={3} />
            <Line type="monotone" dataKey="hard" stroke="#FFFFFF" name="Hard Compound" strokeWidth={3} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="bg-mission-card p-4 border-l-4 border-red-500 rounded">
          <div className="font-bold text-red-500">SOFT</div>
          <div className="text-sm text-gray-400">Fastest degradation. Good for 12 laps.</div>
        </div>
        <div className="bg-mission-card p-4 border-l-4 border-yellow-500 rounded">
          <div className="font-bold text-yellow-500">MEDIUM</div>
          <div className="text-sm text-gray-400">Optimal race tire. Balanced pace.</div>
        </div>
        <div className="bg-mission-card p-4 border-l-4 border-white rounded">
          <div className="font-bold text-white">HARD</div>
          <div className="text-sm text-gray-400">Slow warm up. Zero degradation.</div>
        </div>
      </div>
    </div>
  );
};

export default StrategyPage;