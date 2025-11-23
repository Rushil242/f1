import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ScatterChart, Scatter } from 'recharts';

const TelemetryLab = ({ driver }) => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    axios.get(`/api/historical?driver=${driver}`).then(res => setHistory(res.data));
  }, [driver]);

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center">
        <h2 className="text-3xl font-bold text-mission-text">TELEMETRY LAB</h2>
        <div className="px-4 py-2 bg-mission-card border border-mission-border rounded text-sm text-gray-400">
            Analysis Subject: <span className="text-mission-red font-bold">{driver}</span>
        </div>
      </div>

      {/* CHART 1: LAP TIME EVOLUTION */}
      <div className="bg-mission-card p-6 rounded-xl border border-mission-border h-[400px]">
        <h3 className="text-gray-400 mb-4 font-bold flex items-center gap-2">
            📊 LAP TIME EVOLUTION
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis dataKey="LapNumber" stroke="#666" />
            <YAxis domain={['auto', 'auto']} stroke="#666" />
            <Tooltip contentStyle={{ backgroundColor: '#111', borderColor: '#333' }} />
            <Line type="monotone" dataKey="LapTime" stroke="#00D2BE" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* CHART 2: TIRE DEGRADATION SCATTER */}
      <div className="bg-mission-card p-6 rounded-xl border border-mission-border h-[400px]">
        <h3 className="text-gray-400 mb-4 font-bold flex items-center gap-2">
            📉 TIRE WEAR ANALYSIS (Cleaned Data)
        </h3>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" strokeOpacity={0.1} />
            <XAxis dataKey="TyreLife" type="number" name="Tire Age" stroke="#666" label={{ value: 'Laps on Tire', position: 'insideBottom', offset: -10 }} />
            <YAxis dataKey="LapTime" type="number" name="Lap Time" stroke="#666" domain={['auto', 'auto']} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#111' }} />
            <Scatter name="Pace" data={history} fill="#FF1801" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default TelemetryLab;