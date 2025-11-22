import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts';

const TrackMap = ({ data }) => {
  if (!data) return <div className="text-gray-500">Loading Track Data...</div>;

  // Split data into two drivers for coloring
  const driver1Data = data.filter(p => p.color === '#FF1801'); // Usually Max
  const driver2Data = data.filter(p => p.color !== '#FF1801'); // Usually Rival

  return (
    <div className="h-[400px] w-full bg-mission-card rounded-xl p-4 border border-gray-800 relative overflow-hidden">
      <h2 className="text-lg font-bold mb-2 flex items-center gap-2">
        <span className="w-3 h-3 rounded-full bg-mission-red animate-pulse"></span>
        TRACK DOMINANCE MAP
      </h2>
      
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          {/* Hide Axes for clean map look */}
          <XAxis dataKey="x" type="number" hide domain={['dataMin', 'dataMax']} />
          <YAxis dataKey="y" type="number" hide domain={['dataMin', 'dataMax']} />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ backgroundColor: '#000', border: '1px solid #333' }}
          />
          
          {/* Driver 1 (Red) */}
          <Scatter name="Verstappen" data={driver1Data} fill="#FF1801" line={{ stroke: '#FF1801', strokeWidth: 2 }} shape="circle" />
          
          {/* Driver 2 (Teal) */}
          <Scatter name="Rival" data={driver2Data} fill="#00D2BE" line={{ stroke: '#00D2BE', strokeWidth: 2 }} shape="circle" />
        </ScatterChart>
      </ResponsiveContainer>
      
      <div className="absolute bottom-4 right-4 text-xs text-gray-400 bg-black/50 p-2 rounded">
        <span className="text-mission-red font-bold">RED:</span> Verstappen Faster <br/>
        <span className="text-mission-teal font-bold">TEAL:</span> Rival Faster
      </div>
    </div>
  );
};

export default TrackMap;