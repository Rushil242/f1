import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Clock, Trophy, Zap, ChevronRight } from 'lucide-react';
import Tire3D from '../components/Tire3D';

const StrategyRoom = ({ driver, currentLap }) => {
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    axios.post('/api/strategy/calculate', {
        driver: driver,
        current_lap: currentLap,
        tire_age: 12,
        compound: 'MEDIUM'
    }).then(res => {
        setStrategies(res.data);
        setLoading(false);
    });
  }, [driver, currentLap]);

  if (loading) return <div className="p-10 text-center text-gray-500 animate-pulse font-mono">CALCULATING PREDICTIONS...</div>;
  if (strategies.length === 0) return null;

  const best = strategies[0];

  return (
    <div className="p-6 md:p-10 animate-fade-in">
       <h2 className="text-4xl font-black italic tracking-tighter mb-8 text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-500">
           STRATEGIC <span className="text-[#FF1801]">INTELLIGENCE</span>
       </h2>
       
       <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            
            {/* LEFT: MAIN CARD (8 cols) */}
            <div className="lg:col-span-8 glass-panel p-8 rounded-3xl relative overflow-hidden border-l-4 border-[#FF1801]">
                 {/* Big Background Number */}
                 <div className="absolute -right-10 -bottom-20 text-[200px] font-black text-white/5 select-none">
                     {best.pit_lap}
                 </div>

                 <div className="relative z-10">
                     <div className="flex items-center gap-2 mb-6">
                        <Trophy className="text-[#FF1801]" size={20} />
                        <span className="text-[#FF1801] font-bold tracking-widest text-sm uppercase">AI Recommendation</span>
                     </div>

                     <div className="flex flex-col md:flex-row md:items-end gap-10">
                         <div>
                             <p className="text-gray-400 text-xs font-mono mb-1">STRATEGY</p>
                             <h3 className="text-5xl font-black text-white">{best.name.split('(')[0]}</h3>
                         </div>
                         <div>
                             <p className="text-gray-400 text-xs font-mono mb-1">BOX LAP</p>
                             <h3 className="text-5xl font-black text-white">L{best.pit_lap}</h3>
                         </div>
                         <div>
                             <p className="text-gray-400 text-xs font-mono mb-1">COMPOUND</p>
                             <h3 className={`text-5xl font-black ${best.compound === 'SOFT' ? 'text-red-500' : 'text-yellow-400'}`}>
                                 {best.compound}
                             </h3>
                         </div>
                     </div>

                     <div className="mt-8 flex items-center gap-3 bg-black/40 w-fit px-4 py-2 rounded-lg border border-white/5">
                         <Clock size={16} className="text-gray-400"/>
                         <span className="text-gray-300 text-sm">Estimated Race Time:</span>
                         <span className="text-white font-mono font-bold">{best.total_time}s</span>
                     </div>
                 </div>
            </div>

            {/* RIGHT: 3D TIRE (4 cols) */}
            <div className="lg:col-span-4 glass-panel p-6 rounded-3xl flex flex-col items-center justify-center relative">
                <div className="absolute top-4 left-4 text-xs font-bold text-gray-500 flex items-center gap-2">
                    <Zap size={14} className="text-yellow-500" /> LIVE MODEL
                </div>
                <div className="h-[250px] w-full flex items-center justify-center">
                    <Tire3D compound={best.compound} wear={15} />
                </div>
                <div className="text-center mt-2">
                    <div className="text-2xl font-bold text-white">{best.compound} C4</div>
                    <div className="text-xs text-gray-400 font-mono mt-1">SURFACE TEMP: 105°C</div>
                </div>
            </div>
       </div>

       {/* BOTTOM: ALTERNATIVES TABLE */}
       <div className="mt-8">
           <h3 className="text-gray-400 font-bold mb-4 text-sm uppercase tracking-wider pl-2">Alternative Scenarios</h3>
           <div className="glass-panel rounded-xl overflow-hidden">
               <table className="w-full text-left border-collapse">
                   <thead className="bg-white/5 text-gray-400 text-xs uppercase font-mono">
                       <tr>
                           <th className="p-4 font-normal">Strategy</th>
                           <th className="p-4 font-normal">Pit Window</th>
                           <th className="p-4 font-normal">Tire</th>
                           <th className="p-4 font-normal">Delta</th>
                       </tr>
                   </thead>
                   <tbody className="text-gray-200 text-sm font-mono">
                       {strategies.map((strat, i) => (
                           <tr key={i} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                               <td className="p-4 font-sans font-bold flex items-center gap-3">
                                   {i === 0 && <ChevronRight size={16} className="text-[#FF1801]" />}
                                   {strat.name}
                               </td>
                               <td className="p-4 text-gray-400">Lap {strat.pit_lap}</td>
                               <td className="p-4">
                                   <span className={`px-2 py-1 rounded text-xs font-bold bg-black/30 border border-white/10 ${strat.compound === 'SOFT' ? 'text-red-400' : 'text-yellow-400'}`}>
                                       {strat.compound}
                                   </span>
                               </td>
                               <td className={`p-4 font-bold ${i===0 ? 'text-green-400' : 'text-red-400'}`}>
                                   {i === 0 ? 'OPTIMAL' : `+${(strat.total_time - best.total_time).toFixed(2)}s`}
                               </td>
                           </tr>
                       ))}
                   </tbody>
               </table>
           </div>
       </div>
    </div>
  );
};

export default StrategyRoom;