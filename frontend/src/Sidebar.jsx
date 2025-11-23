import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, GitGraph, Radio, Zap, Activity, Database } from 'lucide-react';

// Props: receive the state setters from App.jsx
const Sidebar = ({ currentLap, setCurrentLap, driver, setDriver }) => {
  const navClass = ({ isActive }) => 
    `flex items-center gap-3 p-3 rounded-lg mb-2 transition-all ${
      isActive 
        ? 'bg-mission-red text-white shadow-[0_0_15px_rgba(255,24,1,0.4)]' 
        : 'text-gray-400 hover:bg-white/5 hover:text-white'
    }`;

  return (
    <div className="w-72 bg-mission-card border-r border-mission-border h-screen p-6 fixed left-0 top-0 flex flex-col z-50 overflow-y-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-black italic tracking-tighter text-mission-text">
          F1 <span className="text-mission-red">INTEL</span>
        </h1>
        <div className="text-xs text-gray-500 font-mono">TEAM RED BULL</div>
      </div>

      {/* --- NEW: RACE CONFIGURATION CONTROLS --- */}
      <div className="mb-8 p-4 bg-mission-bg rounded-xl border border-mission-border">
        <h3 className="text-xs font-bold text-gray-400 mb-4 uppercase tracking-wider">Race Config</h3>
        
        <div className="mb-4">
            <label className="text-xs text-gray-500 block mb-1">DRIVER</label>
            <select 
                value={driver} 
                onChange={(e) => setDriver(e.target.value)}
                className="w-full bg-mission-card border border-mission-border text-mission-text p-2 rounded text-sm outline-none focus:border-mission-red"
            >
                <option value="VER">Max Verstappen</option>
                <option value="PER">Sergio Perez</option>
                <option value="LEC">Charles Leclerc</option>
                <option value="HAM">Lewis Hamilton</option>
                <option value="ALO">Fernando Alonso</option>
            </select>
        </div>

        <div className="mb-2">
            <div className="flex justify-between text-xs text-gray-500 mb-1">
                <span>CURRENT LAP</span>
                <span className="text-mission-red font-bold">{currentLap}/57</span>
            </div>
            <input 
                type="range" 
                min="1" max="57" 
                value={currentLap} 
                onChange={(e) => setCurrentLap(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-mission-red"
            />
        </div>
      </div>
      {/* --------------------------------------- */}

      <nav className="flex-1">
        <NavLink to="/" className={navClass}>
          <LayoutDashboard size={20} />
          <span>Mission Control</span>
        </NavLink>
        <NavLink to="/strategy" className={navClass}>
          <GitGraph size={20} />
          <span>Strategy Room</span>
        </NavLink>
        <NavLink to="/telemetry" className={navClass}>
            <Database size={20} />
            <span>Telemetry Lab</span>
        </NavLink>
        <NavLink to="/briefing" className={navClass}>
          <Radio size={20} />
          <span>Briefing Room</span>
        </NavLink>
      </nav>
    </div>
  );
};

export default Sidebar;