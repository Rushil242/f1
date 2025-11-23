import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, GitGraph, Radio, Database, Zap } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const TopBar = () => {
  const navClass = ({ isActive }) => 
    `flex items-center gap-2 px-4 py-2 rounded-lg transition-all text-sm font-bold tracking-wider ${
      isActive 
        ? 'bg-mission-red text-white shadow-[0_0_15px_rgba(255,24,1,0.4)]' 
        : 'text-gray-400 hover:text-white hover:bg-white/5'
    }`;

  return (
    <div className="w-full h-16 bg-black/80 backdrop-blur-md border-b border-white/10 flex items-center justify-between px-6 fixed top-0 left-0 z-50">
      {/* LOGO */}
      <div className="flex items-center gap-4">
        <div className="text-xl font-black italic tracking-tighter text-white">
          F1 <span className="text-mission-red">INTEL</span>
        </div>
        <div className="h-6 w-px bg-white/10 mx-2"></div>
        <div className="text-xs font-mono text-mission-red flex items-center gap-1">
            <Zap size={12} /> TEAM RED BULL
        </div>
      </div>

      {/* NAVIGATION */}
      <nav className="flex items-center gap-2">
        <NavLink to="/dashboard" className={navClass}>
          <LayoutDashboard size={16} /> MISSION CONTROL
        </NavLink>
        <NavLink to="/strategy" className={navClass}>
          <GitGraph size={16} /> STRATEGY
        </NavLink>
        <NavLink to="/telemetry" className={navClass}>
            <Database size={16} /> TELEMETRY
        </NavLink>
        <NavLink to="/briefing" className={navClass}>
          <Radio size={16} /> BRIEFING
        </NavLink>
      </nav>

      {/* RIGHT ACTIONS */}
      <div className="flex items-center gap-4">
        <div className="hidden md:block text-right">
            <div className="text-[10px] text-gray-500 font-mono">CURRENT SESSION</div>
            <div className="text-xs font-bold text-white">BAHRAIN GP 2023</div>
        </div>
        <ThemeToggle />
      </div>
    </div>
  );
};

export default TopBar;