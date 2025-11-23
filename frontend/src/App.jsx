import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import TopBar from './TopBar'; // NEW IMPORT

// Pages
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import Dashboard from './pages/Dashboard';
import StrategyRoom from './pages/StrategyRoom';
import TelemetryLab from './pages/TelemetryLab';
import RadioControl from './RadioControl';

// Layout for the Dashboard (Top Bar + Content)
const DashboardLayout = ({ driver }) => (
  <div className="min-h-screen bg-mission-bg text-mission-text pt-16"> {/* Added pt-16 for topbar */}
    <TopBar />
    <div className="w-full">
      <Outlet />
    </div>
  </div>
);

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  
  // HARDCODED DRIVER - Removed Inputs
  const driver = 'VER'; 
  const currentLap = 25; // Fixed lap for demo

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={setIsAuthenticated} />} />
        <Route path="/" element={<HomePage />} />

        <Route element={isAuthenticated ? <DashboardLayout driver={driver} /> : <Navigate to="/login" />}>
            <Route path="/dashboard" element={<Dashboard driver={driver} />} />
            <Route path="/strategy" element={<StrategyRoom driver={driver} currentLap={currentLap} />} />
            <Route path="/telemetry" element={<TelemetryLab driver={driver} />} />
            <Route path="/briefing" element={<div className="p-8 h-screen flex flex-col items-center justify-center"><h2 className="text-4xl font-bold mb-8 text-mission-text">STRATEGIC BRIEFING</h2><div className="w-full max-w-2xl"><RadioControl driver={driver} /></div></div>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;