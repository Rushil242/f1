import React, { useMemo } from 'react';
import { motion } from 'framer-motion';

// High-detail Top-Down F1 Car SVG (Correctly oriented)
const F1CarSVG = ({ color, scale = 1 }) => (
  <svg width="40" height="40" viewBox="0 0 512 512" style={{ transform: `scale(${scale})` }}>
    {/* A simpler, clearer car shape for movement */}
    <path 
      fill={color} 
      stroke="black" strokeWidth="5"
      d="M146.6 137.4c-3.7-13.8-16.2-23.6-30.5-24-14.4-.4-27.2 8.7-31.6 22.4l-22.1 68.6c-5.2 16.1-22.3 25.2-38.7 20.6l-1.4-.4c-12.3-3.5-25.2 3.7-28.7 16l-3.2 11.3c-3.5 12.3 3.7 25.2 16 28.7l1.4.4c16.4 4.6 26.2 21.6 22.3 37.9l-16.6 68.9c-3.1 12.7 4.6 25.6 17.3 28.7l11.3 2.7c12.7 3.1 25.6-4.6 28.7-17.3l15.8-65.6c3.9-16.3 20.8-26.2 37.2-21.9l6.3 1.6c36.3 9.4 73.7-12.5 82.9-48.6l2.6-10.1c1.8-6.9 4.9-13.3 9.1-19l19.6-26.6c7.6-10.3 5.5-24.8-4.8-32.4l-26.2-19.4c-10.3-7.6-24.8-5.5-32.4 4.8l-7.8 10.6c-6.3 8.6-17.1 12.6-27.4 10.1l-2.6-.7z"
    />
    {/* Rear Wing for better orientation view */}
    <rect x="0" y="50" width="100" height="20" fill={color} stroke="black" strokeWidth="5" />
  </svg>
);

const TrackMap = ({ data }) => {
  // 1. Data Normalization (Fits track to the container)
  const normalizedPoints = useMemo(() => {
    if (!data || data.length === 0) return [];
    const xs = data.map(p => p.x);
    const ys = data.map(p => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padding = 1500; 
    
    return data.map(p => ({
      ...p,
      // Map to 0-100 percentage for CSS positioning
      xPct: ((p.x - minX + padding) / (maxX - minX + padding * 2)) * 100,
      yPct: ((p.y - minY + padding) / (maxY - minY + padding * 2)) * 100,
      // Map to 1000x1000 coordinate system for SVG drawing
      svgX: ((p.x - minX + padding) / (maxX - minX + padding * 2)) * 1000,
      svgY: 1000 - ((p.y - minY + padding) / (maxY - minY + padding * 2)) * 1000
    }));
  }, [data]);

  // 2. Create the main Path string for the base road
  const trackPathStr = useMemo(() => {
    if(normalizedPoints.length === 0) return "";
    return "M " + normalizedPoints.map(p => `${p.svgX},${p.svgY}`).join(" L ") + " Z";
  }, [normalizedPoints]);

  // 3. Separate paths for animation
  // We take every Nth point to make the animation smoother, otherwise Framer Motion chokes on 500+ keyframes
  const SAMPLE_RATE = 5; 
  const verPath = normalizedPoints.filter((_, i) => i % SAMPLE_RATE === 0);
  const lecPath = normalizedPoints.filter((_, i) => i % SAMPLE_RATE === 0);

  return (
    <div className="glass-panel w-full h-[500px] rounded-2xl relative overflow-hidden bg-[#0a0a0a]">
      
      {/* --- LAYER 1: THE REAL ROAD SVG --- */}
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1000 1000" preserveAspectRatio="none">
        {/* A. Base Asphalt (Dark Grey) */}
        <path d={trackPathStr} fill="none" stroke="#222222" strokeWidth="35" strokeLinecap="round" strokeLinejoin="round" />
        
        {/* B. Dominance Coloring (The core feature) */}
        {/* We draw line segments between points, colored by who is faster there */}
        {normalizedPoints.map((p, i) => {
            if (i === 0) return null;
            const prev = normalizedPoints[i-1];
            return (
                <line 
                    key={i} 
                    x1={prev.svgX} y1={prev.svgY} 
                    x2={p.svgX} y2={p.svgY} 
                    stroke={p.color} // Red or Teal based on data
                    strokeWidth="20" // Slightly narrower than asphalt
                    strokeOpacity="0.8" 
                    strokeLinecap="butt"
                />
            );
        })}

        {/* C. Center Line markings (Dashed White) */}
        <path d={trackPathStr} fill="none" stroke="#ffffff50" strokeWidth="2" strokeDasharray="8, 12" />
      </svg>

      {/* --- LAYER 2: ANIMATED CARS --- */}
      <div className="absolute inset-0 pointer-events-none">
        {/* VERSTAPPEN (Red) */}
        <motion.div
            className="absolute z-20 will-change-transform"
            // We animate 'left' and 'top' (inverted bottom)
            // Using translate(-50%, -50%) centers the icon exactly on the point
            style={{ transform: 'translate(-50%, -50%)' }}
            animate={{
                left: verPath.map(p => `${p.xPct}%`),
                top: verPath.map(p => `${100 - p.yPct}%`), // Invert Y for CSS 'top'
            }}
            // Max is slightly faster overall (15s lap)
            transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
        >
             {/* Rotate car to roughly face driving direction (simplified for demo) */}
            <div style={{ transform: 'rotate(90deg)' }}>
                <F1CarSVG color="#FF1801" scale={1} />
            </div>
        </motion.div>

        {/* LECLERC (Teal) */}
        <motion.div
            className="absolute z-10 will-change-transform"
            style={{ transform: 'translate(-50%, -50%)' }}
            animate={{
                left: lecPath.map(p => `${p.xPct}%`),
                top: lecPath.map(p => `${100 - p.yPct}%`),
            }}
            // Leclerc is slightly slower (15.5s lap) to show the "battle"
            transition={{ duration: 15.5, repeat: Infinity, ease: "linear" }}
        >
            <div style={{ transform: 'rotate(90deg)' }}>
                <F1CarSVG color="#00D2BE" scale={1} />
            </div>
        </motion.div>
      </div>
      
      {/* OVERLAY HEADERS */}
      <div className="absolute top-6 left-6 z-30">
          <div className="flex items-center gap-2 mb-1">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-ping"></div>
            <h2 className="text-lg font-black italic tracking-tighter text-white">LIVE BATTLE</h2>
          </div>
          <div className="text-xs text-gray-400 font-mono">LAP 25 • SECTOR 2</div>
      </div>

      {/* LEGEND (Explains the track colors) */}
      <div className="absolute bottom-4 right-4 z-30 bg-black/80 backdrop-blur-md p-3 rounded-lg border border-white/10 flex flex-col gap-2">
        <div className="flex items-center gap-2">
            <div className="w-8 h-2 bg-[#FF1801] rounded-full"></div>
            <span className="text-[10px] font-bold text-gray-300">VER FASTER</span>
        </div>
        <div className="flex items-center gap-2">
            <div className="w-8 h-2 bg-[#00D2BE] rounded-full"></div>
            <span className="text-[10px] font-bold text-gray-300">LEC FASTER</span>
        </div>
      </div>

    </div>
  );
};

export default TrackMap;