import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowRight, Cpu, Activity, Globe, Zap } from 'lucide-react';
import heroCar from '../assets/hero-car.jpg';

const HomePage = () => {
  return (
    <div className="min-h-screen bg-mission-bg text-white overflow-hidden">
      
      {/* TOP NAVIGATION */}
      <nav className="fixed top-0 w-full z-50 flex justify-between items-center px-10 py-6 bg-gradient-to-b from-black/60 to-transparent">
        <div className="text-2xl font-black italic tracking-tighter">
          F1 <span className="text-mission-red">INTEL</span>
        </div>
        <div className="hidden md:flex gap-8 text-sm font-bold text-gray-400 tracking-widest">
          <a href="#features" className="hover:text-white transition-colors">FEATURES</a>
          <a href="#tech" className="hover:text-white transition-colors">TECHNOLOGY</a>
          <a href="#team" className="hover:text-white transition-colors">TEAM</a>
        </div>
        <Link to="/dashboard" className="px-6 py-2 bg-white text-black font-bold rounded hover:bg-gray-200 transition-colors flex items-center gap-2">
          LAUNCH APP <ArrowRight size={16} />
        </Link>
      </nav>

      {/* HERO SECTION */}
      <section className="relative h-screen flex items-center px-10 pt-20">
        {/* Hero Background Image */}
        <div className="absolute inset-0 z-0">
            <img 
                src={heroCar} 
                className="w-full h-full object-cover opacity-50"
                alt="F1 Car"
            />
            <div className="absolute inset-0 bg-gradient-to-r from-mission-bg via-mission-bg/80 to-transparent"></div>
        </div>

        <div className="relative z-10 max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
          >
            <div className="flex items-center gap-2 text-mission-red font-mono text-sm tracking-[0.3em] mb-4">
                <span className="w-2 h-2 bg-mission-red rounded-full animate-pulse"></span>
                LIVE TELEMETRY SYSTEM
            </div>
            <h1 className="text-7xl md:text-9xl font-black italic tracking-tighter leading-[0.9] mb-6">
              DOMINATE <br/>
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-mission-red to-orange-600">THE GRID</span>
            </h1>
            <p className="text-xl text-gray-300 max-w-xl leading-relaxed mb-10 border-l-4 border-mission-red pl-6">
              Advanced race strategy powered by LSTM Neural Networks and XGBoost. 
              Analyze tire degradation, predict undercuts, and visualize track dominance in real-time.
            </p>
            
            <div className="flex gap-4">
                <Link to="/dashboard" className="px-8 py-4 bg-mission-red hover:bg-red-700 text-white font-bold rounded text-lg transition-all flex items-center gap-2 shadow-[0_0_30px_rgba(255,24,1,0.4)]">
                    ENTER MISSION CONTROL
                </Link>
                <button className="px-8 py-4 border border-white/20 hover:bg-white/10 text-white font-bold rounded text-lg transition-all backdrop-blur-md">
                    VIEW DOCUMENTATION
                </button>
            </div>
          </motion.div>
        </div>
      </section>

      {/* FEATURES SECTION */}
      <section id="features" className="py-32 px-10 relative">
        <div className="max-w-7xl mx-auto">
            <div className="mb-20">
                <h2 className="text-5xl font-black mb-4">ENGINEERED FOR <span className="text-mission-red">SPEED</span></h2>
                <div className="h-1 w-32 bg-mission-red"></div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {/* Feature 1 */}
                <div className="glass-panel p-8 rounded-2xl border-t-4 border-mission-red hover:-translate-y-2 transition-transform duration-300">
                    <div className="w-14 h-14 bg-mission-red/10 rounded-xl flex items-center justify-center mb-6 text-mission-red">
                        <Cpu size={32} />
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Machine Learning Core</h3>
                    <p className="text-gray-400 leading-relaxed">
                        Hybrid AI architecture combining LSTM for time-series prediction and XGBoost for regression analysis.
                    </p>
                </div>

                {/* Feature 2 */}
                <div className="glass-panel p-8 rounded-2xl border-t-4 border-mission-teal hover:-translate-y-2 transition-transform duration-300">
                    <div className="w-14 h-14 bg-mission-teal/10 rounded-xl flex items-center justify-center mb-6 text-mission-teal">
                        <Activity size={32} />
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Real-Time Strategy</h3>
                    <p className="text-gray-400 leading-relaxed">
                        Live calculation of pit windows, tire cliffs, and undercut probabilities using historical degradation data.
                    </p>
                </div>

                {/* Feature 3 */}
                <div className="glass-panel p-8 rounded-2xl border-t-4 border-yellow-500 hover:-translate-y-2 transition-transform duration-300">
                    <div className="w-14 h-14 bg-yellow-500/10 rounded-xl flex items-center justify-center mb-6 text-yellow-500">
                        <Globe size={32} />
                    </div>
                    <h3 className="text-2xl font-bold mb-3">Spatial Telemetry</h3>
                    <p className="text-gray-400 leading-relaxed">
                        Visualize track dominance with GPS-accurate mapping. See exactly where you gain time against rivals.
                    </p>
                </div>
            </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="py-10 border-t border-white/10 text-center text-gray-500 font-mono text-xs">
        <p>F1 INTEL © 2025 | POWERED BY FASTF1 & TENSORFLOW</p>
      </footer>
    </div>
  );
};

export default HomePage;