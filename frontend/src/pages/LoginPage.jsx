import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Lock, ChevronRight, Zap } from 'lucide-react';

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = (e) => {
    e.preventDefault();
    if (email === 'student' && password === 'student@123') {
      onLogin(true);
      navigate('/'); // Go to Home
    } else {
      setError('ACCESS DENIED: INVALID CREDENTIALS');
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden flex items-center justify-center bg-black">
      {/* BACKGROUND VIDEO/IMAGE */}
      <div className="absolute inset-0 z-0">
        <img 
          src="https://images.unsplash.com/photo-1516550893923-42d28e5677af?q=80&w=2072&auto=format&fit=crop" 
          alt="Background" 
          className="w-full h-full object-cover opacity-40"
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black via-transparent to-black"></div>
      </div>

      {/* SCANLINE EFFECT */}
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 z-10"></div>

      {/* LOGIN CARD */}
      <motion.div 
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: "easeOut" }}
        className="relative z-20 w-full max-w-md p-8 glass-panel rounded-2xl border-l-4 border-mission-red shadow-[0_0_50px_rgba(255,24,1,0.2)]"
      >
        <div className="mb-8 text-center">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-mission-red/20 text-mission-red mb-4">
            <Lock size={24} />
          </div>
          <h1 className="text-3xl font-black tracking-tighter text-white">
            F1 <span className="text-mission-red">INTEL</span>
          </h1>
          <p className="text-xs font-mono text-gray-500 tracking-[0.2em] mt-2">RESTRICTED ACCESS // TEAM RED BULL</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label className="block text-xs font-bold text-gray-400 mb-2 ml-1">OPERATOR ID</label>
            <input 
              type="text" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-black/50 border border-white/10 rounded-lg p-4 text-white outline-none focus:border-mission-red focus:shadow-[0_0_15px_rgba(255,24,1,0.3)] transition-all"
              placeholder="Enter ID"
            />
          </div>
          
          <div>
            <label className="block text-xs font-bold text-gray-400 mb-2 ml-1">ACCESS KEY</label>
            <input 
              type="password" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-black/50 border border-white/10 rounded-lg p-4 text-white outline-none focus:border-mission-red focus:shadow-[0_0_15px_rgba(255,24,1,0.3)] transition-all"
              placeholder="••••••••"
            />
          </div>

          {error && (
            <motion.div 
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="text-red-500 text-xs font-bold flex items-center gap-2 bg-red-500/10 p-3 rounded border border-red-500/20"
            >
              <Zap size={14} /> {error}
            </motion.div>
          )}

          <button 
            type="submit"
            className="w-full bg-mission-red hover:bg-red-600 text-white font-bold py-4 rounded-lg flex items-center justify-center gap-2 transition-all hover:tracking-widest group"
          >
            AUTHENTICATE <ChevronRight size={18} className="group-hover:translate-x-1 transition-transform" />
          </button>
        </form>

        <div className="mt-6 text-center">
          <p className="text-[10px] text-gray-600 font-mono">
            SECURE CONNECTION ESTABLISHED <br/>
            LATENCY: 12ms | ENCRYPTION: AES-256
          </p>
        </div>
      </motion.div>
    </div>
  );
};

export default LoginPage;