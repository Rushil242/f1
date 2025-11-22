/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // <--- 1. ENABLE THIS
  theme: {
    extend: {
      colors: {
        mission: {
          // 2. USE VARIABLES INSTEAD OF HEX CODES
          bg: 'var(--mission-bg)',      
          card: 'var(--mission-card)',  
          text: 'var(--mission-text)',
          border: 'var(--mission-border)',
          
          // Keep these static (they look good in both modes)
          red: '#FF1801',     
          teal: '#00D2BE',    
          green: '#00ff00',   
        }
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', "Liberation Mono", "Courier New", 'monospace'],
      }
    },
  },
  plugins: [],
}