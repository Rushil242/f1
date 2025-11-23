# 🏎️ F1 MISSION CONTROL: AI-Powered Race Strategy Platform

**A Next-Generation Decision Support System for Formula 1 Strategy, powered by Hybrid AI (LSTM + XGBoost) and Generative Agents.**

![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20Tailwind-blue)
![Python](https://img.shields.io/badge/Backend-Flask%20%7C%20Python-yellow)
![AI](https://img.shields.io/badge/AI-TensorFlow%20%7C%20XGBoost%20%7C%20Gemini-orange)
![Data](https://img.shields.io/badge/Data-FastF1%20%7C%20Ergast-red)

## 🚀 Project Overview

This is not just a dashboard; it is a comprehensive **Digital Twin** of an F1 Race Strategy environment. Unlike traditional static analysis tools, this project leverages **Deep Learning (LSTM)** for time-series forecasting, **XGBoost** for race simulations, and **Google Gemini** for natural language tactical briefings.

It transforms raw telemetry data into actionable intelligence, presented through a cinematic, high-performance **React** interface inspired by real-world Mission Control screens.

### 🌟 Key Capabilities (The "Novelty")

* **🧠 Hybrid AI Architecture**: Combines **LSTM Neural Networks** (for non-linear tire degradation prediction) with **XGBoost** (for recursive gap evolution forecasting).
* **🗣️ Generative AI Race Engineer**: A fully autonomous Voice Agent that analyzes telemetry JSON and speaks strategic commands using **Google Gemini 1.5 Pro** and Text-to-Speech.
* **🧊 3D & Spatial Visualization**: Interactive **WebGL 3D Tire Models** (React Three Fiber) and **Animated Track Maps** with real-time physics simulation.
* **🔬 Unsupervised Machine Learning**: Uses **K-Means Clustering** to segment track telemetry into "Performance DNA" (identifying mechanical vs. aerodynamic grip).

---

## 📸 System Modules

### 1. Mission Control (Live Dashboard)
The central hub featuring a **Neon-Glassmorphism UI**.
* **Live Battle Map**: SVG-based track visualization with real-time car interpolation and dominance coloring (Red vs. Teal).
* **Ghost Telemetry**: Speed trace overlays comparing driver inputs (Throttle/Brake) down to the meter.
* **LSTM Tire Health**: A "Sentient" tire model that predicts the exact lap a tire will hit the "Cliff."

### 2. Strategy Room (Simulation Engine)
* **XGBoost Projector**: Simulates the race 30 laps into the future to visualize "Undercut" and "Overcut" scenarios.
* **3D Tire Lab**: A rotating 3D model representing physical tire wear and surface temperature in real-time.

### 3. Telemetry Lab (Data Science)
* **Performance DNA Radar**: Uses K-Means clustering to categorize track sectors and compare drivers on 5 dimensions (Traction, Braking, Aero, etc.).
* **Recursive Gap Forecast**: Predicts "Laps to Catch" by feeding model outputs back into inputs recursively.

---

## 🏗️ Technical Architecture

The system follows a decoupled **Microservices** pattern:

```mermaid
graph LR
    A[FastF1 API] --> B(Python Backend / Flask)
    B --> C{AI Engine}
    C --> D[LSTM Model - Time Series]
    C --> E[XGBoost - Regression]
    C --> F[K-Means - Clustering]
    B --> G[Gemini LLM Agent]
    G --> H[TTS Audio Engine]
    B --> I[React Frontend / Vite]
    I --> J[Three.js / Recharts / Framer Motion]