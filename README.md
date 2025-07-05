# ⚡ Electricity Demand Forecasting using LSTM + Attention

This project is a deep learning-based system to **forecast daily electricity demand and peak usage** using historical power consumption data, weather conditions, and temporal features. The system leverages **LSTM networks**, **attention mechanisms**, and **Keras Tuner** for hyperparameter optimization to enhance predictive performance.

---

## 📌 Problem Statement

Electric utilities and grid operators need accurate forecasting of electricity demand to:
- Ensure reliable power supply
- Prevent blackouts and overloading
- Optimize power generation and costs

This model solves that by predicting:
- **Total Demand** (kWh)
- **Peak Demand** (kWh)

...for any user-provided date range, using the past 30 days' data.

---

## 📁 Dataset Overview

**Source**: `power_demand_dataset.csv`  
**Duration**: Jan 2021 – mid-2024  
**Granularity**: 5-minute intervals  
**Features**:
- `Power demand` – actual power usage
- `temp` – temperature
- `rhum` – relative humidity

The data was resampled to **daily** format and engineered with additional features:
- Daily **total demand**
- Daily **peak demand**
- **Mean temperature** and **humidity**
- **Weekday index (0–6)**
- **Lag features**: Lag1, Lag2
- **Rolling averages**: 7-day, 14-day
- **Month** (seasonal effect)

---

## 🧠 Model Architecture

The final model is built with the following components:

- 🔁 **LSTM layer (128 units)**: Captures time dependencies in demand patterns
- 🌐 **Bidirectional LSTM**: Learns from past and future context
- 🧲 **Attention layer**: Focuses on important time steps to improve relevance
- 🧪 **Dropout (0.3)**: Prevents overfitting
- 🧮 **Dense layer**: Outputs 2 values → Total Demand & Peak Demand

> Trained with `mean squared error` loss and `Adam` optimizer.

---

## 🛠️ Tech Stack

- **Modeling**: TensorFlow / Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib
- **Hyperparameter Tuning**: Keras Tuner
- **Training**: Google Colab (GPU runtime)
- **Deployment (in progress)**: Flask + React

---

## 📈 Visualizations


- 📅 Time-series forecast plots of Total Demand and Peak Demand

---

## 🚀 Prediction Flow

1. The model reads the **last 30 days** of scaled daily data.
2. User inputs: `start_date`, `end_date`
3. For each day in that range:
   - It appends predictions back into the sequence
   - Uses last temperature, humidity, weekday info for future context
4. Final results are **inverse-transformed** and displayed as daily forecasts.

---

## 💡 How to Use

> 🔧 You can run the entire notebook:  
[`Electricity_demand_Forecasting.ipynb`](notebooks/Electricity_demand_Forecasting.ipynb)

---

## 🌍 Real-World Applications

- Smart grid systems
- Energy trading and pricing
- Demand-response automation
- City-level infrastructure planning
- Seasonal and weather-based power optimization

---

## 🗺️ Future Work

- 🌪️ Integrate more weather features (e.g. wind, precipitation)
- 🧠 Use transformer-based time series models (e.g. Temporal Fusion Transformer)
- 🌐 Fully deploy using Flask backend + React frontend
- 📡 Connect to live API for real-time weather and demand feeds

---



