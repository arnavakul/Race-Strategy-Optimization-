# 🏎️ Race Strategy Optimization System

An AI-driven Formula 1 race strategy and lap time analysis platform built for predictive analytics, race simulation, and strategy optimization using historical telemetry and lap data.

---

# 📌 Project Overview

The **Race Strategy Optimization System** is designed to analyze Formula 1 race data and simulate race outcomes using machine learning, telemetry analysis, and predictive modeling.

The system focuses on:

* Predicting normalized lap times
* Strategy optimization based on tire degradation
* Race simulation using historical data
* Driver and constructor performance analysis
* Telemetry and stint-based insights
* Future race pace forecasting

This project combines:

* Data Engineering
* Machine Learning
* Motorsport Analytics
* Backend API Development
* Visualization & Simulation

---

# 🚀 Features

## 📊 Data Processing

* Fetches Formula 1 session data using FastF1
* Cleans and normalizes lap time datasets
* Stores processed data in efficient parquet format
* Handles multi-season race datasets

## 🧠 Machine Learning

* Lap time prediction models
* Feature engineering for tire wear and fuel load
* Track-specific pace modeling
* Driver consistency analysis
* Future stint pace estimation

## 🏁 Race Strategy Simulation

* Pit stop strategy comparison
* Undercut / overcut analysis
* Tire degradation simulation
* Stint performance prediction
* Race pace forecasting

## 📈 Analytics & Visualization

* Interactive telemetry analysis
* Lap time trend visualization
* Driver comparison charts
* Track evolution analysis
* Sector performance analytics

## ⚙️ Backend Services

* REST API support
* Data pipelines for ingestion and preprocessing
* Modular architecture for experimentation

---

# 🧱 Tech Stack

## Languages

* Python
* SQL

## Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* FastF1
* Matplotlib
* Seaborn
* XGBoost / LightGBM (optional)

## Data Storage

* Parquet Files
* PostgreSQL / MongoDB (optional)

## Tools

* Jupyter Notebook
* Git & GitHub
* VS Code

---

# 📂 Project Structure

```bash
Race-Strategy-Optimization-System/
│
├── backend/
│
│   ├── api/
│   │
│   │   ├── cache/
│   │   │
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── processed/
│   │   │   ├── telemetry/
│   │   │   ├── weather/
│   │   │   └── pitstop_data/
│   │   │
│   │   ├── features/
│   │   │   ├── lap_times.ipynb
│   │   │   ├── raw_calendars.ipynb
│   │   │   ├── base_pace_builder.py
│   │   │   ├── fuel_correction.py
│   │   │   └── tyre_feature_engineering.py
│   │   │
│   │   ├── models/
│   │   │
│   │   │   ├── tyre_degradation/
│   │   │   │   ├── degradation_model.py
│   │   │   │   ├── degradation_utils.py
│   │   │   │   ├── cliff_model.py
│   │   │   │   └── fit_degradation_curves.py
│   │   │   │
│   │   │   ├── pace_models/
│   │   │   │   ├── base_pace_model.py
│   │   │   │   ├── fuel_model.py
│   │   │   │   └── pace_adjustments.py
│   │   │   │
│   │   │   ├── simulation/
│   │   │   │   ├── strategy_simulator.py
│   │   │   │   ├── stint_simulator.py
│   │   │   │   ├── lap_time_engine.py
│   │   │   │   ├── pitstop_model.py
│   │   │   │   ├── tyre_state.py
│   │   │   │   ├── fuel_state.py
│   │   │   │   └── race_constraints.py
│   │   │   │
│   │   │   ├── optimization/
│   │   │   │   ├── strategy_optimizer.py
│   │   │   │   ├── monte_carlo_simulator.py
│   │   │   │   ├── strategy_generator.py
│   │   │   │   └── stochastic_models.py
│   │   │   │
│   │   │   ├── evaluation/
│   │   │   │   ├── evaluate_strategy.py
│   │   │   │   ├── compare_predictions.py
│   │   │   │   └── validation_metrics.py
│   │   │   │
│   │   │   └── saved_models/
│   │   │       ├── all_tracks_degradation.pkl
│   │   │       ├── track_base_pace.pkl
│   │   │       ├── fuel_models.pkl
│   │   │       ├── pitstop_loss.pkl
│   │   │       └── tyre_cliff_models.pkl
│   │   │
│   │   ├── outputs/
│   │   │   ├── simulations/
│   │   │   ├── strategy_reports/
│   │   │   ├── race_predictions/
│   │   │   └── telemetry_exports/
│   │   │
│   │   ├── pipeline/
│   │   │   ├── data_pipeline.py
│   │   │   ├── preprocessing_pipeline.py
│   │   │   ├── simulation_pipeline.py
│   │   │   └── training_pipeline.py
│   │   │
│   │   ├── scripts/
│   │   │   ├── run_simulation.py
│   │   │   ├── train_degradation_models.py
│   │   │   ├── generate_strategy_report.py
│   │   │   └── benchmark_strategies.py
│   │   │
│   │   ├── visualizations/
│   │   │   ├── tyre_deg_plots.py
│   │   │   ├── strategy_comparison.py
│   │   │   ├── race_trace_visualizer.py
│   │   │   └── telemetry_dashboard.py
│   │   │
│   │   └── notebooks/
│   │       ├── race_strat_optimization_model.ipynb
│   │       ├── DetailsExtractedForARace.ipynb
│   │       ├── degradation_analysis.ipynb
│   │       └── stint_analysis.ipynb
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── dashboard/
│   ├── strategy-viewer/
│   ├── telemetry/
│   └── race-comparison-ui/
│
├── docs/
│   ├── architecture.md
│   ├── tyre_modeling.md
│   ├── simulation_engine.md
│   └── optimization_notes.md
│
├── tests/
│   ├── test_degradation.py
│   ├── test_strategy_simulator.py
│   ├── test_fuel_model.py
│   └── test_optimizer.py
│
└── README.md
```

---

# 🏎️ Data Pipeline Workflow

```text
FastF1 API
     ↓
Raw Session Data
     ↓
Data Cleaning & Normalization
     ↓
Feature Engineering
     ↓
Model Training
     ↓
Race Simulation
     ↓
Strategy Optimization
```

---

# 📥 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/race-strategy-optimization-system.git
cd race-strategy-optimization-system
```

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / MacOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Run Data Collection

```bash
python scripts/data_collection.py
```

## Run Preprocessing

```bash
python scripts/preprocessing.py
```

## Train the Model

```bash
python scripts/training.py
```

## Run Simulation

```bash
python scripts/simulation.py
```

---

# 🧪 Example Use Cases

* Predicting lap times for upcoming races
* Simulating alternative pit strategies
* Driver pace comparison across stints
* Tire degradation modeling
* Track evolution analysis
* Race pace forecasting for future seasons

---

# 📊 Sample Analysis Goals

## Common Track Modeling

The project can identify tracks that remain constant across multiple Formula 1 seasons and use them for long-term model stability.

## Normalized Lap Time Prediction

Models can normalize lap times by:

* Tire compound
* Fuel load
* Track conditions
* Safety car periods
* Driver pace variability

## Strategy Evaluation

The system can compare:

* One-stop vs two-stop strategies
* Medium-hard vs soft-medium tire plans
* Undercut effectiveness
* Pit window optimization

---

# 📈 Future Improvements

* Real-time telemetry streaming
* Reinforcement learning for strategy optimization
* Full race Monte Carlo simulation
* Weather-aware race prediction
* Driver risk modeling
* Dashboard frontend using React
* Cloud deployment pipeline

---

# 🧠 Learning Outcomes

This project demonstrates practical knowledge in:

* Machine Learning Pipelines
* Data Engineering
* Motorsport Analytics
* Predictive Modeling
* API Integration
* Backend System Design
* Data Visualization

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a pull request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

**Arnav Pandey**

Computer Science Engineering Student
Passionate about:

* Formula 1 Analytics
* Machine Learning
* Astrophysics
* Backend Development
* Data Science

---

# ⭐ Repository Support

If you found this project useful:

* Star the repository
* Fork the project
* Share feedback
* Suggest improvements

---

# 📷 Suggested GitHub Additions

You can further improve the repository by adding:

* Screenshots of telemetry visualizations
* Model accuracy metrics
* GIFs of simulations
* Architecture diagrams
* Jupyter notebook previews
* API documentation

---
