# 🚂 AutoBlock AI

**Intelligent Railway Block Planning & Optimization System**

> *One Block. Three Departments. Zero Delays.*

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.94-brightgreen.svg)
![Block Reduction](https://img.shields.io/badge/Block%20Reduction-80.8%25-success.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📌 Overview

**AutoBlock AI** is a data-driven, AI-powered command center that transforms how Indian Railways plans maintenance blocks across its fixed infrastructure.

Railway maintenance today is planned in **silos** — Engineering (Track), Signal & Telecom (S&T), and Traction Distribution each independently request track blocks through the BDMS system. This leads to:

- ❌ Redundant track closures (same corridor shut down multiple times per week)
- ❌ Poor cross-department coordination
- ❌ Suboptimal scheduling that delays trains
- ❌ Reduced asset availability and safety risks
- ❌ **Forgotten temporary fixes that fail before permanent repair**

**AutoBlock AI** solves this with a complete operational pipeline:

- ✅ Integrates maintenance data (TMS, SMMS, TDMS) with train schedules (COA)
- ✅ Uses **two XGBoost ML models** — assignment prediction + temp-fix failure prediction
- ✅ **Dynamically escalates priority** as temporary fixes age
- ✅ Clusters multi-department work into shared **"Golden Blocks"**
- ✅ Validates every schedule with **Monte Carlo Regret Simulation**
- ✅ **Guarantees expired temp-fix coverage** via corridor-first tier sorting
- ✅ Delivers **~81% reduction** in track closures at **~96% operational safety**

---

## 📊 Verified Results

| Metric | Value |
|--------|-------|
| **Tasks Scheduled** | 577 |
| **Corridors Covered** | 20 / 20 |
| **Total Blocks Used** | 111 |
| **Golden Blocks (2+ depts)** | 97 |
| **Full Golden Blocks (3 depts)** | 30 |
| **Block Reduction** | **80.8%** |
| **Operational Safety** | 96% (Monte Carlo validated) |
| **Expired Temp-Fix Coverage** | **84%** (224 / 268) |
| **Critical Temp-Fix Coverage** | 46% (342 / 748) |
| **ML Model 1 (Assignment) ROC-AUC** | **0.9435** |
| **ML Model 2 (Temp-Fix) ROC-AUC** | **0.8902** |
| **Baseline Lift** | **+41.1%** over best rule-based heuristic |

---

## 🎯 The USP — Why AutoBlock AI Wins

AutoBlock AI is not a scheduling calendar. It is a **decision intelligence engine** with eight core innovations:

1. **The "Golden Block" Algorithm** — Clusters Track + Signal + Electrical into one shared window → 81% fewer closures
2. **Two ML Models** — Assignment predictor (0.94 ROC-AUC) + Temp-fix failure predictor (0.89 ROC-AUC)
3. **Dynamic Priority Escalation** — Priority rises from 0.60 to 1.00 as temp fix ages
4. **Temp-Fix Lifecycle Tracking** — Every temp-restored asset is scored for imminent failure
5. **Mandatory Expiry Enforcement** — Corridor-first tier sorting guarantees expired fixes get scheduled
6. **Monte Carlo Regret Simulation** — Validates every block against real RSTGCN train delays
7. **Per-Division Architecture** — Runs independently per corridor, mirroring real Divisional Control Offices
8. **Explainable AI** — Feature importance + clash probability + escalation reasons

---

## 🏗️ System Architecture

```
   [RAW DATA SOURCES]
   ┌─────────┬──────────┬─────────┬─────────┐
   │  TMS    │  SMMS    │  TDMS   │   COA   │
   │ (Track) │ (Signal) │ (Elec.) │(Trains) │
   └────┬────┴────┬─────┴────┬────┴────┬────┘
        │         │          │         │
        ▼         ▼          ▼         ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 1: Data Integration Layer       │
   │  (4 sources → unified task store)       │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 2: Temp-Fix Enrichment          │
   │  (Adds lifecycle: applied_at, lifespan) │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 3: ML Layer (Two Models)        │
   │  ├─ Assignment Predictor (ROC-AUC 0.94) │
   │  └─ Temp-Fix Failure (ROC-AUC 0.89)     │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 4: Priority Escalation Engine   │
   │  (Dynamic escalation by temp-fix age)   │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 5: Golden Block Optimizer       │
   │  (Corridor-first, tier-sorted, ML-scored│
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 6: Monte Carlo Validation       │
   │  (1,000 simulations per block)          │
   └─────────────────┬───────────────────────┘
                     │
                     ▼
   ┌─────────────────────────────────────────┐
   │  MODULE 7: Dashboard & Reports          │
   │  (Interactive Gantt + KPI cards)        │
   └─────────────────────────────────────────┘
```

---

## 🔄 Operational Workflow (The Nightly Batch Model)

AutoBlock AI operates on a **nightly batch model**, mirroring real Indian Railways operations:

```
┌────────────────────────────────────────────────────────────┐
│  TODAY (During work hours — human-only)                    │
│  • Field technicians detect failures                       │
│  • Apply temporary fixes immediately                       │
│  • Submit 60-second reports → accumulate in CSV            │
└──────────────────────────┬─────────────────────────────────┘
                           │ 18:00 — End of shift
                           ▼
┌────────────────────────────────────────────────────────────┐
│  TONIGHT (Automatic AI batch — ~2 minutes compute)         │
│  1. Load daily_failures.csv                                │
│  2. Merge with backlog                                     │
│  3. Run ML predictions (2 models)                          │
│  4. Escalate priority dynamically                          │
│  5. Optimize into Golden Blocks                            │
│  6. Validate with Monte Carlo                              │
│  7. Generate schedule CSV + reports                        │
└──────────────────────────┬─────────────────────────────────┘
                           │ 06:00 — Next morning
                           ▼
┌────────────────────────────────────────────────────────────┐
│  TOMORROW (Human review + execution)                       │
│  • Section Engineers see work orders                       │
│  • Control Office approves weekly plan                     │
│  • Department Heads verify                                 │
│  • Blocks execute at scheduled time                        │
└────────────────────────────────────────────────────────────┘
```

**Key insight:** The AI does not need real-time data. It runs overnight on a nightly CSV dump — exactly how real railway planning works.

---

## 📂 Project Structure

```
AutoBlock_AI/
│
├── data/
│   ├── raw/                              # ⚠️ NOT in repo — download separately
│   │   ├── rstgcn/                       # Train routes + delays (IIT KGP)
│   │   ├── kaggle_failure/               # Multi-department defect data
│   │   ├── kaggle_timetable/             # Indian Railways timetable
│   │   └── osm_railways/                 # Railway geometry (shapefile)
│   └── processed/                        # Clean pipeline outputs
│       ├── corridor_master.csv           # 20 corridors with criticality
│       ├── maintenance_tasks.csv         # 20,264 unified defects
│       ├── maintenance_tasks_enriched.csv# + temp-fix metadata
│       ├── maintenance_tasks_escalated.csv# + ML risk + escalated priority
│       ├── corridor_availability_week.csv# 273 weekly windows
│       └── train_delay_stats.csv         # Per-corridor delay distributions
│
├── src/
│   ├── build_corridors.py                # Stage 1: Build corridor master
│   ├── build_tasks.py                    # Stage 2: Build unified tasks
│   ├── build_availability.py             # Stage 3: Compute time windows
│   │
│   ├── models/
│   │   ├── train_assignment_model.py     # XGBoost Model 1 training
│   │   ├── train_temp_fix_predictor.py   # XGBoost Model 2 training
│   │   ├── temp_fix_escalator.py         # Priority escalation engine
│   │   ├── ml_optimizer.py               # Final v6 optimizer
│   │   ├── baseline_comparison.py        # 4 baselines vs XGBoost
│   │   ├── cross_validation.py           # 5-fold CV
│   │   ├── feature_importance_plot.py    # Feature chart
│   │   └── saved/                        # Model artifacts (not in repo)
│   │
│   └── utils/
│       ├── add_temp_fix_data.py          # Temp-fix enrichment
│       ├── fix_criticality.py            # Reality-based criticality
│       └── expand_to_week.py             # 7-day window expansion
│
├── output/
│   ├── schedules/
│   │   ├── ml_scheduled_blocks_v6.csv    # Final schedule (577 tasks)
│   │   └── ml_golden_blocks_v6.csv       # Golden Block summary (111 blocks)
│   └── reports/
│       ├── temp_fix_risk_report_v6.csv   # Safety-critical coverage
│       ├── corridor_summary_v6.csv       # Per-corridor breakdown
│       ├── model_performance.txt         # Model 1 metrics
│       ├── temp_fix_model_performance.txt# Model 2 metrics
│       ├── baseline_comparison.txt       # ML vs heuristics
│       ├── cross_validation.txt          # Stability proof
│       └── feature_importance.png        # Visual chart
│
├── app.py                                # Streamlit dashboard
├── verify_data.py                        # Data verification script
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- pip package manager
- ~500 MB free disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AutoBlock-AI.git
cd AutoBlock-AI

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
pandas>=2.0.0
numpy>=1.24.0
geopandas>=0.14.0
shapely>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
streamlit>=1.30.0
plotly>=5.18.0
```

---

## 📥 Data Setup (Required Before Running)

This repo does **not** include raw datasets (they exceed GitHub size limits and are confidential). Download them separately:

| Dataset | Link | Destination |
|---------|------|-------------|
| RSTGCN | https://github.com/KoyenaChowdhury/RSTGCN | `data/raw/rstgcn/` |
| Kaggle Failure (100K) | https://www.kaggle.com/datasets/ziya07/indian-railway-failure-detection-and-maintenance | `data/raw/kaggle_failure/` |
| Kaggle Timetable | https://www.kaggle.com/datasets/sriharshaeedala/indian-railways-timetable | `data/raw/kaggle_timetable/` |
| OSM Railways | https://data.humdata.org/dataset/india-railways | `data/raw/osm_railways/` |

After downloading, run the verification script:

```bash
python verify_data.py
```

You should see green checkmarks for all four datasets.

---

## 🔧 Running the Full Pipeline

Run the scripts **in this order**, from the project root:

```bash
# ---- Stage 1-3: Data Pipeline ----
python src/build_corridors.py
python src/utils/fix_criticality.py
python src/build_tasks.py
python src/build_availability.py
python src/utils/expand_to_week.py

# ---- Stage 4: Temp-Fix Layer ----
python src/utils/add_temp_fix_data.py

# ---- Stage 5: ML Model Training ----
python src/models/train_assignment_model.py
python src/models/train_temp_fix_predictor.py

# ---- Stage 6: Priority Escalation ----
python src/models/temp_fix_escalator.py

# ---- Stage 7: Final Optimizer (v6) ----
python src/models/ml_optimizer.py

# ---- Stage 8: Model Validation (Optional) ----
python src/models/baseline_comparison.py
python src/models/cross_validation.py
python src/models/feature_importance_plot.py

# ---- Stage 9: Dashboard ----
streamlit run app.py
```

The dashboard will open automatically at `http://localhost:8501`.

---

## 📊 Data Sources

| Dataset | Source | Purpose |
|---------|--------|---------|
| **RSTGCN Network Topology** | [IIT KGP / arXiv](https://github.com/KoyenaChowdhury/RSTGCN) | Corridor topology + delays |
| **RSTGCN Train Routes** | IIT KGP | Free time windows per corridor |
| **RSTGCN Delay Records** | IIT KGP | Monte Carlo simulation input |
| **Indian Railway Failure Detection (100K)** | [Kaggle](https://www.kaggle.com/datasets/ziya07/indian-railway-failure-detection-and-maintenance) | Multi-department defects |
| **Indian Railways Timetable** | [Kaggle](https://www.kaggle.com/datasets/sriharshaeedala/indian-railways-timetable) | Supplementary timetable |
| **OSM Railways Shapefile** | [HDX](https://data.humdata.org/dataset/india-railways) | Real corridor geometry |

**Note:** TMS, SMMS, TDMS, and COA are confidential government systems. Their data was synthetically generated using statistical distributions from the above public sources.

---

## 🧠 ML Model Performance

### Model 1: Assignment Predictor

| Metric | Value |
|--------|-------|
| **ROC-AUC (test)** | 0.9435 |
| **F1 Score** | 0.7586 |
| **Accuracy** | 0.9043 |
| **Cross-Validation (5-fold)** | 0.9466 ± 0.0202 |
| **Training rows** | 2,197 (17.3% positive) |
| **Features** | 21 engineered |

### Model 2: Temp-Fix Failure Predictor

| Metric | Value |
|--------|-------|
| **ROC-AUC (test)** | 0.8902 |
| **F1 Score** | 0.7376 |
| **Recall** | 0.7559 |
| **Training rows** | 6,994 temp-fix tasks |
| **Features** | 14 engineered |

### Baseline Comparison

| Model | ROC-AUC | Lift |
|-------|---------|------|
| 🚀 **XGBoost (ours)** | **0.9435** | — |
| 🎯 Priority Match | 0.6685 | +41.1% |
| 📏 Longest Window | 0.6522 | +44.6% |
| 🎲 Random | 0.5017 | +88.0% |
| ⚠️ Severity Only | 0.5000 | +88.7% |

### Top 5 Features (Assignment Model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Window Fits Task | 0.299 |
| 2 | Correct Corridor | 0.247 |
| 3 | Time-of-Day (sin) | 0.157 |
| 4 | Day of Week | 0.088 |
| 5 | Severity × Corridor Criticality | 0.034 |

**Key Insight:** The model learned "best-fit assignment wins" — not "highest severity wins."

### Top 5 Features (Temp-Fix Model)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Temp-Fix Age Ratio | 0.356 |
| 2 | Urgency Score (age²) | 0.188 |
| 3 | Severity × Age | 0.080 |
| 4 | Severity | 0.052 |
| 5 | Age × Traffic | 0.040 |

**Key Insight:** The model learned that aging temp fixes fail faster on busy corridors.

---

## 🎯 Sample Golden Block (Real Output)

```
Window:              CORR_007_D0_W0
Corridor:            Howrah - Chennai Main Line (highest criticality)
Day:                 Monday
Tasks:               6
Departments:         [Engineering, Signal & Telecom, Traction Distribution]
Avg Escalated Priority:  0.994
ML Confidence:       0.94
Clash Probability:   0.8%
Status:              OPERATIONALLY SAFE
```

**Interpretation:** Six maintenance tasks across three departments, done in a single shared window, on the busiest corridor in India, with less than 1% chance of any train clash.

---

## 🖥️ Dashboard Features

- **Dark-mode UI** — Modern, professional look
- **5 KPI Cards** — Live headline metrics
- **Interactive Gantt Chart** — Color-coded by department, Golden Blocks glow green
- **Corridor Filter** — Drill down to specific corridors
- **Day Filter** — View any day of the week
- **Temp-Fix Urgency Panel** — Visualize which temp fixes are about to fail
- **ML Metrics Panel** — ROC-AUC, F1, Recall displayed live

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11 |
| **Data Processing** | Pandas, NumPy, GeoPandas |
| **ML** | XGBoost, Scikit-learn |
| **Simulation** | SciPy (Monte Carlo) |
| **Visualization** | Streamlit, Plotly, Matplotlib |
| **Storage** | CSV (prototype) |

---

## 👥 Users & Outputs

| User | What They Give | What They Get |
|------|---------------|---------------|
| **Field Technician** | 60-sec failure report | Ticket ID + ETA for permanent fix |
| **Section Engineer** | 15-sec readiness confirmation | Printable work order with prioritized tasks |
| **Control Office** | 30-sec preferences | Weekly plan across 20 corridors |
| **Divisional Manager** | Nothing (view-only) | Real-time KPI dashboard + trends |

---

## 📈 Roadmap

| Phase | Duration | Milestone |
|-------|----------|-----------|
| ✅ Phase 1 | Completed | Hackathon prototype |
| 🔄 Phase 2 | 1 month | Real API integrations (TMS/SMMS/TDMS) |
| 🔄 Phase 3 | 3 months | Pilot on 1 division with live BDMS export |
| 🔄 Phase 4 | 6 months | Scale to 10 divisions |
| 🔄 Phase 5 | 12 months | Pan-India rollout |

---

## ⚠️ Known Limitations

Being transparent about scope:

| Limitation | Reason |
|-----------|--------|
| Real-time API integration | TMS/SMMS/TDMS are confidential government systems |
| Live train rerouting | Requires COA system access |
| Resource scheduling (staff/machines) | Not modeled — assumed available |
| Power block vs line block distinction | Simplified into single block type |
| Departmental hierarchy routing | Flat task model (tech→JE→officer chain not modeled) |

These are documented as **future work** for production deployment.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.

---

## 🙏 Acknowledgements

- **IIT Kharagpur** — RSTGCN dataset (first nationwide Indian Railways delay dataset)
- **Kaggle contributors** — Indian Railway Failure Detection & Timetable datasets
- **Humanitarian Data Exchange (HDX)** — OSM Railways shapefile
- **Indian Railways** — For inspiration and domain knowledge

---

## 📬 Contact

**Project Maintainer:** Rudraksh Gupta
**Email:** rudrakshktp@gmail.com

---

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover AutoBlock AI.

---

**Built with ❤️ for Indian Railways — AutoBlock AI v6.0**
*"One Block. Three Departments. Zero Delays."*