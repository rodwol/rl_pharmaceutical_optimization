# Rebex: A Reinforcement Learning Approach to Essential Medicine Stockout Prevention in Eritrean District Hospital Pharmacies
 
## Table of Contents
 
1. [Project Description](#project-description)
2. [System Architecture](#system-architecture)
3. [Features](#features)
4. [Tech Stack](#tech-stack)
5. [How to Install and Run](#how-to-install-and-run)
6. [How to Use](#how-to-use)
7. [Project Structure](#project-structure)
8. [Methodology](#methodology)
9. [Performance Metrics](#performance-metrics)
10. [Challenges & Future Work](#challenges--future-work)
11. [Credits](#credits)
12. [License](#license)
---
 
## Project Description
 
**Rebex** is an intelligent pharmaceutical inventory management system that uses **Deep Q-Network (DQN) Reinforcement Learning** to prevent essential medicine stockouts in Eritrean district hospital pharmacies.
 
District hospital pharmacies in Eritrea are the primary point of medicine access for large portions of the population, yet essential medicine availability sits at around 80% — meaning roughly 1 in 5 prescribed medicines is unavailable at the point of care. Traditional inventory control methods such as Economic Order Quantity (EOQ) and manual periodic reviews cannot adapt to the unpredictable demand patterns, seasonal disease surges, and supply chain disruptions common in low-resource healthcare environments.
 
This system models the pharmacy inventory replenishment problem as a **Markov Decision Process (MDP)** and trains a DQN agent — augmented with a **Hidden Markov Model (HMM)** for demand regime detection — to generate optimal daily restocking recommendations. The trained agent is served through a RESTful FastAPI backend and surfaced to pharmacy staff via an interactive **Streamlit dashboard**.
 
**Why these technologies?**
- **Python ecosystem (PyTorch, Stable-Baselines3, hmmlearn):** Mature, open-source, and well-documented RL tooling with minimal infrastructure cost — critical for resource-constrained environments.
- **Streamlit:** Enables rapid development of a data-driven UI without requiring frontend engineering, making it maintainable by a single developer.
- **PostgreSQL + Docker:** Lightweight, reliable, and portable — the containerized stack can be deployed on modest hardware found in district-level facilities.
- **Synthetic data calibrated from literature:** Real aggregate statistics from Siele et al. (2022) and Abdu et al. (2020) parameterize the simulation, grounding it in Eritrean epidemiological realities.
---
 
## System Architecture
 
```
┌─────────────────────────────────┐
│        Streamlit Dashboard       │  ← User Interface Layer
│  (Inventory monitoring, alerts,  │
│   replenishment recommendations) │
└────────────────┬────────────────┘
                 │ HTTP / REST
┌────────────────▼────────────────┐
│         FastAPI Backend          │  ← RL Inference Layer
│  (DQN agent, HMM demand model,  │
│   /api/recommend endpoint)       │
└────────────────┬────────────────┘
                 │ SQLAlchemy ORM
┌────────────────▼────────────────┐
│         PostgreSQL Database      │  ← Data Layer
│  (Medicines, stock records,      │
│   orders, recommendations)       │
└─────────────────────────────────┘
```
 
---
 
## Features
 
- **Daily stock count recording** — pharmacy technicians log current inventory levels
- **RL-powered replenishment recommendations** — DQN agent recommends order quantities per medicine based on current state
- **Demand regime detection** — HMM identifies latent demand states (stable, surge, disruption) to provide context-aware recommendations
- **Stockout risk dashboard** — visual alerts for medicines approaching critical stock levels
- **Order approval workflow** — pharmacy manager reviews and approves recommendations before submission
- **Inventory performance reports** — historical stockout frequency, service level, and holding cost trends
- **EOQ baseline comparison** — side-by-side metrics comparing RL agent vs. traditional EOQ policy
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| Dashboard / UI | Streamlit |
| Backend / API | FastAPI |
| RL Framework | Stable-Baselines3 (DQN), PyTorch |
| Demand Modeling | hmmlearn (HMM) |
| RL Environment | Custom OpenAI Gym-compatible env |
| Database | PostgreSQL |
| ORM | SQLAlchemy |
| Containerization | Docker + Docker Compose |
| Data & Viz | Pandas, NumPy, Matplotlib, Seaborn |
| Language | Python 3.10+ |
 
---
 
## How to Install and Run
 
### Prerequisites
 
- [Docker](https://www.docker.com/get-started) and Docker Compose installed
- Git installed
- Python 3.10+ (if running outside Docker)
### 1. Clone the repository
 
```bash
git clone https://github.com/rodwol/rl_pharmaceutical_optimization.git
cd rl_pharmaceutical_optimization
```
 
### 2. Set up environment variables
 
```bash
cp .env.example .env
```
 
```env
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
DATABASE_URL=
```
 
### 3. Build and run with Docker Compose
 
```bash
docker-compose up --build
```
 
This will spin up three services:
- `db` — PostgreSQL database on port `5432`
- `api` — FastAPI backend on port `8000`
- `dashboard` — Streamlit UI on port `8501`
### 4. Access the application
 
| Service | URL |
|---|---|
| Streamlit Dashboard | http://|
| FastAPI Docs (Swagger) | http:// |
| Database |  |
 
### 5. Running outside Docker (development mode)
 
```bash
# Install dependencies
pip install -r requirements.txt
 
# Start the database (Docker only for DB is fine)
docker-compose up db
 
# Run the API
uvicorn app.main:app --reload --port 8000
 
# Run the dashboard (separate terminal)
streamlit run dashboard/app.py
```
 
### 6. Train the RL agent
 
```bash
python rl/train.py --episodes 1000 --save-path models/dqn_agent.pt
```
 
---
 
## How to Use
 
### Pharmacy Technician
 
1. Log in with your assigned credentials.
2. Navigate to **Daily Stock Count** and enter current quantity on hand for each medicine.
3. Click **Get Recommendation** — the RL agent will generate an order suggestion.
4. Submit the count. The recommendation is forwarded to the Pharmacy Manager for approval.
### Pharmacy Manager
 
1. Navigate to **Pending Recommendations**.
2. Review the recommended order quantities alongside current stock levels and stockout risk scores.
3. Approve or adjust quantities and submit the order to the district medical store.
4. View historical performance on the **Reports** tab.
### Default credentials (development only)
 
| Role | Username | Password |
|---|---|---|
| Technician | `tech_demo` | `demo1234` |
| Manager | `manager_demo` | `demo1234` |
 
> Change all credentials before any real-world deployment.
 
---
 
## Project Structure
 
```
rxguard-rl-pharmacy/
├── api/                    # FastAPI backend
│   ├── main.py
│   ├── routes/
│   │   ├── recommend.py    # /api/recommend endpoint
│   │   └── orders.py
│   └── models/             # SQLAlchemy ORM models
├── dashboard/              # Streamlit frontend
│   ├── app.py
│   └── pages/
│       ├── stock_count.py
│       ├── recommendations.py
│       └── reports.py
├── rl/                     # Reinforcement learning core
│   ├── environment.py      # Custom Gym environment
│   ├── agent.py            # DQN agent
│   ├── train.py            # Training script
│   └── hmm_demand.py       # HMM demand regime model
├── data/                   # Synthetic data generation
│   └── generate_synthetic.py
├── models/                 # Saved model weights
├── notebooks/              # Jupyter notebooks (EDA, training)
│   ├── 01_data_exploration.ipynb
│   ├── 02_hmm_demand_modeling.ipynb
│   └── 03_dqn_training_evaluation.ipynb
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```
 
---
 
## Methodology
 
### Synthetic Data Generation
 
Training data is generated synthetically but **calibrated against published Eritrean healthcare statistics**: baseline stockout rates (~20%) from Siele et al. (2022), antibiotic and NSAID demand distributions from Abdu et al. (2020) and Amaha et al. (2019), and lead time distributions typical of district-level supply chains in Sub-Saharan Africa.
 
### HMM Demand Regime Modeling
 
A Hidden Markov Model (hmmlearn) identifies latent demand states — **stable**, **surge** (outbreak/seasonal spike), and **disruption** (supply-side shock) — from the observed demand sequences. The inferred regime belief vector is incorporated into the MDP state space, giving the DQN agent richer context for decisions.
 
### Markov Decision Process
 
| Component | Definition |
|---|---|
| **State (S)** | Stock level, days since last order, pending order qty, 7-day demand signal, HMM regime belief |
| **Action (A)** | Discrete order quantities: {0, Q_min, Q_mid, Q_max} |
| **Reward (R)** | −10 per stockout day / −0.05 per unit overstocked / −5 per expired unit / +1 per day full availability |
| **Discount (γ)** | 0.95 |
 
### Baseline Comparison
 
DQN agent performance is benchmarked against an **Economic Order Quantity (EOQ)** baseline and a **fixed periodic review** policy using:
- Stockout frequency
- Service level (% of demand days fully met)
- Inventory holding cost
- Medicine waste (expired units)
---
 
## Performance Metrics
 
> Results will be updated as training progresses.
 
| Metric | EOQ Baseline | DQN Agent |
|---|---|---|
| Stockout Frequency | — | — |
| Service Level | — | — |
| Holding Cost | — | — |
| Medicine Waste | — | — |
 
---
 
## Challenges & Future Work
 
**Challenges faced:**
- Absence of real transaction-level pharmacy data from Eritrea required careful literature-based calibration of the synthetic generator.
- Designing a reward function that balances stockout avoidance, overstocking penalties, and expiry costs simultaneously required significant tuning.
- Integrating HMM belief states into the DQN state representation added complexity to the training pipeline.
**Future work:**
- **Real-world pilot validation:** Deploy in shadow mode alongside the existing manual system at a district hospital pharmacy, log agent recommendations vs. actual decisions, and evaluate retrospectively — without disrupting operations.
- Extend to multi-medicine, multi-facility supply chain optimization.
- Incorporate real eLMIS data feeds when available.
- Explore more advanced RL algorithms (PPO, A3C) for comparison.
---
 
## Credits
 
**Author:** Rodas Goniche
BSc. Software Engineering Capstone Project
**Supervisor:** Simeon Nsabiyumva
 
 
---
 
## License
 
This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
 
You are free to use, modify, and distribute this software for academic and non-commercial purposes with attribution.
 
