# AFL Goal-Scoring Predictor: An Enterprise Data Science Project

## 🌐 Live Demo
**[👉 Play with the Interactive Dashboard Here!](https://insy674-team5-afl-performance-analysis.streamlit.app/)**

<details>
<summary><b>📸 Dashboard Screenshots (Click to view)</b></summary>
<br>
<img width="959" height="499" alt="image" src="https://github.com/user-attachments/assets/250e1dab-a831-45a9-832c-4a28b0c6f038" />
<img width="959" height="498" alt="image" src="https://github.com/user-attachments/assets/57a16c0c-8791-41e8-b8c8-4d398b10285a" />
<img width="959" height="497" alt="image" src="https://github.com/user-attachments/assets/272a4107-a8f4-4513-8573-c614d343f318" />
<img width="959" height="496" alt="image" src="https://github.com/user-attachments/assets/199731fe-1eff-451b-b63b-3e600cdb71c9" />
<img width="959" height="497" alt="image" src="https://github.com/user-attachments/assets/56849a6d-e9e6-49f9-ba7a-ed98901babfb" />
</details>


## Project Overview
This repository contains the code, analysis, and documentation for our Enterprise Data Science & Machine Learning in Production (INSY674) project. We built an **explainable machine learning system** to predict goal-scoring probability for AFL players, transforming over a century of historical sports data into a strategic decision-support tool for coaching departments.

**Core Value Proposition:** Professional sports are high-stakes enterprises. AFL teams invest millions, yet critical match strategies often rely on intuition rather than empirical evidence. Our solution helps the Coaching Department make better decisions by providing data-backed predictions. This isn't just a scoreboard forecast; it's a decision-support system to understand opponent weaknesses and optimize team selection.

## Business Challenge & Solution

### The Challenge
- **High-Stakes Decisions:** AFL teams make multi-million dollar decisions on player selection, strategy, and recruitment with limited empirical backing.
- **Data Underutilization:** While vast historical data exists, it remains siloed and not systematically analyzed for predictive insights.
- **Intuition vs. Evidence:** Coaches and selectors often rely on experience over data-driven probability, potentially missing key performance indicators.

### Our Solution
We developed a **two-part analytics engine**:

1. **Predictive Model** *(located in `/models` folder)*: A machine learning system that predicts goal-scoring probability for individual players based on their attributes and game context, with full explainability via SHAP.
2. **Causal Inference Model** *(located in `/models` folder)*: Tests hypotheses about *why* performance happens—understanding the causal impact of physical attributes (height, weight, BMI) and rule changes on player performance across positions.

## Predictive Model (in /models folder)

### Model Architecture
We are developing a **stacked ensemble model** that combines:
- **XGBoost/LightGBM** for handling non-linear relationships and feature interactions
- **Neural Networks** for capturing complex patterns in player performance
- **Logistic Regression** as a interpretable baseline

### Features Used for Prediction
- **Player Physical Attributes:** Height, weight, BMI, age, primary position
- **Performance Metrics:** Kicks, handballs, marks, tackles, inside-50s, clearances (historical averages)
- **Contextual Features:** Home/away status, opponent strength, venue, weather conditions
- **Derived Metrics:** Efficiency ratios, form trends (last 5 games), career stage indicators

### Explainability (XAI)
We implement **SHAP (SHapley Additive exPlanations)** to:
- Identify which features most influence each player's goal-scoring probability
- Provide coaches with transparent, interpretable predictions
- Validate or challenge our causal hypotheses
- Generate actionable insights for team selection and opponent analysis

### Model Evaluation
- **Primary Metric:** AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- **Secondary Metrics:** Precision, Recall, F1-Score, Log-Loss
- **Validation Strategy:** Time-based split to prevent data leakage (train on 2012-2022, validate on 2023-2025)

## Hypothesis & Research Questions (Causal Analysis)
Our causal inference analysis tested six core hypotheses about factors influencing AFL player performance:

| Hypothesis | Treatment | Effect On | Key Finding |
|------------|-----------|-----------|-------------|
| **H1** | Height | Position-specific outcomes | Rucks (+4.84 HitOuts) - MASSIVE effect; Forwards (-0.31) - negative |
| **H2** | Weight | Clearances, HitOuts | Rucks (+4.69) & Midfield (+0.68) benefit; Forwards/Defenders don't |
| **H3** | BMI | Running vs contest stats | Higher BMI benefits EVERY position - modern game rewards physicality |
| **H4** | is_home | Key outcomes | Only rucks benefit (+0.32 HitOuts) - familiar bounce rhythms matter |
| **H5** | Rule changes | How effects changed over time | 6-6-6 rule made height 9x more valuable (+896%); rotation caps nearly eliminated weight advantage |

## Data Overview

**Source:** [Kaggle 'AFL Stats' Dataset](https://www.kaggle.com/datasets/stoney71/aflstats)
- **players.csv:** Individual players' performance statistics per game (e.g., kicks, handballs, marks, tackles)
- **stats.csv:** Match outcomes, scores, venues, attendance, and team-level statistics

### Data Challenges & Preprocessing
This project involved significant real-world data engineering:
- **Historical Inconsistencies:** Rule changes over decade
- **Entity Resolution:** Team name variations required careful mapping
- **Data Integration:** Joining player-level and match-level data to create unified feature set

## Repository Structure
```
AFL-prediction/
│
├── data/                            # Data directory
│   ├── raw/                         # Original, immutable data (stats.csv, players.csv, games.csv)
│   └── processed/                   # Cleaned data and processing logic (df_final_final.csv, Cleaned_Data.ipynb)
│
├── Models/                          # Model artifacts and notebooks
│   ├── Casual Model.ipynb           # Completed causal inference analysis (H1-H5)
│   └── Predictive Model.ipynb       # Completed position-specific predictive modeling
│
├── reports/                         # Generated outputs
│   ├── figures/                     
│   │   ├── Causal Model/            # HTE plots for physical attributes
│   │   └── Predictive Model/        # Coefficients, SHAP values, and model comparisons
│   └── tables/                      
│       └── predictive_model_summary.xlsx # Quantitative performance metrics
│
├── src/                             # Source code modules
│   └── visualization/               
│       └── Dashboard.v3.py          # Streamlit dashboard source
│
├── .gitignore                       # Files to ignore (e.g., .DS_Store)
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Jupyter Notebook / VSCode

### Installation
```bash
# Clone the repository
git clone https://github.com/fayeflight2727-coder/AFL-prediction.git

# Navigate to project directory
cd AFL-prediction

# Install dependencies
pip install -r requirements.txt
```

### Run the Analysis
1. **Causal Analysis (Completed)**
   ```bash
   jupyter notebook Models/Casual\ Model.ipynb
   ```

2. **Predictive Modeling (Coming Soon)**
   ```bash
   jupyter notebook Models/Predictive\ Model.ipynb
   ```

## Key Results

### Causal Analysis Findings
- **6-6-6 Rule (2019):** Made height **9x more valuable** for rucks (+896% effect)
- **Rotation Caps:** Nearly eliminated weight advantage for midfielders (-87.6% effect)
- **Home Advantage:** Only real for rucks (+0.32 hitouts)
- **BMI:** Benefits EVERY position - modern game rewards physicality everywhere

### Predictive Model (Expected Outcomes)
- **Goal:** Predict goal-scoring probability with >0.80 AUC-ROC
- **Top Predictors Expected:** Inside-50s, kicks, clearances, position-specific factors
- **Business Impact:** Provide coaches with data-backed player selection and opponent analysis

## Strategic Applications
1. **Talent Identification:** Pinpoint players with high scoring potential who may be undervalued
2. **Opponent Analysis:** Identify weaknesses in opposing teams' defensive structures
3. **Tactical Optimization:** Determine which playing styles maximize scoring probability
4. **In-Game Decision Support:** Real-time substitution and strategy recommendations
5. **Recruitment:** Data-backed decisions on which physical attributes matter for each position

## Team
**McGill University MMA8 INSY674 Team:**
- Faye Wu
- Jacob Featherstone
- Rui Zhao
- Monica Jang
- Joohee Kim

## License
This project is for educational purposes as part of the McGill University MMA program.

## Acknowledgments
- Kaggle for providing the AFL dataset
- McGill University MMA faculty for guidance
- Australian Football League for the rich historical data

---

*For questions about this project, please contact the team members or open an issue in this repository.*
