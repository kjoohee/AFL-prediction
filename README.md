# AFL Goal-Scoring Predictor: An Enterprise Data Science Project

## Project Overview
This repository contains the code, analysis, and documentation for our Enterprise Data Science & Machine Learning in Production (INSY674) project. We built an **explainable machine learning system** to predict goal-scoring probability for AFL players, transforming over a century of historical sports data into a strategic decision-support tool for coaching departments.

**Core Value Proposition:** Professional sports are high-stakes enterprises. AFL teams invest millions, yet critical match strategies often rely on intuition rather than empirical evidence. Our solution helps the Coaching Department make better decisions by providing data-backed predictions. This isn't just a scoreboard forecast; it's a decision-support system to understand opponent weaknesses and optimize team selection.

## Business Challenge & Solution

### The Challenge
- **High-Stakes Decisions:** AFL teams make multi-million dollar decisions on player selection, strategy, and recruitment with limited empirical backing.
- **Data Underutilization:** While vast historical data exists, it remains siloed and not systematically analyzed for predictive insights.
- **Intuition vs. Evidence:** Coaches and selectors often rely on experience over data-driven probability, potentially missing key performance indicators.

### Our Solution
We developed a predictive analytics engine that:
1.  **Predicts Goal-Scoring Probability:** Estimates the likelihood of individual players scoring goals based on their attributes and game context.
2.  **Explains the "Why":** Uses Explainable AI (XAI) to identify which factors (e.g., kicks, inside-50s, height) most influence each prediction.
3.  **Provides Actionable Insights:** Translates model outputs into strategic recommendations for team selection and opponent analysis.

## Hypothesis & Research Questions
Our project tests six core hypotheses about factors influencing goal-scoring probability in AFL:

| Hypothesis Category | Hypothesis Statement | Rationale & Business Insight |
| :--- | :--- | :--- |
| **Physical Attributes** | **H1 (Height):** Increased player height is positively correlated with goal-scoring probability. | Taller players may have an advantage in aerial contests and marking near goals. Helps in recruiting and positioning key forwards. |
| | **H2 (Weight):** Higher player weight is negatively correlated with goal-scoring probability. | Heavier players may sacrifice agility and speed needed to create scoring opportunities. Informs fitness and conditioning programs. |
| **Performance Metrics** | **H3 (Kicks):** A higher frequency of kicks will significantly increase the likelihood of goals. | Directly measures shooting volume and opportunity creation. Identifies players who effectively convert possession into attempts. |
| | **H4 (Handballs):** Increased handball volume will show a positive correlation with scoring opportunities. | Measures involvement in build-up play and team ball movement. Highlights playmakers who create chances for others. |
| | **H5 (Inside-50s):** The number of 'Inside-50' entries will be the strongest positive predictor of goal-scoring. | Represents a team's ability to move the ball into scoring positions. A key metric for assessing midfield-to-forward connection. |
| **Contextual Factor** | **H6 (HomeTeam):** Playing as the Home Team will amplify the impact of other performance metrics. | Tests the "Home Ground Advantage" effect and whether it makes player strengths more pronounced. Helps in pre-game psychological preparation. |

## Data Overview

**Source:** [Kaggle 'AFL Stats' Dataset](https://www.kaggle.com/datasets/stoney71/aflstats)
- **players.csv:** Individual players' performance statistics per game (e.g., kicks, handballs, marks, tackles).
- **stats.csv:** Match outcomes, scores, venues, attendance, and team-level statistics.
- **Scope:** Covers **125+ years** of league history (1897 - 2025).

### Data Challenges & Preprocessing
This project involved significant real-world data engineering:
- **Historical Inconsistencies:** Rule changes, stat definitions, and recording methods evolved over 12+ decades.
- **Data Gaps:** Missing values are prevalent in early 20th-century records.
- **Entity Resolution:** Team name variations across decades (e.g., "University" team existed only from 1908-1914) required careful mapping.
- **Data Integration:** Joining player-level and match-level data to create a unified feature set for modeling.

## Repository Structure
```
afl-goal-predictor/
│
├── data/                           # Data directory
│   ├── raw/                        # Original, immutable data
│   ├── processed/                  # Cleaned, transformed data
│   └── external/                   # Any supplementary data
│
├── notebooks/                      # Jupyter notebooks for analysis
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature creation & selection
│   ├── 03_modeling.ipynb          # Model training & evaluation
│   └── 04_explainability.ipynb    # SHAP analysis & interpretation
│
├── src/                            # Source code modules
│   ├── data/                       # Data processing scripts
│   │   ├── make_dataset.py
│   │   └── build_features.py
│   ├── models/                     # Modeling scripts
│   │   ├── train_model.py
│   │   └── predict_model.py
│   └── visualization/              # Visualization utilities
│       └── visualize.py
│
├── models/                         # Saved model artifacts
│   ├── final_model.pkl
│   └── model_performance.json
│
├── reports/                        # Generated reports
│   ├── figures/                    # Graphs and visualizations
│   └── final_report.pdf
│
├── .github/workflows/              # CI/CD pipelines (if applicable)
├── .gitignore                      # Files to ignore in version control
├── requirements.txt                # Python dependencies
├── environment.yml                 # Conda environment
├── mlflow.yaml                     # MLflow configuration
└── README.md                       # This file
```

## Getting Started

### Prerequisites
- Python 3.8+
- Git

### Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/afl-goal-predictor.git
   cd afl-goal-predictor
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the data**
   - Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/stoney71/aflstats)
   - Place `players.csv` and `stats.csv` in the `data/raw/` directory

### Usage
Run the analysis pipeline in order:

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_eda.ipynb
   ```

2. **Feature Engineering**
   ```bash
   jupyter notebook notebooks/02_feature_engineering.ipynb
   ```

3. **Model Training & Evaluation**
   ```bash
   python src/models/train_model.py
   ```

4. **Model Interpretation**
   ```bash
   jupyter notebook notebooks/04_explainability.ipynb
   ```

## Methodology

### Feature Engineering
We created several feature categories from the raw data:
- **Player Physical Attributes:** Height, weight, position
- **Performance Metrics:** Kicks, handballs, marks, tackles, inside-50s
- **Contextual Features:** Home/away status, venue, season stage
- **Derived Metrics:** Efficiency ratios, per-game averages, trend indicators

### Modeling Approach
1. **Data Splitting:** Time-based split to prevent data leakage
2. **Baseline Models:** Logistic Regression, Random Forest
3. **Advanced Models:** XGBoost, LightGBM, Neural Networks
4. **Hyperparameter Tuning:** Grid search with cross-validation
5. **Ensemble Methods:** Stacking and blending best-performing models

### Explainable AI (XAI)
We implemented SHAP (SHapley Additive exPlanations) to:
- Identify global feature importance across all predictions
- Provide local explanations for individual player predictions
- Validate or challenge our initial hypotheses
- Generate actionable insights for coaching staff

## Results & Business Impact

### Key Findings
- **Most Predictive Features:** [To be filled after analysis - e.g., Inside-50s, kicks, specific player positions]
- **Hypothesis Validation:** [Which hypotheses were supported/refuted by the data]
- **Model Performance:** [Accuracy, precision, recall, AUC-ROC metrics]

### Strategic Applications
1. **Talent Identification:** Pinpoint players with high scoring potential who may be undervalued
2. **Opponent Analysis:** Identify weaknesses in opposing teams' defensive structures
3. **Tactical Optimization:** Determine which playing styles maximize scoring probability
4. **In-Game Decision Support:** Provide real-time substitution and strategy recommendations

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
