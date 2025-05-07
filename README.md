# 🏀 March Machine Learning Mania 2024

## 🔍 Project Overview
This repository presents a full data science pipeline developed for the [Kaggle March Machine Learning Mania 2024](https://www.kaggle.com/competitions/march-machine-learning-mania-2024) competition.  
The objective is to predict outcomes of NCAA Men's Basketball Tournament games using a combination of historical data, statistical features, and diverse machine learning techniques.

Key elements of the project include:
- Preprocessing pipeline in [`scripts/pre_processing.py`](scripts/pre_processing.py)
- Exploratory Data Analysis in [`notebooks/EDA/`](notebooks/EDA/)
- Model development using Logistic Regression, Random Forest, XGBoost, and Neural Networks
- Ensemble modeling strategy combining predictions across models
- Evaluation using **log loss** and **Brier score**, the official metrics of the competition

Our final ensemble model outperformed all single models individually, demonstrating the power of stacked generalization and weighted averaging for tournament outcome prediction.

## 🏆 Competition Context
The **March Machine Learning Mania** competition challenges participants to forecast the results of NCAA Men's Basketball Tournament games.  
Participants submit win probabilities for each potential matchup, and are evaluated using the **log loss** metric — which rewards both accurate predictions and appropriate confidence levels.

The tournament’s unpredictable nature, with frequent upsets and underdog victories, makes this a complex and rewarding modeling challenge, ideal for testing robust classification approaches under real-world uncertainty.

## 📁 Repository Structure
```bash
.
├── data/                         # Tournament data
│   └── march-machine-learning-mania-2024.zip
├── models/                       # Saved model files
├── notebooks/                    # Jupyter notebooks for analysis and modeling
│   ├── EDA/                      # Exploratory Data Analysis
│   ├── ensemble/                 # Ensemble modeling approaches
│   ├── logistic_regression/     # Logistic Regression models
│   ├── neural_networks/         # Neural Network models
│   ├── random_forest/           # Random Forest models
│   └── xgboost/                 # XGBoost models
└── scripts/                      # Utility scripts
    └── pre_processing.py        # Data preprocessing functionality



## Running the Project
You can run this project either locally on your machine or within a Kaggle Notebook. For local use, simply clone the repository and ensure all dependencies are installed.

To run the project in a Kaggle environment, make sure to upload `/scripts/pre_processing.py` as a Kaggle Dataset, and name it "preprocessing-module". Then, attach this dataset to your Kaggle Notebook. This step is necessary because Kaggle requires external Python modules to be attached as datasets for import.
Next, you must also upload all the 9 csv prediction files in the directory `/predictions/` as a Kaggle Dataset, and name it "predictions".

## Modeling Approach

## 📊 Data Preparation and Exploration

Initial exploration and feature understanding were conducted using the notebook [`notebooks/EDA/eda_march_madness.ipynb`](notebooks/EDA/eda_march_madness.ipynb).  
This step focused on uncovering key patterns in historical tournament data and informing downstream feature engineering and model design.

Key insights from our EDA include:
- 📈 **Historical trends** in tournament outcomes, highlighting dominant seeds and common upset scenarios
- 🧮 **Correlation analysis** between team-level statistical features and game outcomes
- 🔁 **Season-to-season consistency** for metrics such as win percentage, point differentials, and strength of schedule
- 🎯 **Feature importance visualizations** (e.g., SHAP values, permutation importance) to prioritize inputs for models
- ⚠️ **Upset detection patterns** based on seed differences, recent form, and adjusted efficiency metrics

These findings directly shaped the preprocessing pipeline and model input schema, helping us build interpretable and effective predictive models.


### Model Implementation

#### Logistic Regression (`/notebooks/logistic_regression/`)
Our logistic regression approach incorporates:
- Team efficiency metrics (offensive and defensive)
- Historical tournament performance
- Strength of schedule adjustments
- Comprehensive feature engineering with polynomial interactions
- Balanced regularization to prevent overfitting to historical data

#### Random Forest (`/notebooks/random_forest/`)
The random forest models leverage:
- Ensemble decision trees with optimized depth
- Feature importance-based selection
- Bootstrapped sampling to address tournament volatility
- Out-of-bag error estimation for model validation
- Path probability analysis for bracket completion

#### XGBoost (`/notebooks/xgboost/`)
Our gradient boosting implementation includes:
- Advanced hyperparameter tuning with Bayesian optimization
- Learning rate scheduling for optimal convergence
- Feature importance analysis for interpretable predictions
- Cross-validation across multiple tournament years
- Calibrated probability outputs with isotonic regression

### Neural Networks [`notebooks/neural_networks/`](notebooks/neural_networks/)

Our deep learning model explores tournament prediction via a compact feedforward network trained on engineered team features.

Key characteristics include:
- A 3-layer Multi-Layer Perceptron (MLP) with ReLU activations
- Dropout layers to mitigate overfitting, with dropout rates tuned experimentally
- Binary cross-entropy loss, optimized using the Adam optimizer
- Evaluation conducted through rolling cross-validation across multiple seasons
- Final Brier Score: 0.19086, performing slightly worse than simpler models such as logistic regression
- Based on empirical results, this model was excluded from the final ensemble due to underperformance in probability calibration

#### Ensemble Methods (`/notebooks/ensemble/`)
Our final prediction strategy combines:
- Stacked generalization across all model types
- Weighted averaging based on historical model performance
- Bayesian model averaging for improved probability calibration
- Meta-model training using leave-one-tournament-out validation
- Confidence-based prediction adjustment

## Results and Findings

The `notebooks/final_model/` directory contains our championship model that achieved:
- Superior log loss scores compared to baseline models
- Balanced accuracy between favorites and potential upset predictions
- Reliable confidence calibration validated through Brier score analysis
- Effective identification of key statistical signals for tournament success
- Robust performance across different tournament scenarios

### Key Insights
Our modeling revealed several interesting patterns in tournament prediction:
- Late-season performance trends provide stronger signals than full-season averages
- Team experience in close games correlates with tournament success
- Specific conference matchups show persistent historical patterns
- Pace-adjusted statistics offer superior predictive power compared to raw metrics

## Data Sources
The competition data includes:
- Regular season and tournament game results
- Team box scores and advanced metrics
- Historical seed performance
- Coaching statistics
- Tournament seeding information
