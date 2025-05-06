# March Machine Learning Mania 2024

## Project Overview
This repository contains machine learning models and analysis for the March Machine Learning Mania 2024 competition hosted on Kaggle. The project predicts NCAA Men's Basketball Tournament outcomes using historical tournament data, team statistics, and advanced machine learning techniques to forecast game winners and tournament progression.

## Competition Context
March Machine Learning Mania challenges data scientists to predict the outcomes of the NCAA Men's Basketball Tournament. The competition evaluates predictions using log loss scoring against actual tournament results, rewarding both accurate predictions and appropriate confidence levels.

## Repository Structure

```
.
├── data/                         # Tournament data
│   └── march-machine-learning-mania-2024.zip
├── models/                       # Saved model files
├── notebooks/                    # Jupyter notebooks for analysis and modeling
│   ├── EDA/                      # Exploratory Data Analysis
│   ├── ensemble/                 # Ensemble modeling approaches
│   ├── logistic_regression/      # Logistic Regression models
│   ├── neural_networks/          # Neural Network models
│   │   └── nn_tuner_dir/         # Neural Network hyperparameter tuning
│   ├── random_forest/            # Random Forest models
│   └── xgboost/                  # XGBoost models
└── scripts/                      # Utility scripts
    └── pre_processing.py         # Data preprocessing functionality
```


## Running the Project
You can run this project either locally on your machine or within a Kaggle Notebook. For local use, simply clone the repository and ensure all dependencies are installed.

To run the project in a Kaggle environment, make sure to upload `/scripts/pre_processing.py` as a Kaggle Dataset, and name it "preprocessing-module". Then, attach this dataset to your Kaggle Notebook. This step is necessary because Kaggle requires external Python modules to be attached as datasets for import.
Next, you must also upload all the 9 csv prediction files in the directory `/predictions/` as a Kaggle Dataset, and name it "predictions".

## Modeling Approach

### Data Preparation and Exploration
Our `/notebooks/EDA/eda_march_madness.ipynb` notebook provides comprehensive exploratory data analysis including:
- Historical tournament performance patterns
- Team statistical features correlation with winning probability
- Season-to-season consistency analysis
- Feature importance visualization for predictive modeling
- Identification of upset patterns and their statistical signatures

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

#### Neural Networks (`/notebooks/neural_networks/`)
The deep learning approach features:
- Multi-layer perceptron architecture optimized for tournament prediction
- Hyperparameter tuning (documented in `nn_tuner_dir`)
- Dropout regularization to improve generalization
- Custom loss functions prioritizing upset detection
- Embedding layers for team identity representation

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
