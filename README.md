# March Machine Learning Mania 2024

## Project Overview
This repository presents a full data science pipeline developed for the [Kaggle March Machine Learning Mania 2024](https://www.kaggle.com/competitions/march-machine-learning-mania-2024) competition.
The objective is to predict outcomes of NCAA Men's and Women's Basketball Tournament games using a combination of historical data, statistical features, and diverse machine learning techniques.

Key elements of the project include:
- Preprocessing pipeline in [`scripts/pre_processing.py`](scripts/pre_processing.py)
- Exploratory Data Analysis in [`notebooks/EDA/`](notebooks/EDA/)
- Model development using Logistic Regression, Random Forest, XGBoost, and Neural Networks
- Ensemble modeling strategy combining predictions across models
- Evaluation using **Brier score**, the official metric of the competition

Our final ensemble model outperformed all single models individually, demonstrating the power of stacked generalization and weighted averaging for tournament outcome prediction.

## Competition Context
The **March Machine Learning Mania** competition challenges participants to forecast the results of NCAA Men's and Women's Basketball Tournaments. Participants submit a portfolio of brackets (between 1 and 100,000) based on probabilistic predictions, which are evaluated using the **Brier score** metric — rewarding both accurate predictions and appropriate confidence levels.

The competition required forecasting outcomes for both tournaments by submitting bracket portfolios that reflect probability estimates for each team advancing through each round. Our approach focused on developing a robust probabilistic framework rather than simply predicting winners.

The tournament's unpredictable nature, with frequent upsets and underdog victories (aptly named "March Madness"), makes this a complex and rewarding modeling challenge, ideal for testing robust classification approaches under real-world uncertainty.

### Evaluation Metric
Submissions were evaluated by their Average Brier Bracket Score, which:
1. Computes the implied probability a team wins each round based on the portfolio of brackets
2. Evaluates these probabilities against the ground truth using Brier score for each round
3. Takes the mean of six Brier scores to compute the overall score

For our model development, we used the standard Brier score to evaluate individual game predictions, which translates well to the competition's bracket-level evaluation.

## Repository Structure
```bash
.
├── data/                         # Tournament data
│   └── march-machine-learning-mania-2024.zip
├── notebooks/                    # Jupyter notebooks for analysis and modeling
│   ├── EDA/                      # Exploratory Data Analysis
│   ├── ensemble/                 # Ensemble modeling approaches
│   ├── logistic_regression/     # Logistic Regression models
│   ├── neural_networks/         # Neural Network models
│   ├── random_forest/           # Random Forest models
│   └── xgboost/                 # XGBoost models
├── predictions/                  # CSV prediction files from model runs
│   ├── predictions_2022_lr.csv  # Logistic regression predictions for 2022
│   ├── predictions_2022_rf.csv  # Random forest predictions for 2022
│   ├── predictions_2022_xgb.csv # XGBoost predictions for 2022
│   ├── predictions_2023_lr.csv  # Logistic regression predictions for 2023
│   ├── predictions_2023_rf.csv  # Random forest predictions for 2023
│   ├── predictions_2023_xgb.csv # XGBoost predictions for 2023
│   ├── predictions_2024_lr.csv  # Logistic regression predictions for 2024
│   ├── predictions_2024_rf.csv  # Random forest predictions for 2024
│   └── predictions_2024_xgb.csv # XGBoost predictions for 2024
└── scripts/                      # Utility scripts
    └── pre_processing.py        # Data preprocessing functionality
```

## Methodology

### Data Sources and Preparation
We utilized NCAA datasets containing:
- Regular-season and tournament results spanning multiple years
- Team identifiers and seed information
- Box score statistics (field goals, rebounds, assists, etc.)

Key preparation steps included:
- Parsing textual seeds (e.g., "W16a") into numeric form
- Validating completeness of essential fields (scores, IDs, seeds)
- Aggregating team-level season statistics
- Initially keeping men's and women's data separate for analysis before combining them

### Exploratory Data Analysis (EDA)
Our EDA revealed important insights:
- Tournament teams consistently showed higher win rates and scoring margins than non-qualified teams
- Lower seeds (better ranked teams) strongly correlate with stronger regular season performance
- Game statistics like assist differential, turnover differential, and rebound differential showed interesting patterns in relation to game outcomes
- Statistical features needed to be engineered to encode team strength and matchup quality effectively

### Feature Engineering
Based on EDA insights, we constructed a matchup-level dataset with two types of features:

**Core Comparative Features**:
- Win percentage (WinPct)
- Average scoring margin (AvgMargin)
- Numeric seed value (SeedNum)
- Derived difference metrics (SeedDiff, WinPctDiff, AvgMarginDiff)

**Advanced Performance Metrics**:
- Elo ratings: Custom system that updates team ratings based on game outcomes and opponent strength
- GLM-based team quality: Estimated using a Generalized Linear Model on point differentials
- Short-term momentum indicators (Last14WinRate)
- Neutral-site performance metrics

To address team order bias, we applied a symmetrization strategy—duplicating each game with reversed team roles and negated feature values—doubling the training data and preventing the model from overfitting to input ordering.

### Modeling Approach
To ensure consistency across models, we implemented rolling-window cross-validation over 2011–2021 (excluding 2020), training each model on data up to year t-1 and testing on year t. We used Brier score as our primary metric and reserved 2022 and 2023 as untouched validation sets for final evaluation.

#### Logistic Regression
Our baseline model provided interpretability and strong performance:
- Started with core features derived from regular-season data: SeedDiff, WinPctDiff, AvgMarginDiff
- Added EloDiff after feature selection showed its importance
- Applied symmetrization and MinMax scaling
- Tuned regularization parameters with Optuna, finding L2 regularization with C ≈ 17.5 optimal
- Achieved a Brier score of 0.174 on the test years

#### XGBoost Tournament Predictor
The workhorse of our bracket-forecasting system:
- Leveraged detailed box-score features (45 initial variables)
- Used SHAP analysis to identify the most informative predictors
- Applied redundancy pruning (removing highly correlated features)
- Employed forward feature selection to optimize the feature set
- Fine-tuned hyperparameters (tree depth, learning rate, min child weight, etc.)
- Achieved a Brier score of 0.176 and 72.7% accuracy

#### Random Forest Classifier
A fast, reliable alternative model:
- Utilized compact regular-season data
- Engineered features such as Last14WinRate to capture momentum
- Applied forward feature selection and hyperparameter tuning
- Achieved a Brier score of 0.173 with 15× faster training than XGBoost

#### Neural Network Attempt
While we experimented with a feedforward neural network, it was ultimately excluded due to:
- Limited training data (under 2000 labeled tournament games)
- High variability across cross-validation folds
- Lower overall performance compared to simpler models
- Lack of interpretability

### Final Ensemble Model
Our final solution combined models to capitalize on their complementary strengths:

1. **Methodology**: We used weighted-average ensemble, searching over the 3-model simplex to minimize hold-out Brier score:
   ```
   P_ensemble = wLR * P_LR + wRF * P_RF + wXGB * P_XGB, where wLR + wRF + wXGB = 1
   ```

2. **Weight Optimization**: Grid search (0.02-step grid) over all possible weight combinations yielded optimal weights:
   ```
   (wLR, wRF, wXGB) = (0.30, 0.00, 0.70)
   ```

3. **Performance**: The ensemble yielded a Brier score of 0.18793, representing:
   - 0.21% reduction in squared-error loss compared to XGBoost alone (0.18833)
   - 1.2% reduction compared to Logistic Regression (0.19027)

4. **Bracket Generation**: We generated 50,000 brackets for men and 50,000 for women via Monte Carlo sampling, applying historical realism constraints (e.g., limiting how far lower seeds can advance)

## Results and Conclusions

Our final approach achieved a competition score of 0.0611, placing us in the top 150 teams.

Key insights from the project:
- **Feature engineering trumped model choice**: The four difference features (Seed, Win%, Margin, Elo) explained over 80% of signal and transferred well across models
- **Calibration was critical**: Reliable probabilities reduced per-game Brier score by ~1.4 percentage points
- **Ensemble benefits**: A 30% LR + 70% XGB blend outperformed either model alone while dropping RF entirely
- **Cross-validation strategy matters**: Rolling-window CV prevented hindsight bias and produced realistic results
- **Self-contained approach**: Using only historical data (no external sources) still placed us in the top 9% of competitors

### Future Enhancements
For future competitions, we identified potential improvements:
- Incorporating external power ratings (such as Nate Silver's 538 ratings)
- Developing real-time injury and availability tracking
- Expanding the feature set with possession-level efficiency metrics
- Implementing Bayesian hierarchical models to better account for season-to-season variability

## Running the Project
You can run this project either locally on your machine or within a Kaggle Notebook.

### Local Execution
1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebooks in the `/notebooks/` directory in the following recommended order:
   - Start with EDA notebooks to understand the data
   - Execute model notebooks (logistic_regression, random_forest, xgboost)
   - Run the ensemble notebook to combine model predictions

### Kaggle Execution
To run the project in a Kaggle environment:

1. Upload `/scripts/pre_processing.py` as a Kaggle Dataset, and name it "preprocessing-module". Then, attach this dataset to your Kaggle Notebook. This step is necessary because Kaggle requires external Python modules to be attached as datasets for import.

2. Upload the CSV prediction files to Kaggle. Each of the model notebooks (logistic_regression, random_forest, xgboost) will create their own set of three CSV files in the `/predictions/` directory when they run. Each model creates prediction files for three years (2022, 2023, and 2024). These 9 CSV files need to be uploaded to Kaggle as a dataset named "predictions".

3. The ensemble notebook will use these prediction files to create the final tournament predictions and bracket simulations.

## Dependencies
All required dependencies are listed in the `requirements.txt` file. The main libraries used include:
- pandas and numpy for data handling
- scikit-learn for machine learning models
- xgboost for gradient boosting
- tensorflow for neural networks
- matplotlib and seaborn for visualization