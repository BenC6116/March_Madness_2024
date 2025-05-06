import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


# ==============================================================================
# External Functions (Public API)
# These functions are intended for use by external modules.
# ==============================================================================


# ==========================================================================
# PUBLIC helper – create full matchup feature matrix for a single season
# ==========================================================================

def generate_matchup_matrix(model_type: str,
                            season: int,
                            gender: str = 'M',
                            data_dir: str | None = None):
    """
    Build a feature matrix for **all pair-wise combinations** of seeded teams
    in `season`, using *exactly* the same engineering pipeline as training.

    Parameters
    ----------
    model_type : {'lr', 'lr_basic', 'rf', 'xgb'}
        Determines which columns are ultimately returned.
    season : int
        Target season (e.g. 2024).
    gender : {'M', 'W'}
        Men's or Women's bracket to consider.
    data_dir : str | None
        Optional root directory that holds the CSV files / zip.

    Returns
    -------
    (X, game_ids) : (pd.DataFrame, pd.Series)
        X   – dataframe ready for inference (only the core predictor columns).  
        game_ids – synthetic IDs uniquely identifying each matchup
                   (same convention as `_generate_game_ids`).
    """
    if model_type not in ('lr', 'lr_basic', 'rf', 'xgb'):
        raise ValueError("model_type must be 'lr', 'lr_basic', 'rf', or 'xgb'")

    # ----------------------------------------------------------------------
    # 1. Get seed list for the requested season / gender
    # ----------------------------------------------------------------------
    # Get model parameters for the specified model type
    params = _get_model_parameters(model_type)
    
    std_data, _ = _prepare_all(data_dir=data_dir, combine=False,
                               gender=gender, detailed=params['detailed'],
                               reorder=True, symmetry=False)
    seeds_df = _process_seeds(std_data['Seeds'], split=False)
    season_seeds = seeds_df.query("Season == @season")['TeamID'].unique()
    if len(season_seeds) == 0:
        raise ValueError(f"No seeds found for {gender} {season}")

    # ----------------------------------------------------------------------
    # 2. Fabricate a neutral-site TourneyResults skeleton with *all* pairings
    # ----------------------------------------------------------------------
    from itertools import combinations
    fake_rows = [{
        'Season': season,
        'DayNum': 136,          # arbitrary but consistent
        'Team1ID': t1, 'Team1Score': 0,
        'Team2ID': t2, 'Team2Score': 0,
        'Location': 'N', 'NumOT': 0
    } for t1, t2 in combinations(sorted(season_seeds), 2)]
    fake_tourney = pd.DataFrame(fake_rows)

    # ----------------------------------------------------------------------
    # 3. Run entire feature pipeline on this synthetic tourney frame
    # ----------------------------------------------------------------------
    feats = _build_full_feature_set(
        data_dir=data_dir, combine=False, gender=gender,
        detailed=params['detailed'], reorder=True, symmetry=params['symmetry'],
        symmetric_cols=params['symmetric_cols'],
        tourney_override=fake_tourney
    )

    # ----------------------------------------------------------------------
    # 4. Extract model-specific feature sets
    # ----------------------------------------------------------------------
    core_cols = _get_model_core_features(model_type)
    
    # Filter only available columns (in case some are missing)
    available_cols = [col for col in core_cols if col in feats.columns]
    if not available_cols:
        raise ValueError(f"None of the required columns for model_type '{model_type}' are available in the generated features")
    
    X = feats[available_cols].copy()
    return X, feats['GameID']


def build_model_pipeline(model_type, data_dir=None, combine=True, gender=None):
    """
    Builds a model-specific dataset based on the provided model type and configuration.
    
    Parameters:
        model_type (str): Type of model pipeline to build.
        data_dir (str, optional): Directory containing competition data.
        combine (bool): If True, combine men's and women's data.
        gender (str, optional): Used if combine is False; 'M' for men's or 'W' for women's.
    
    Returns:
        For 'lr' or 'lr_basic': DataFrame with features and target 'Team1Wins'.
        Otherwise: Tuple (X, y) where X is features DataFrame and y is target.
    """
    # Check if model_type is valid
    valid_types = ['eda',"eda_detailed", 'lr', 'lr_basic', 'rf', 'nn', 'xgb']
    if model_type not in valid_types:
        raise ValueError(
            "Unknown model_type. Choose from 'lr', 'rf', 'ensemble', 'nn', or 'xgb'."
        )
    
    # If exploratory analysis is requested, return raw standardized data
    if model_type == 'eda':
        return _prepare_all(data_dir=data_dir, combine=combine,
                            symmetry=True, raw_data=True)
    
    if model_type == 'eda_detailed':
        return _prepare_all(data_dir=data_dir, combine=combine,
                            symmetry=True, raw_data=True, detailed=True)
        
    # Validate gender/combine configuration
    if (not combine and gender not in ('M', 'W')) or (combine and gender is not None):
        raise ValueError(
            "Invalid gender/combine combo: gender must be 'M' or 'W' if combine=False, "
            "else None if model_type != 'eda'."
        )
    
     # Get model parameters from helper function
    params = _get_model_parameters(model_type)
    
    # Build the full feature set by processing raw data and merging engineered features
    full_df = _build_full_feature_set(
        data_dir=data_dir,
        combine=combine,
        gender=gender,
        detailed=params['detailed'],
        reorder=params['reorder'],
        symmetry=params['symmetry'],
        symmetric_cols=params['symmetric_cols']
    )
    
    # Create a unique GameID for each row based on game information
    if 'GameID' not in full_df.columns:
        full_df['GameID'] = _generate_game_ids(full_df)
    
    # Get model-specific features from helper function
    selected_features = _get_model_features(model_type)
    
    # Extract feature matrix X and target variable y
    X = full_df[selected_features].copy()
    y = full_df['Team1Wins']
    # Get GameIDs corresponding to the rows in X
    game_ids = full_df.loc[X.index, 'GameID']
    
    return X, y, game_ids


def rolling_window_cv(X, y, start_season, end_season=2021, window_size=None, model_fn=None,
                      model_params=None, sample_weight_fn=None, verbose=True, return_preds=False):
    """
    Implements rolling window cross-validation based on the 'Season' column.
    
    Parameters:
        X (DataFrame): Feature DataFrame that includes a 'Season' column.
        y (Series or DataFrame): Target variable corresponding to X.
        start_season (int): The starting season for cross-validation.
        end_season (int, optional): Last season for testing.
        window_size (int, optional): Number of seasons to use for training.
        model_fn (callable): Function that trains the model and returns performance metrics.
        model_params (dict, optional): Parameters for the model function.
        sample_weight_fn (callable, optional): Function to compute sample weights.
        verbose (bool, optional): If True, prints detailed progress.
        return_preds (bool, optional): If True, include predictions in metrics. 
                                      Model function must return 'predictions' in metrics.
    
    Returns:
        DataFrame: Performance metrics indexed by test season.
    """
    results = {}
    # Ensure that the 'Season' column is present in the feature DataFrame
    if 'Season' not in X.columns:
        raise ValueError("X must contain a 'Season' column for rolling window evaluation.")
    
    # Determine the maximum season in the dataset
    # max_season = X['Season'].max()
    max_season = end_season
    for year in range(start_season, max_season):
        test_season = year + 1  # The season immediately after the training window
        if test_season not in X['Season'].unique():
            continue  # Skip if test season is not present
        
        # Define the start of the training window
        train_start = X['Season'].min() if window_size is None else max(
            X['Season'].min(), year - window_size + 1
        )
        
        # Select training and test data based on seasons
        X_train_full = X[(X['Season'] >= train_start) & (X['Season'] <= year)].copy()
        y_train = y[(X['Season'] >= train_start) & (X['Season'] <= year)].copy()
        X_test_full = X[X['Season'] == test_season].copy()
        y_test = y[X['Season'] == test_season].copy()
        
        # Compute sample weights if a function is provided
        sample_weight = (sample_weight_fn(X_train_full, test_season)
                         if sample_weight_fn is not None else None)
        
        # Remove the 'Season' column from features for model training
        X_train = X_train_full.drop(columns=['Season'], errors='ignore')
        X_test = X_test_full.drop(columns=['Season'], errors='ignore')
        
        # Call the provided model function to train and evaluate the model
        metrics = model_fn(X_train, y_train, X_test, y_test, test_season,
                           model_params, sample_weight)
        
        # Add the test season to metrics
        metrics['test_season'] = test_season
        
        # If return_preds is False and 'predictions' exists in metrics, remove it
        if not return_preds and 'predictions' in metrics:
            metrics.pop('predictions')
            
        results[test_season] = metrics
        
        # If verbose, print out metrics for the current season
        if verbose:
            print(
                f"Season {test_season} – Accuracy: {metrics['accuracy']:.3f}, "
                f"Log Loss: {metrics['logloss']:.3f}, Brier: {metrics['brier']:.3f}"
            )
    
    # Convert the results dictionary into a DataFrame
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    metrics_df.index.name = 'test_season'
    if verbose and not metrics_df.empty:
        avg_accuracy = metrics_df['accuracy'].mean()
        avg_logloss = metrics_df['logloss'].mean()
        avg_brier = metrics_df['brier'].mean()
        print(
            f"\nOverall average metrics: Accuracy: {avg_accuracy:.3f}, "
            f"Log Loss: {avg_logloss:.3f}, Brier: {avg_brier:.3f}"
        )
    return metrics_df

def get_y_true(game_ids_df):
    """
    Return the historical outcome (y_true) for each Season / GameID pair.

    The GameID convention used by generate_matchup_matrix is:
        GameID = Season · 1e8  +  min(TeamID) · 1e4  +  max(TeamID)

    Therefore the lower-numbered team is always Team1.  We reproduce
    that exact orientation here before we compute the label.

    Parameters
    ----------
    game_ids_df : pd.DataFrame
        Must contain columns ['Season', 'GameID'].

    Returns
    -------
    pd.DataFrame
        Columns ['Season', 'GameID', 'y_true'] where
        y_true = 1  if Team1 won
               = 0  if Team2 won
               = NaN if the matchup never happened.
    """
    # ------------------------------------------------------------------
    # 1. sanity-check input
    # ------------------------------------------------------------------
    required = {'Season', 'GameID'}
    if required - set(game_ids_df.columns):
        raise ValueError("game_ids_df must contain 'Season' and 'GameID' columns")

    # ------------------------------------------------------------------
    # 2. load raw tournament results (men + women combined)
    # ------------------------------------------------------------------
    std_data, _ = _prepare_all(
        data_dir=None,
        combine=True,          # men & women together
        detailed=False,
        reorder=False,         # we’ll reorder manually next
        symmetry=False         # one row per real game
    )
    tourney_results = std_data['TourneyResults'].copy()

    # ------------------------------------------------------------------
    # 3. standardise *and* reorder so lower TeamID is always Team1
    # ------------------------------------------------------------------
    _standardize_results(tourney_results, detailed=False, reorder=True)

    # ------------------------------------------------------------------
    # 4. rebuild GameID and compute the true label
    # ------------------------------------------------------------------
    tourney_results['GameID'] = _generate_game_ids(tourney_results)
    tourney_results['y_true'] = (
        tourney_results['Team1Score'] > tourney_results['Team2Score']
    ).astype(int)

    tourney_results = tourney_results[['Season', 'GameID', 'y_true']]

    # ------------------------------------------------------------------
    # 5. align dtypes for the merge (string vs int GameID)
    # ------------------------------------------------------------------
    if tourney_results['GameID'].dtype != game_ids_df['GameID'].dtype:
        tourney_results['GameID'] = tourney_results['GameID'].astype(str)
        lookup = game_ids_df.copy()
        lookup['GameID'] = lookup['GameID'].astype(str)
    else:
        lookup = game_ids_df

    # ------------------------------------------------------------------
    # 6. merge and return
    # ------------------------------------------------------------------
    merged = pd.merge(lookup, tourney_results,
                      on=['Season', 'GameID'], how='left')

    return merged[['Season', 'GameID', 'y_true']]
# ==============================================================================
# Internal Functions (Not meant for external use)
# These functions perform data loading, processing, and feature engineering.
# ==============================================================================

def _get_model_parameters(model_type):
    """
    Returns model-specific processing parameters.
    
    Parameters
    ----------
    model_type : str
        The type of model ('lr', 'lr_basic', 'rf', 'nn', 'xgb', or 'eda').
        
    Returns
    -------
    dict
        Dictionary with processing parameters for the specified model type.
    """
    params = {
        'detailed': False,
        'symmetric_cols': False,
        'symmetry': True,
        'reorder': False
    }
    
    # Set common parameters based on model type
    if model_type == 'nn':
        params['detailed'] = True
    elif model_type == 'xgb':
        params['detailed'] = True
        params['symmetric_cols'] = True
    elif model_type == 'lr':
        params['reorder'] = True
    elif model_type == 'lr_basic':
        params['reorder'] = True
        params['symmetry'] = False
    
    return params


def _get_model_features(model_type):
    """
    Returns model-specific feature lists.
    
    Parameters
    ----------
    model_type : str
        The type of model ('lr', 'lr_basic', 'rf', 'nn', 'xgb').
        
    Returns
    -------
    list
        List of feature column names for the specified model type.
    """
    if model_type == 'lr':
        return [
            'Season', 'Team1ID', 'Team2ID', 'Team1Seed', 'Team2Seed',
            'Team1WinPct', 'Team2WinPct', 'Team1AvgMargin', 'Team2AvgMargin',
            'SeedDiff', 'WinPctDiff', 'AvgMarginDiff', 'Team1Elo', 'Team2Elo', 'EloDiff'
        ]
    elif model_type == 'lr_basic':
        return [
            'Season', 'Team1ID', 'Team2ID', 'Team1Seed', 'Team2Seed',
            'Team1WinPct', 'Team2WinPct', 'Team1AvgMargin', 'Team2AvgMargin',
            'SeedDiff', 'WinPctDiff', 'AvgMarginDiff'
        ]
    elif model_type == 'rf':
        return [
            'Season', 'SeedDiff', 'WinPctDiff', 'PointsForDiff',
            'PointsAgainstDiff', 'AvgMarginDiff', 'Team1Games', 'Team2Games',
            'Last14WinRateDiff', 'NeutralWinRateDiff'
        ]
    elif model_type == 'nn':
        return [
            'Season',
            'Team1WinPct', 'Team2WinPct',
            'Team1Last14WinRate', 'Team2Last14WinRate',
            'Team1Quality', 'Team2Quality',
            'SeedDiff'
        ]
    elif model_type == 'xgb':
        return [
            'Season', 'Team1FGM', 'Team1FGA', 'Team1FGM3', 'Team1FGA3', 'Team1OR', 'Team1Ast',
            'Team1TO', 'Team1Stl', 'Team1PF',
            'Team2_opponent_FGM', 'Team2_opponent_FGA', 'Team2_opponent_FGM3', 'Team2_opponent_FGA3',
            'Team2_opponent_OR', 'Team2_opponent_Ast', 'Team2_opponent_TO', 'Team2_opponent_Stl',
            'Team2_opponent_PF',
            'Team1PointDiff',
            'Team2FGM', 'Team2FGA', 'Team2FGM3', 'Team2FGA3', 'Team2OR', 'Team2Ast',
            'Team2TO', 'Team2Stl', 'Team2PF',
            'Team1_opponent_FGM', 'Team1_opponent_FGA', 'Team1_opponent_FGM3', 'Team1_opponent_FGA3',
            'Team1_opponent_OR', 'Team1_opponent_Ast', 'Team1_opponent_TO', 'Team1_opponent_Stl',
            'Team1_opponent_PF',
            'Team2PointDiff',
            'Team1Seed', 'Team2Seed', 'Team1Last14WinRate', 'Team2Last14WinRate', 'SeedDiff',
            'Team1Quality', 'Team2Quality'
        ]
    else:
        return []

def _get_model_core_features(model_type):
    """
    Returns the core features needed for model inference (without metadata columns).
    Used by generate_matchup_matrix to select only prediction-relevant features.
    
    Parameters
    ----------
    model_type : str
        The type of model ('lr', 'lr_basic', 'rf', 'xgb').
        
    Returns
    -------
    list
        List of core feature column names for the specified model type.
    """
    if model_type == 'lr':
        return ['SeedDiff', 'WinPctDiff', 'AvgMarginDiff', 'EloDiff']
    elif model_type == 'lr_basic':
        return ['SeedDiff', 'WinPctDiff', 'AvgMarginDiff']
    elif model_type == 'rf':
        return [
            'SeedDiff', 'WinPctDiff', 'PointsForDiff',
            'PointsAgainstDiff', 'AvgMarginDiff', 'Team1Games', 'Team2Games',
            'Last14WinRateDiff', 'NeutralWinRateDiff'
        ]
    elif model_type == 'xgb':
        return _get_model_features('xgb')  # XGBoost uses all features
    else:
        return []

def _resolve_data_dir(data_dir):
    """
    Determines the directory where the competition data is located.
    
    Parameters:
        data_dir (str or None): User-specified data directory.
    
    Returns:
        str: The resolved directory path, or None if not found.
    """
    # If a data_dir is provided, return it immediately
    if data_dir is not None:
        return data_dir
    # Check for a known Kaggle input path
    if os.path.exists('/kaggle/input'):
        return '/kaggle/input/march-machine-learning-mania-2024'
    # Attempt to resolve from the project structure
    try:
        project_root = Path(__file__).resolve().parent.parent
    except Exception:
        project_root = Path.cwd().resolve().parent
    local_dir = project_root / 'data' / 'march-machine-learning-mania-2024'
    return str(local_dir) if local_dir.exists() else None


def _load_csv(path):
    """
    Loads a CSV file from the specified path.
    
    Parameters:
        path (str): Full path to the CSV file.
    
    Returns:
        DataFrame: Loaded CSV data, or an empty DataFrame if an error occurs.
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f'Error loading {path}: {e}')
        return pd.DataFrame()


def _load_from_zip(files):
    """
    Loads competition data from a zip archive.
    
    Parameters:
        files (dict): Mapping of keys to expected CSV filenames in the zip archive.
    
    Returns:
        dict: A dictionary mapping keys to DataFrames loaded from the CSV files.
    """
    try:
        project_root = Path(__file__).resolve().parent.parent
    except Exception:
        project_root = Path.cwd().resolve().parent
    zip_path = project_root / 'data' / 'march-machine-learning-mania-2024.zip'
    if not zip_path.exists():
        raise FileNotFoundError(f'Zip file not found at {zip_path}')
    data = {}
    # Open the zip file and iterate over the expected files
    with zipfile.ZipFile(zip_path, 'r') as z:
        for key, fname in files.items():
            try:
                with z.open(fname) as f:
                    data[key] = pd.read_csv(f)
            except Exception as e:
                print(f'Error loading {fname} from zip: {e}')
                data[key] = pd.DataFrame()
    return data


def _process_seeds(seeds, split=False):
    """
    Processes the seeds DataFrame by extracting numeric seed values.
    
    Parameters:
        seeds (DataFrame): DataFrame containing a 'Seed' column.
        split (bool): If False, returns the DataFrame with an added 'SeedNum' column.
                      If True, splits the DataFrame into two:
                        - One with columns ['Season', 'Team1ID', 'Team1Seed']
                        - One with columns ['Season', 'Team2ID', 'Team2Seed']
    
    Returns:
        If split is False: DataFrame with a new 'SeedNum' column.
        If split is True: Tuple (seeds_team1, seeds_team2).
    """
    if not split:
        # Ensure 'Seed' column exists
        if 'Seed' not in seeds.columns:
            raise KeyError("Expected column 'Seed' not found in seeds DataFrame.")
        seeds = seeds.copy()
        # Use regex to extract numeric digits from the seed string
        extracted = seeds['Seed'].astype(str).str.strip().str.extract(r'(\d+)')
        if extracted.empty or extracted.shape[1] == 0:
            raise ValueError("Could not extract digits from Seed column.")
        seeds['SeedNum'] = extracted.iloc[:, 0].astype(int)
        return seeds
    else:
        # Use slicing to extract numeric seed; assumes a fixed format
        seeds = seeds.copy()
        seeds['seed'] = seeds['Seed'].apply(lambda x: int(x[1:3]))
        # Split into two DataFrames for Team1 and Team2
        seeds_team1 = seeds[['Season', 'TeamID', 'seed']].copy()
        seeds_team2 = seeds[['Season', 'TeamID', 'seed']].copy()
        seeds_team1.columns = ['Season', 'Team1ID', 'Team1Seed']
        seeds_team2.columns = ['Season', 'Team2ID', 'seed']
        seeds_team2.rename(columns={'seed': 'Team2Seed'}, inplace=True)
        return seeds_team1, seeds_team2


def _prepare_all(data_dir=None, combine=True, gender=None, detailed=False,
                 reorder=False, symmetry=True, raw_data=False):
    """
    Loads and processes all competition data into a standardized format.
    
    Parameters:
        data_dir (str, optional): Directory path to the data files.
        combine (bool): If True, combine men's and women's data.
        gender (str, optional): If combine is False, select data for 'M' or 'W'.
        detailed (bool): If True, use detailed game results.
        reorder (bool): If True, reorder team IDs so the lower is Team1.
        symmetry (bool): If True, create swapped copies for data augmentation.
        raw_data (bool): If True, return raw standardized data (for EDA).
    
    Returns:
        If raw_data is True:
            dict: Raw data dictionary.
        Otherwise:
            tuple: (standardized_data, team_stats)
    """
    raw = _load_competition_data(data_dir, combine=combine, detailed=detailed)
    if raw_data:
        # Process seeds for raw data output
        raw['M']['Seeds'] = _process_seeds(raw['M']['Seeds'], split=False)
        raw['W']['Seeds'] = _process_seeds(raw['W']['Seeds'], split=False)
        return raw
    elif not combine:
        raw = raw[gender]
    
    # Standardize column names for both regular and tournament results
    _standardize_results(raw['RegResults'])
    _standardize_results(raw['TourneyResults'], detailed=detailed, reorder=reorder)
    
    std = raw.copy()
    # Process regular season data and compute team statistics
    std['RegResults'], team_stats = _prepare_data(
        raw['RegResults'], detailed=detailed, symmetry=symmetry, reg=True
    )
    # Process tournament data
    std['TourneyResults'] = _prepare_data(
        raw['TourneyResults'], detailed=detailed, symmetry=symmetry
    )
    # Process seeds to add numeric seed column
    std['SeedNum'] = _process_seeds(raw['Seeds'], split=False)
    return std, team_stats


def _build_full_feature_set(data_dir=None, combine=True, gender=None,
                            detailed=False, reorder=False, symmetry=True,
                            symmetric_cols=False,
                            tourney_override=None):
    """
    Builds the complete feature matrix by merging various statistics and engineered features.

    Parameters
    ----------
    data_dir : str | None
        Directory path to the data.
    combine : bool
        If True, combines men's and women's data.
    gender : str | None
        If combine is False, selects gender-specific data.
    detailed : bool
        If True, uses detailed game statistics.
    reorder : bool
        If True, applies team reordering.
    symmetry : bool
        If True, includes swapped game copies.
    symmetric_cols : bool
        If True, renames aggregated stats with team-specific prefixes.
    tourney_override : pd.DataFrame | None
        Supply a *pre-fabricated* tournament-style dataframe (same schema as
        compact results) to build features for seasons with no official
        TourneyResults (e.g. 2024 predictions).

    Returns
    -------
    pd.DataFrame
        Full feature matrix with engineered columns.
    """
    # ----------------------------------------------------------------------
    # 1. Load & standardise all raw data
    # ----------------------------------------------------------------------
    data, team_stats = _prepare_all(
        data_dir=data_dir, combine=combine, gender=gender,
        detailed=detailed, reorder=reorder, symmetry=symmetry
    )

    # --- optional replacement of the tournament frame ---------------------
    if tourney_override is not None:
        data['TourneyResults'] = tourney_override.copy()

    # ----------------------------------------------------------------------
    # 2. Compute auxiliary stats (season means, last-14 win-rates, Elo, ...)
    # ----------------------------------------------------------------------
    season_stats_team1, season_stats_team2 = _compute_season_statistics(
        data['RegResults'], symmetric_cols, detailed=detailed
    )
    last14_team1, last14_team2 = _compute_last14day_win_ratios(data['RegResults'])
    team1_neutral, team2_neutral = _compute_neutral_win_rate(data['RegResults'])
    elo_team1, elo_team2 = _calculate_elo(data['RegResults'])

    # Merge helpers into (possibly overridden) tourney dataframe
    tourney_data = _merge_df_with_stats(data['TourneyResults'],
                                        last14_team1, last14_team2)
    tourney_data = _merge_df_with_stats(tourney_data, elo_team1, elo_team2)
    tourney_data = _merge_df_with_stats(tourney_data, team1_neutral, team2_neutral)

    # ----------------------------------------------------------------------
    # 3. Add aggregated regular-season team stats
    # ----------------------------------------------------------------------
    t1_stats = team_stats.rename(
        columns=lambda x: x if x in ['Season', 'TeamID'] else f'Team1{x}'
    )
    tourney_data = tourney_data.merge(
        t1_stats, left_on=['Season', 'Team1ID'], right_on=['Season', 'TeamID'],
        how='left'
    ).drop('TeamID', axis=1)

    t2_stats = team_stats.rename(
        columns=lambda x: x if x in ['Season', 'TeamID'] else f'Team2{x}'
    )
    tourney_data = tourney_data.merge(
        t2_stats, left_on=['Season', 'Team2ID'], right_on=['Season', 'TeamID'],
        how='left'
    ).drop('TeamID', axis=1)

    # ----------------------------------------------------------------------
    # 4. Regular-season effects & quality estimates
    # ----------------------------------------------------------------------
    regular_season_effects = _compute_regular_season_effects(
        data['RegResults'], data['Seeds']
    )
    glm_quality = _compile_team_quality(regular_season_effects)

    glm_quality_team1 = glm_quality.copy()
    glm_quality_team2 = glm_quality.copy()
    glm_quality_team1.columns = ['Team1ID', 'Team1Quality', 'Season']
    glm_quality_team2.columns = ['Team2ID', 'Team2Quality', 'Season']

    tourney_data = pd.merge(tourney_data, glm_quality_team1,
                            on=['Season', 'Team1ID'], how='left')
    tourney_data = pd.merge(tourney_data, glm_quality_team2,
                            on=['Season', 'Team2ID'], how='left')

    # ----------------------------------------------------------------------
    # 5. Add seeds & optional symmetric-column aggregation
    # ----------------------------------------------------------------------
    seeds_team1, seeds_team2 = _process_seeds(data['Seeds'], split=True)
    tourney_data = pd.merge(tourney_data, seeds_team1,
                            on=['Season', 'Team1ID'], how='left')
    tourney_data = pd.merge(tourney_data, seeds_team2,
                            on=['Season', 'Team2ID'], how='left')

    if symmetric_cols:
        box_score_cols = _get_box_score_cols(detailed=detailed)
        tourney_data = tourney_data.drop(
            columns=[c for c in box_score_cols if c in tourney_data.columns],
            errors='ignore'
        )
        tourney_data = pd.merge(tourney_data, season_stats_team1,
                                on=['Season', 'Team1ID'], how='left')
        tourney_data = pd.merge(tourney_data, season_stats_team2,
                                on=['Season', 'Team2ID'], how='left')

    # ----------------------------------------------------------------------
    # 6. Difference features
    # ----------------------------------------------------------------------
    tourney_data = _compute_all_diff_features(tourney_data)

    # ----------------------------------------------------------------------
    # 7. ALWAYS build a GameID so downstream code (incl. generate_matchup_matrix)
    #    can rely on it, even for synthetic 2024 match-ups.
    # ----------------------------------------------------------------------
    if 'GameID' not in tourney_data.columns:
        tourney_data['GameID'] = _generate_game_ids(tourney_data)

    return tourney_data


# ------------------------------------------------------------------
# INTERNAL – load every raw CSV / zip and (NEW) splice-in 2024 seeds
# ------------------------------------------------------------------
def _load_competition_data(data_dir=None, combine=True, detailed=False):
    """
    Loads competition data from CSV files (or fallback zip) **and**
    appends the special `2024_tourney_seeds.csv` so that Season 2024
    seeds are available even though no TourneyResults exist yet.

    Parameters
    ----------
    data_dir : str | None
        Root directory that holds the Kaggle CSV files *and* (optionally)
        the extra 2024 seeds file.  If ``None`` we fall back to the usual
        project / Kaggle-input resolution logic.
    combine : bool
        If True, return a single dict containing joint men's + women's
        frames.  Otherwise return ``{'M': {...}, 'W': {...}}``.
    detailed : bool
        Whether to load the *DetailedResults* flavour of game files.

    Returns
    -------
    dict
        Same structure as before, except that the **Seeds** frame(s)
        now contain rows for Season 2024 when the extra file is present.
    """
    data_dir = _resolve_data_dir(data_dir)

    # ------------ choose which CSV names we need -------------------
    if not detailed:
        files = {
            'MTeams': 'MTeams.csv',
            'WTeams': 'WTeams.csv',
            'MSeasons': 'MSeasons.csv',
            'WSeasons': 'WSeasons.csv',
            'MReg': 'MRegularSeasonCompactResults.csv',
            'WReg': 'WRegularSeasonCompactResults.csv',
            'MTourney': 'MNCAATourneyCompactResults.csv',
            'WTourney': 'WNCAATourneyCompactResults.csv',
            'MSeeds': 'MNCAATourneySeeds.csv',
            'WSeeds': 'WNCAATourneySeeds.csv',
        }
    else:                     # detailed = True
        files = {
            'MTeams': 'MTeams.csv',
            'WTeams': 'WTeams.csv',
            'MSeasons': 'MSeasons.csv',
            'WSeasons': 'WSeasons.csv',
            'MReg': 'MRegularSeasonDetailedResults.csv',
            'WReg': 'WRegularSeasonDetailedResults.csv',
            'MTourney': 'MNCAATourneyDetailedResults.csv',
            'WTourney': 'WNCAATourneyDetailedResults.csv',
            'MSeeds': 'MNCAATourneySeeds.csv',
            'WSeeds': 'WNCAATourneySeeds.csv',
        }

    # ------------ load every file (directory first, else zip) ------
    if (
        data_dir is not None
        and all(os.path.exists(os.path.join(data_dir, f)) for f in files.values())
    ):
        data = {k: _load_csv(os.path.join(data_dir, v)) for k, v in files.items()}
    else:
        data = _load_from_zip(files)

    # ------------ splice-in the **extra 2024 seed file** -----------
    #
    # This file is supplied by Kaggle during Stage-1 (it actually
    # contains 2023 seeds during the open phase, but will be swapped
    # to the real 2024 seeds before final scoring).  Either way,
    # having it in our master Seeds frame means the feature pipeline
    # can happily build 2024 match-ups.
    #
    extra_seed_fname = '2024_tourney_seeds.csv'
    extra_seed_paths = [
        os.path.join(data_dir, extra_seed_fname) if data_dir else None,
        os.path.join(Path.cwd(), extra_seed_fname),
    ]
    extra_seeds = None
    for pth in extra_seed_paths:
        if pth and os.path.exists(pth):
            extra_seeds = _load_csv(pth)
            break

    if extra_seeds is not None and not extra_seeds.empty:
        # Ensure schema lines up with historical seed files
        if 'Season' not in extra_seeds.columns:
            extra_seeds['Season'] = 2024
        # Men’s rows have TeamID 1xxx; women’s 3xxx – append blindly
        if combine:
            # append, then remove any duplicate Season‑TeamID rows
            data['MSeeds'] = (
                pd.concat(
                    [data['MSeeds'], extra_seeds[extra_seeds.TeamID < 3000]],
                    ignore_index=True,
                )
                .drop_duplicates(subset=['Season', 'TeamID'], keep='last')
            )
            data['WSeeds'] = (
                pd.concat(
                    [data['WSeeds'], extra_seeds[extra_seeds.TeamID >= 3000]],
                    ignore_index=True,
                )
                .drop_duplicates(subset=['Season', 'TeamID'], keep='last')
            )
        else:  # gender‑specific load
            mask_m = extra_seeds.TeamID < 3000      # Men  → TeamID 1xxx
            mask_w = ~mask_m                        # Women → TeamID 3xxx+

            data['MSeeds'] = (
                pd.concat([data['MSeeds'], extra_seeds[mask_m]], ignore_index=True)
                .drop_duplicates(subset=['Season', 'TeamID'], keep='last')
            )
            data['WSeeds'] = (
                pd.concat([data['WSeeds'], extra_seeds[mask_w]], ignore_index=True)
                .drop_duplicates(subset=['Season', 'TeamID'], keep='last')
            )

    # ------------ collapse men + women if caller asked for combine --
    if combine:
        teams   = pd.concat([data['MTeams'],   data['WTeams']],   ignore_index=True)
        seasons = pd.concat([data['MSeasons'], data['WSeasons']], ignore_index=True)
        reg     = pd.concat([data['MReg'],     data['WReg']],     ignore_index=True)
        tourney = pd.concat([data['MTourney'], data['WTourney']], ignore_index=True)
        seeds   = pd.concat([data['MSeeds'],   data['WSeeds']],   ignore_index=True)
        return {
            'Teams':           teams,
            'Seasons':         seasons,
            'RegResults':      reg,
            'TourneyResults':  tourney,
            'Seeds':           seeds,
        }
    else:
        # keep the gender-split dict exactly as before
        return {
            'M': {
                'Teams':          data['MTeams'],
                'Seasons':        data['MSeasons'],
                'RegResults':     data['MReg'],
                'TourneyResults': data['MTourney'],
                'Seeds':          data['MSeeds'],
            },
            'W': {
                'Teams':          data['WTeams'],
                'Seasons':        data['WSeasons'],
                'RegResults':     data['WReg'],
                'TourneyResults': data['WTourney'],
                'Seeds':          data['WSeeds'],
            },
        }



def _merge_df_with_stats(df, team1_stat_df, team2_stat_df):
    """
    Merges the input DataFrame with team statistics DataFrames for Team1 and Team2.
    
    Parameters:
        df (DataFrame): The main DataFrame to merge.
        team1_stat_df (DataFrame): DataFrame with Team1 statistics.
        team2_stat_df (DataFrame): DataFrame with Team2 statistics.
    
    Returns:
        DataFrame: The merged DataFrame with team statistics included.
    """
    # Merge on Season and Team1ID for Team1 stats
    df = pd.merge(df, team1_stat_df, on=['Season', 'Team1ID'], how='left')
    # Merge on Season and Team2ID for Team2 stats
    df = pd.merge(df, team2_stat_df, on=['Season', 'Team2ID'], how='left')
    return df


def _standardize_results(df, detailed=False, reorder=False):
    """
    Standardizes column names and computes Winner/Loser information for game results.
    
    Parameters:
        df (DataFrame): Game results DataFrame.
        detailed (bool): If True, indicates detailed stats are present.
        reorder (bool): If True, reorders team-specific columns so that the lower team ID is Team1.
    
    Returns:
        None. The DataFrame is modified in place.
    """
    # Define a mapping from original column names to standardized names
    rename_dict = {
        'WTeamID': 'Team1ID',
        'WScore': 'Team1Score',
        'LTeamID': 'Team2ID',
        'LScore': 'Team2Score',
        'WFGM': 'Team1FGM',
        'WFGA': 'Team1FGA',
        'WFGM3': 'Team1FGM3',
        'WFGA3': 'Team1FGA3',
        'WFTM': 'Team1FTM',
        'WFTA': 'Team1FTA',
        'WOR': 'Team1OR',
        'WDR': 'Team1DR',
        'WAst': 'Team1Ast',
        'WTO': 'Team1TO',
        'WStl': 'Team1Stl',
        'WBlk': 'Team1Blk',
        'WPF': 'Team1PF',
        'LFGM': 'Team2FGM',
        'LFGA': 'Team2FGA',
        'LFGM3': 'Team2FGM3',
        'LFGA3': 'Team2FGA3',
        'LFTM': 'Team2FTM',
        'LFTA': 'Team2FTA',
        'LOR': 'Team2OR',
        'LDR': 'Team2DR',
        'LAst': 'Team2Ast',
        'LTO': 'Team2TO',
        'LStl': 'Team2Stl',
        'LBlk': 'Team2Blk',
        'LPF': 'Team2PF',
        'WLoc': 'Location'
    }
    # Rename only the columns present in the DataFrame
    available_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    df.rename(columns=available_rename, inplace=True)
    
    # Compute Winner: if Team1Score > Team2Score then Team1ID, else Team2ID
    df['Winner'] = np.where(df['Team1Score'] > df['Team2Score'],
                             df['Team1ID'], df['Team2ID'])
    # Compute Loser similarly
    df['Loser'] = np.where(df['Team1Score'] > df['Team2Score'],
                            df['Team2ID'], df['Team1ID'])
    
    # If reordering is requested, swap columns where necessary
    if reorder:
        _apply_full_reordering_adjustments(df, detailed)


def _apply_full_reordering_adjustments(df, detailed):
    """
    For rows where Team1ID > Team2ID, swaps team-specific columns so that the lower ID is Team1.
    
    Parameters:
        df (DataFrame): Game results DataFrame.
        detailed (bool): If True, also swaps additional detailed statistics.
    
    Returns:
        None. The DataFrame is modified in place.
    """
    # Create a boolean mask for rows needing a swap
    swap_mask = df['Team1ID'] > df['Team2ID']
    # Swap the team IDs for the identified rows
    df.loc[swap_mask, ['Team1ID', 'Team2ID']] = df.loc[swap_mask, ['Team2ID', 'Team1ID']].values
    # Swap the corresponding scores
    df.loc[swap_mask, ['Team1Score', 'Team2Score']] = df.loc[swap_mask, ['Team2Score', 'Team1Score']].values
    
    # For detailed data, swap additional statistics
    if detailed:
        stat_suffixes = [
            'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR',
            'Ast', 'TO', 'Stl', 'Blk', 'PF'
        ]
        for suffix in stat_suffixes:
            col1 = 'Team1' + suffix
            col2 = 'Team2' + suffix
            if col1 in df.columns and col2 in df.columns:
                df.loc[swap_mask, [col1, col2]] = df.loc[swap_mask, [col2, col1]].values
    
    # Recompute Winner and Loser after swapping
    df['Winner'] = np.where(df['Team1Score'] > df['Team2Score'], df['Team1ID'], df['Team2ID'])
    df['Loser'] = np.where(df['Team1Score'] > df['Team2Score'], df['Team2ID'], df['Team1ID'])


def _build_team_stats(reg_results):
    """
    Builds aggregated team statistics from regular season game results.
    
    Parameters:
        reg_results (DataFrame): Regular season game results.
    
    Returns:
        DataFrame: Aggregated statistics including games played, wins, losses, win percentage,
                   total points for/against, and average margin.
    """
    games_list = []
    # Iterate over each game to create two records (one per team)
    for _, row in reg_results.iterrows():
        season = row['Season']
        winner = row['Team1ID']
        loser = row['Team2ID']
        win_score = row['Team1Score']
        lose_score = row['Team2Score']
        # Append record for the winning team
        games_list.append({
            'Season': season, 'TeamID': winner,
            'PointsFor': win_score, 'PointsAgainst': lose_score, 'Win': 1
        })
        # Append record for the losing team
        games_list.append({
            'Season': season, 'TeamID': loser,
            'PointsFor': lose_score, 'PointsAgainst': win_score, 'Win': 0
        })
    reg_teamgames = pd.DataFrame(games_list)
    # Aggregate the records by Season and TeamID
    team_stats = reg_teamgames.groupby(['Season', 'TeamID']).agg(
        Games=('Win', 'count'),
        NumWins=('Win', 'sum'),
        PointsFor=('PointsFor', 'sum'),
        PointsAgainst=('PointsAgainst', 'sum')
    ).reset_index()
    # Compute additional statistics
    team_stats['Losses'] = team_stats['Games'] - team_stats['NumWins']
    team_stats['WinPct'] = team_stats['NumWins'] / team_stats['Games']
    team_stats['AvgMargin'] = (team_stats['PointsFor'] - team_stats['PointsAgainst']) / team_stats['Games']
    return team_stats


def _prepare_data(df, detailed=False, symmetry=True, reg=False):
    """
    Prepares game results data for modeling by selecting key columns, recoding variables,
    and optionally augmenting the data with symmetric (swapped) copies.
    
    Parameters:
        df (DataFrame): Raw game results.
        detailed (bool): Indicates if detailed statistics are present.
        symmetry (bool): If True, includes swapped copies of each game.
        reg (bool): If True, also returns aggregated team statistics.
    
    Returns:
        If reg is True: Tuple (processed_df, team_stats).
        Otherwise: Processed DataFrame.
    """
    if reg:
        team_stats = _build_team_stats(df)
    
    # Define expected columns; add detailed columns if available
    expected_cols = (
        ['Season', 'DayNum', 'Team2ID', 'Team2Score', 'Team1ID', 'Team1Score', 'Location', 'NumOT'] +
        (['Team2FGM', 'Team2FGA', 'Team2FGM3', 'Team2FGA3', 'Team2FTM', 'Team2FTA',
          'Team2OR', 'Team2DR', 'Team2Ast', 'Team2TO', 'Team2Stl', 'Team2Blk', 'Team2PF',
          'Team1FGM', 'Team1FGA', 'Team1FGM3', 'Team1FGA3', 'Team1FTM', 'Team1FTA',
          'Team1OR', 'Team1DR', 'Team1Ast', 'Team1TO', 'Team1Stl', 'Team1Blk', 'Team1PF']
         if detailed else []
        )
    )
    df_fixed = df[expected_cols].copy()
    
    if symmetry:
        # Create a swapped copy of the data
        dfswap = df_fixed.copy()
        # Swap the 'Location' indicator (home <-> away)
        dfswap['Location'] = dfswap['Location'].map({'H': 'A', 'A': 'H', 'N': 'N'})
        # For each statistic, swap the values between Team1 and Team2
        stat_cols = ['ID', 'Score'] + (['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']
                                         if detailed else [])
        for stat in stat_cols:
            col_team1 = 'Team1' + stat
            col_team2 = 'Team2' + stat
            dfswap[col_team1], dfswap[col_team2] = dfswap[col_team2].copy(), dfswap[col_team1].copy()
        # Combine original and swapped data
        output = pd.concat([df_fixed, dfswap], ignore_index=True)
    else:
        output = df_fixed.copy()
    
    # Recode 'Location' to numeric: Home=1, Away=-1, Neutral=0
    output['Location'] = output['Location'].map({'H': 1, 'A': -1, 'N': 0}).astype(int)
    # Compute point differential
    output['PointDiff'] = output['Team1Score'] - output['Team2Score']
    # Determine if Team1 wins (binary indicator)
    output['Team1Wins'] = (output['Team1Score'] > output['Team2Score']).astype(int)
    
    if reg:
        return output, team_stats
    return output


def _compute_season_statistics(data, symmetric_cols, detailed):
    """
    Computes aggregated season statistics from regular season data.
    
    Parameters:
        data (DataFrame): Processed regular season results.
        symmetric_cols (bool): If True, renames columns to include team-specific prefixes.
        detailed (bool): If True, expects detailed box score columns.
    
    Returns:
        Tuple of DataFrames: (season_stats_team1, season_stats_team2)
    """
    regular_data = data.copy()
    # Get the list of box score columns based on the detailed flag
    box_score_cols = _get_box_score_cols(detailed=detailed)
    # Group by Season and Team1ID and compute the mean for each box score column
    season_stats = regular_data.groupby(['Season', 'Team1ID'])[box_score_cols].agg('mean').reset_index()
    # Flatten column names
    season_stats.columns = [''.join(col).strip() for col in season_stats.columns.values]
    
    if symmetric_cols:
        # Create copies for Team1 and Team2 perspectives with renamed columns
        season_stats_team1 = season_stats.copy()
        season_stats_team2 = season_stats.copy()
        season_stats_team1.columns = [
            'Team1' + col.replace('Team1', '').replace('Team2', '_opponent_')
            for col in list(season_stats_team1.columns)
        ]
        season_stats_team2.columns = [
            'Team2' + col.replace('Team1', '').replace('Team2', '_opponent_')
            for col in list(season_stats_team2.columns)
        ]
        # Ensure that the 'Season' column remains unchanged
        season_stats_team1.columns.values[0] = 'Season'
        season_stats_team2.columns.values[0] = 'Season'
    else:
        season_stats_team1 = season_stats.copy()
        season_stats_team2 = season_stats.copy()
    
    return season_stats_team1, season_stats_team2


def _compute_regular_season_effects(data, seeds_df):
    """
    Computes regular season effects based on point differential and merges with seeds data.
    
    Parameters:
        data (DataFrame): Processed regular season game results.
        seeds_df (DataFrame): Seeds data.
    
    Returns:
        DataFrame: A DataFrame with columns 'Season', 'Team1ID', 'Team2ID', 'PointDiff', and 'win' indicator,
                   merged with all possible matchups based on seeds.
    """
    reg_effects = data[['Season', 'Team1ID', 'Team2ID', 'PointDiff']].copy()
    # Convert team IDs to string for merging purposes
    reg_effects['Team1ID'] = reg_effects['Team1ID'].astype(str)
    reg_effects['Team2ID'] = reg_effects['Team2ID'].astype(str)
    # Create a win indicator (1 if Team1 wins, 0 otherwise)
    reg_effects['win'] = np.where(reg_effects['PointDiff'] > 0, 1, 0)
    
    # Merge with a DataFrame that contains all possible matchups based on seeds
    mm = pd.merge(seeds_df[['Season', 'TeamID']], seeds_df[['Season', 'TeamID']], on='Season')
    mm.columns = ['Season', 'Team1ID', 'Team2ID']
    mm.Team1ID = mm.Team1ID.astype(str)
    mm.Team2ID = mm.Team2ID.astype(str)
    
    reg_effects = pd.merge(reg_effects, mm, on=['Season', 'Team1ID', 'Team2ID'])
    return reg_effects


def _compute_last14day_win_ratios(data):
    """
    Computes the win ratio for the last 14 days of the regular season for each team.
    
    Parameters:
        data (DataFrame): Processed regular season game results (must include 'DayNum').
    
    Returns:
        Tuple of DataFrames:
            - last14_team1_ratio: DataFrame with ['Season', 'Team1ID', 'Team1Last14WinRate'].
            - last14_team2_ratio: DataFrame with ['Season', 'Team2ID', 'Team2Last14WinRate'].
    """
    # Get all unique seasons and team IDs to ensure complete coverage
    all_seasons = data['Season'].unique()
    all_team1_ids = data['Team1ID'].unique()
    all_team2_ids = data['Team2ID'].unique()
    all_teams = np.union1d(all_team1_ids, all_team2_ids)
    
    # Create complete cross product of seasons and teams for full coverage
    seasons_teams1 = pd.DataFrame([(s, t) for s in all_seasons for t in all_teams], 
                                 columns=['Season', 'Team1ID'])
    seasons_teams2 = pd.DataFrame([(s, t) for s in all_seasons for t in all_teams], 
                                 columns=['Season', 'Team2ID'])
    
    # Select games where DayNum > 118 (last 14 days)
    last14_team1 = data.loc[data.DayNum > 118].reset_index(drop=True)
    # Create win indicator for Team1 in the last 14 days
    last14_team1['win'] = np.where(last14_team1['PointDiff'] > 0, 1, 0)
    last14_team1_ratio = last14_team1.groupby(['Season', 'Team1ID'])['win'].mean().reset_index(
        name='Team1Last14WinRate'
    )
    
    last14_team2 = data.loc[data.DayNum > 118].reset_index(drop=True)
    # Create win indicator for Team2 (reverse condition)
    last14_team2['win'] = np.where(last14_team2['PointDiff'] < 0, 1, 0)
    last14_team2_ratio = last14_team2.groupby(['Season', 'Team2ID'])['win'].mean().reset_index(
        name='Team2Last14WinRate'
    )
    
    # Merge with the complete seasons-teams cross product to ensure no missing values
    last14_team1_ratio = pd.merge(seasons_teams1, last14_team1_ratio, 
                                 on=['Season', 'Team1ID'], how='left')
    last14_team2_ratio = pd.merge(seasons_teams2, last14_team2_ratio, 
                                 on=['Season', 'Team2ID'], how='left')
    
    # Fill missing values with 0.5 (neutral win rate) - FIXED to avoid FutureWarning
    last14_team1_ratio = last14_team1_ratio.fillna({'Team1Last14WinRate': 0.5})
    last14_team2_ratio = last14_team2_ratio.fillna({'Team2Last14WinRate': 0.5})
    
    return last14_team1_ratio, last14_team2_ratio


def _team_quality(season, reg_effects):
    """
    Estimates team quality for a given season using a GLM based on point differential.
    
    Parameters:
        season (int): The season for which to compute team quality.
        reg_effects (DataFrame): Regular season effects for all games.
    
    Returns:
        DataFrame: A DataFrame with columns ['TeamID', 'quality', 'Season'] containing quality estimates.
                 Only coefficients corresponding to Team1ID are used.
    """
    formula = 'I(PointDiff) ~ 1 + C(Team1ID) + C(Team2ID)'
    glm = sm.GLM.from_formula(
        formula=formula,
        data=reg_effects.loc[reg_effects.Season == season, :],
        family=sm.families.Gaussian()
    ).fit()
    # Create a DataFrame from GLM coefficients
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID', 'quality']
    quality['Season'] = season
    # Filter for coefficients related to Team1ID and extract the numeric team ID
    quality = quality.loc[quality.TeamID.str.contains('Team1ID')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[13:-1]).astype(int)
    return quality


def _compile_team_quality(reg_effects):
    """
    Computes team quality estimates for each season and concatenates them.
    
    Parameters:
        reg_effects (DataFrame): Regular season effects for all seasons.
    
    Returns:
        DataFrame: A concatenated DataFrame of team quality estimates with columns ['TeamID', 'quality', 'Season'].
    """
    seasons = reg_effects['Season'].unique()
    quality_list = [_team_quality(season, reg_effects) for season in seasons]
    glm_quality = pd.concat(quality_list).reset_index(drop=True)
    return glm_quality


def _compute_neutral_win_rate(data):
    """
    Computes the neutral win rate for both teams from games played at a neutral location.
    
    Parameters:
        data (DataFrame): Processed game results containing 'Location' and 'Team1Wins'.
    
    Returns:
        Tuple of DataFrames:
            - team1_neutral: DataFrame with ['Season', 'Team1ID', 'Team1NeutralWinRate'].
            - team2_neutral: DataFrame with ['Season', 'Team2ID', 'Team2NeutralWinRate'].
    """
    # Get all unique seasons and team IDs to ensure complete coverage
    all_seasons = data['Season'].unique()
    all_team1_ids = data['Team1ID'].unique()
    all_team2_ids = data['Team2ID'].unique()
    all_teams = np.union1d(all_team1_ids, all_team2_ids)
    
    # Create complete cross product of seasons and teams for full coverage
    seasons_teams1 = pd.DataFrame([(s, t) for s in all_seasons for t in all_teams], 
                                 columns=['Season', 'Team1ID'])
    seasons_teams2 = pd.DataFrame([(s, t) for s in all_seasons for t in all_teams], 
                                 columns=['Season', 'Team2ID'])
    
    # Check if 'Location' column exists
    if 'Location' not in data.columns:
        # Create empty DataFrames with the required columns and fill with default 0.5
        team1_neutral = seasons_teams1.copy()
        team1_neutral['Team1NeutralWinRate'] = 0.5
        team2_neutral = seasons_teams2.copy()
        team2_neutral['Team2NeutralWinRate'] = 0.5
        return team1_neutral, team2_neutral
    
    # Determine the value that represents a neutral location
    neutral_value = 0 if pd.api.types.is_numeric_dtype(data['Location']) else 'N'
    neutral_games = data[data['Location'] == neutral_value].reset_index(drop=True)
    
    if neutral_games.empty:
        # Create empty DataFrames with the required columns and fill with default 0.5
        team1_neutral = seasons_teams1.copy()
        team1_neutral['Team1NeutralWinRate'] = 0.5
        team2_neutral = seasons_teams2.copy()
        team2_neutral['Team2NeutralWinRate'] = 0.5
        return team1_neutral, team2_neutral
    
    # Calculate win rates for teams with neutral game data
    team1_neutral_rates = (
        neutral_games.groupby(['Season', 'Team1ID'])['Team1Wins']
        .mean().reset_index(name='Team1NeutralWinRate')
    )
    team2_neutral_rates = (
        neutral_games.groupby(['Season', 'Team2ID'])['Team1Wins']
        .apply(lambda x: 1 - x.mean()).reset_index(name='Team2NeutralWinRate')
    )
    
    # Merge with complete season-team cross product
    team1_neutral = pd.merge(seasons_teams1, team1_neutral_rates, 
                            on=['Season', 'Team1ID'], how='left')
    team2_neutral = pd.merge(seasons_teams2, team2_neutral_rates, 
                            on=['Season', 'Team2ID'], how='left')
    
    # Fill missing values with 0.5 (neutral win rate) - FIXED to avoid FutureWarning
    team1_neutral = team1_neutral.fillna({'Team1NeutralWinRate': 0.5})
    team2_neutral = team2_neutral.fillna({'Team2NeutralWinRate': 0.5})
    
    return team1_neutral, team2_neutral


def _calculate_elo(data, initial_elo=1500, k=20):
    """
    Compute Elo ratings for each season from **all** regular-season games.

    The previous implementation filtered to rows where Team1Wins == 1,
    which (a) threw away half the information and (b) gave every game a
    positive point-differential bias.  We now:

    • use every game once (winner + loser both updated);  
    • when symmetry-augmentation is on, keep only one copy of each match
      so the update is not applied twice.

    Parameters
    ----------
    data : pd.DataFrame
        Output of _prepare_data for the regular season (may contain
        swapped rows if symmetry=True).
    initial_elo : int, default 1500
        Starting Elo rating for unseen teams.
    k : int, default 20
        K-factor for the update step.

    Returns
    -------
    (elo_team1_rating, elo_team2_rating) : tuple of DataFrames
        Each has columns ['Season', 'Team1ID/Team2ID', 'Team1Elo/Team2Elo'].
    """
    # ------------------------------------------------------------
    # 1)  Drop the swapped copy when symmetry augmentation is on
    #     -> keep the variant where Team1ID < Team2ID
    # ------------------------------------------------------------
    data_dedup = data.copy()
    mask_swap = data_dedup["Team1ID"] > data_dedup["Team2ID"]
    data_dedup = data_dedup.loc[~mask_swap]            # one row per real game

    elo_records = []                                    # collect rows for output
    for season in sorted(data_dedup["Season"].unique()):
        season_df = data_dedup[data_dedup["Season"] == season].copy()
        # keep chronological order if we have DayNum information
        if "DayNum" in season_df.columns:
            season_df.sort_values("DayNum", inplace=True)

        teams = pd.unique(season_df[["Team1ID", "Team2ID"]].values.ravel())
        elo = {team: initial_elo for team in teams}

        # --------------------------------------------------------
        # 2)  Per-game Elo update
        # --------------------------------------------------------
        for _, g in season_df.iterrows():
            t1, t2 = g["Team1ID"], g["Team2ID"]
            # outcome from Team1 perspective
            s1 = 1 if g["Team1Wins"] == 1 else 0
            s2 = 1 - s1

            r1, r2 = elo[t1], elo[t2]
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 - e1

            elo[t1] = r1 + k * (s1 - e1)
            elo[t2] = r2 + k * (s2 - e2)

        # snapshot season-end ratings
        for team, rating in elo.items():
            elo_records.append({"Season": season, "TeamID": team, "Elo": rating})

    elo_df = pd.DataFrame(elo_records)
    elo_team1_rating = elo_df.rename(columns={"TeamID": "Team1ID", "Elo": "Team1Elo"})
    elo_team2_rating = elo_df.rename(columns={"TeamID": "Team2ID", "Elo": "Team2Elo"})
    return elo_team1_rating, elo_team2_rating



def _compute_all_diff_features(df):
    """
    Computes difference features for numeric columns that have both Team1 and Team2 versions.
    
    Parameters:
        df (DataFrame): DataFrame containing team-specific features.
    
    Returns:
        DataFrame: The DataFrame with additional columns for differences (e.g., 'SeedDiff', etc.).
    """
    df = df.copy()
    # Iterate over each column to check if it should have a diff feature created
    for col in df.columns:
        if (col.startswith('Team1') and '_opponent' not in col and 'Diff' not in col and 'ID' not in col):
            suffix = col[len('Team1'):]
            team2_col = 'Team2' + suffix
            # Only create the diff column if the corresponding Team2 column exists and is numeric
            if team2_col in df.columns:
                if (pd.api.types.is_numeric_dtype(df[col]) and
                        pd.api.types.is_numeric_dtype(df[team2_col])):
                    diff_col = suffix + 'Diff'
                    df[diff_col] = df[col] - df[team2_col]
    return df


def _get_box_score_cols(detailed=False):
    """
    Returns a list of column names representing box score statistics.
    
    Parameters:
        detailed (bool): If True, returns detailed box score columns.
    
    Returns:
        list of str: Column names for box score statistics.
    """
    if detailed:
        return [
            'Team1FGM', 'Team1FGA', 'Team1FGM3', 'Team1FGA3', 'Team1OR',
            'Team1Ast', 'Team1TO', 'Team1Stl', 'Team1PF',
            'Team2FGM', 'Team2FGA', 'Team2FGM3', 'Team2FGA3', 'Team2OR',
            'Team2Ast', 'Team2TO', 'Team2Stl', 'Team2PF',
            'PointDiff'
        ]
    else:
        return ['Team1Score', 'Team2Score', 'PointDiff']


def _generate_game_ids(df: pd.DataFrame) -> np.ndarray:
    """
    Unique, order-agnostic ID:  SSSS · 1e8  +  minID · 1e4  +  maxID
    Guarantees uniqueness for every unordered pair within a season.
    """
    id_low  = df[['Team1ID', 'Team2ID']].min(axis=1)
    id_high = df[['Team1ID', 'Team2ID']].max(axis=1)
    return (df['Season']*100_000_000 + id_low*10_000 + id_high).values

