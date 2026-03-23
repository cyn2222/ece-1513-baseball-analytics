"""
Shared data pipeline for MLB win prediction.

Loads the Lahman Teams table, selects features, lags by one season, splits into train/test sets, and applies standard scaling.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# feature specs and some explanations if you don't know ball
# 15 features grouped: Batting (7), Pitching (6), Fielding (2)
# Dropped 2B, 3B (subsets of H, highly correlated with it)
FEATURE_COLS = [
    # --- Batting (7) ---
    'R',   # Runs scored, most powerful single predictor, core of the Pythagorean baseline
    'H',   # Hits by batters, main way batters reach base
    'HR',  # Home runs, big run creator
    'BB',  # Base on balls (walks), another way batters reach base
    'SO',  # Strikeouts by batters, captures offensive weakness
    'SB',  # Stolen bases, baserunning/speed dimension
    'HBP', # Hit by pitch, small but nonzero on-base contributor uncorrelated with the others
    # --- Pitching (6) ---
    'RA',  # Runs allowed (opponents runs scored), pitching counterpart to R
    'ERA', # Earned run average, rate-stat perspective on pitching
    'HA',  # Hits allowed, component-level pitching performance
    'HRA', # Home runs allowed, component-level pitching performance
    'BBA', # Base on balls allowed, component-level pitching performance
    'SOA', # Strikeouts by pitchers, component-level pitching performance
    # --- Fielding (2) ---
    'E',   # Errors/misplays by fielders, lower the better
    'FP',  # Fielding percentage, higher the better
]

# columns kept alongside features for identification / downstream use
META_COLS = [
    'yearID', 
    'teamID', 
    'franchID', # this is the franchise ID, more consistent than teamID since teams might rename
    'G', # Games played in the season (not always 162!)
    'W' # Wins, obviously
]

# seasons to keep, excluding 2020 COVID-shortened season for obvious reasons
YEAR_MIN = 2000
YEAR_MAX = 2025
EXCLUDED_YEARS = {2020}

# split boundaries
TRAIN_YEARS = (2001, 2019)
TEST_YEARS  = (2021, 2025)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# some housekeeping for the data file
def _resolve_csv_path(csv_path=None):
    if csv_path is not None:
        return csv_path
    candidate = os.path.join(_REPO_ROOT, 'data', 'Teams.csv')
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(os.getcwd(), 'data', 'Teams.csv')
    if os.path.isfile(candidate):
        return candidate
    raise FileNotFoundError('Teams.csv not found!')


# load the Lahman Teams table filtered to the target year range.
def load_raw_teams(csv_path=None):
    path = _resolve_csv_path(csv_path)
    teams = pd.read_csv(path, encoding='utf-8-sig')
    teams = teams[teams['yearID'].between(YEAR_MIN, YEAR_MAX)]
    teams = teams[~teams['yearID'].isin(EXCLUDED_YEARS)]

    # validate columns, just in case
    required = META_COLS + FEATURE_COLS
    missing = [c for c in required if c not in teams.columns]
    if missing:
        raise KeyError("Teams.csv is missing expected these columns: ", str(missing) + '\nDid you download the right file?')

    teams = teams[required].copy()
    for col in FEATURE_COLS:
        teams[col] = pd.to_numeric(teams[col], errors='coerce')

    # drop rows where any required feature is missing (rare but possible)
    teams = teams.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return teams


"""Create a dataframe where each row's features come from the *previous*
season for the same franchise, while the target W stays at the
current season. This avoids data leakage."""
def build_lagged_df(teams):
    teams_sorted = teams.sort_values(['franchID', 'yearID']).copy()
    lagged = teams_sorted.groupby('franchID')[FEATURE_COLS].shift(1)
    lagged.columns = [f'{c}_prev' for c in FEATURE_COLS]
    df = pd.concat([teams_sorted[META_COLS].reset_index(drop=True), lagged.reset_index(drop=True)], axis=1)
    df = df.dropna().reset_index(drop=True)
    prev_cols = [c for c in df.columns if c.endswith('_prev')]
    df[prev_cols] = df[prev_cols].astype(float)
    return df


# split train/test
def split_train_test(df):
    train = df[df['yearID'].between(*TRAIN_YEARS)].reset_index(drop=True)
    test  = df[df['yearID'].between(*TEST_YEARS)].reset_index(drop=True)
    return train, test

# pull numpy arrays from the train / test DataFrames
def extract_arrays(train, test):
    prev_cols = [c for c in train.columns if c.endswith('_prev')]
    X_train = train[prev_cols].values
    y_train = train['W'].values.astype(float)
    X_test  = test[prev_cols].values
    y_test  = test['W'].values.astype(float)
    return X_train, y_train, X_test, y_test


# feature standardisation
def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# convenience wrapper
def prepare_data(csv_path=None):
    teams = load_raw_teams(csv_path)
    df    = build_lagged_df(teams)
    train, test = split_train_test(df)
    X_train, y_train, X_test, y_test = extract_arrays(train, test)
    X_train_scaled, X_test_scaled, scaler = standardize(X_train, X_test)
    feature_names = [c for c in train.columns if c.endswith('_prev')]
    return {
        'X_train_scaled': X_train_scaled,
        'y_train':        y_train,
        'X_test_scaled':  X_test_scaled,
        'y_test':         y_test,
        'scaler':         scaler,
        'train':          train,
        'test':           test,
        'feature_names':  feature_names,
    }


if __name__ == '__main__':
    data = prepare_data()

    print("=== Data Pipeline Summary ===")
    print("  Features (", len(data['feature_names']), "): ", data['feature_names'])
    print("  Train set : ", data['X_train_scaled'].shape[0], " rows (seasons ", TRAIN_YEARS[0], "–", TRAIN_YEARS[1], ")")
    print("  Test  set : ", data['X_test_scaled'].shape[0], " rows (seasons ", TEST_YEARS[0], "–", TEST_YEARS[1], ")")
    print("  y_train range: [", f"{data['y_train'].min():.0f}", ", ", f"{data['y_train'].max():.0f}", "]")
    print("  y_test  range: [", f"{data['y_test'].min():.0f}", ", ", f"{data['y_test'].max():.0f}", "]")