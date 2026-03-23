"""
This is our baseline: Pythagorean Win Expectation.

Bill James's formula:  W_hat = games_per_season * R^2 / (R^2 + RA^2)
"""

import os
import numpy as np
from data_pipeline import prepare_data, _REPO_ROOT
from evaluate import compute_mae, compute_rmse, plot_pred_vs_actual

# predict using Bill James's formula
def pythagorean_predict(R, RA, exponent=2, games=162):
    R = np.asarray(R, dtype=float)
    RA = np.asarray(RA, dtype=float)
    games = np.asarray(games, dtype=float)
    return games * (R ** exponent) / (R ** exponent + RA ** exponent)


# baseline evaluation
def run_baseline(csv_path=None):
    data = prepare_data(csv_path)
    test = data['test']
    y_test = data['y_test']

    R_prev  = test['R_prev'].values
    RA_prev = test['RA_prev'].values
    games   = test['G'].values
    predictions = pythagorean_predict(R_prev, RA_prev, games=games)

    mae  = compute_mae(y_test, predictions)
    rmse = compute_rmse(y_test, predictions)

    return {
        'predictions': predictions,
        'y_test': y_test,
        'mae': mae,
        'rmse': rmse,
        'test_df': test,
    }


if __name__ == '__main__':
    results = run_baseline()
    print("=== Pythagorean Win Expectation Baseline ===")
    print("  Test samples : ", len(results['y_test']))
    print("  MAE          : ", str(results['mae']), " wins")
    print("  RMSE         : ", str(results['rmse']), " wins")
    save_path = os.path.join(_REPO_ROOT, 'results', 'pythagorean_pred_vs_actual.png')
    plot_pred_vs_actual(results['y_test'], results['predictions'], model_name='Pythagorean Baseline', save_path=save_path)