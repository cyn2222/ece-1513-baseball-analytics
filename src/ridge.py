"""
Ridge Regression model for MLB win prediction.

Based on Lectures 6–7 (linear regression)
and Lecture 11 (regularization / generalization).

TODO:
    - Implement fit_ridge() with cross-validated alpha selection
    - Add alpha validation curve plotting
    - Report feature importances (coefficients)
"""
from data_pipeline import prepare_data, _REPO_ROOT, FEATURE_COLS
from evaluate import compute_mae, compute_rmse, plot_pred_vs_actual, print_metrics
