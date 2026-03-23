"""
Support Vector Regression (SVR) model for MLB win prediction.

Based on Lectures 8–10 (SVC/SVM, kernel methods).

TODO:
    - Implement fit_svr() with grid search over kernel, C, epsilon
    - Compare linear vs RBF kernels
    - Discuss why nonlinear kernels might capture feature interactions
"""
from data_pipeline import prepare_data, _REPO_ROOT, FEATURE_COLS
from evaluate import compute_mae, compute_rmse, plot_pred_vs_actual, print_metrics