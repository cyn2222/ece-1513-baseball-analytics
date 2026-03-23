"""
Shared evaluation utilities so that we have consistent metrics and visualizations across all models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


# compute mean absolute error
def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

# compute root mean squared error
def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# draw a predicted vs. actual scatter with a 45° reference line
def plot_pred_vs_actual(y_true, y_pred, model_name='Model', save_path=None, ax=None, show=False):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', linewidths=0.5)

    # reference line
    lo = min(y_true.min(), y_pred.min()) - 5
    hi = max(y_true.max(), y_pred.max()) + 5
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1, label='Perfect prediction')

    ax.set_xlabel('Actual Wins')
    ax.set_ylabel('Predicted Wins')
    ax.set_title(model_name + ': Predicted vs. Actual')
    ax.legend(loc='upper left')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')

    # annotate with our metrics
    mae  = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    ax.text(0.05, 0.92, 'MAE  = ' + str(mae) + '\nRMSE = ' + str(rmse), transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return ax

# print metrics out for a model
def print_metrics(y_true, y_pred, model_name='Model'):
    mae  = compute_mae(y_true, y_pred)
    rmse = compute_rmse(y_true, y_pred)
    print(model_name + ' metrics:')
    print('\tMAE: ' + str(mae))
    print('\tRMSE: ' + str(rmse))
    return mae, rmse
