"""
Utility functions for plotting in Trunkline ML.
"""
import os
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# Set the style for all plots
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.1)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette
COLOR_PALETTE = sns.color_palette("viridis")
PRIMARY_COLOR = '#2e7d32'  # Green
SECONDARY_COLOR = '#546e7a'  # Blue-gray
ACCENT_COLOR = '#4caf50'  # Light green


def set_plot_style() -> None:
    """Set the style for all plots."""
    sns.set_style("whitegrid", {
        'axes.edgecolor': '#dddddd',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'axes.linewidth': 1.0,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'grid.color': '#eeeeee',
        'grid.linestyle': '--',
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.fancybox': True,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
    })
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['savefig.transparent'] = False


def plot_predicted_vs_true(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    model_name: str = "Model",
    x_label: str = "True Values",
    y_label: str = "Predictions",
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.6,
    color: str = PRIMARY_COLOR,
    line_color: str = 'red',
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot predicted vs true values with a reference line.
    
    Args:
        y_true: Array of true target values
        y_pred: Array of predicted values
        model_name: Name of the model (for title)
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size (width, height)
        alpha: Transparency of points
        color: Color of the points
        line_color: Color of the reference line
        title: Plot title (if None, auto-generate)
        ax: Matplotlib Axes object (if None, create new figure)
        
    Returns:
        Tuple containing the Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Scatter plot of predictions vs true values
    ax.scatter(y_true, y_pred, alpha=alpha, color=color, edgecolor='white', s=80)
    
    # Add reference line (y = x)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            '--', color=line_color, linewidth=2, label='Perfect Prediction')
    
    # Add metrics to the plot
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_text = f"{model_name}\n" \
                 f"MSE = {mse:.4f}\n" \
                 f"R² = {r2:.4f}"
    
    # Position the text in the top-left corner
    ax.text(0.02, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title is None:
        title = f"{model_name}: Predicted vs True Values"
    ax.set_title(title, fontsize=14, pad=15)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_residuals(
    y_true: np.ndarray, 
    y_pred: np.ndarray,
    model_name: str = "Model",
    x_label: str = "Predicted Values",
    y_label: str = "Residuals",
    figsize: Tuple[int, int] = (8, 6),
    color: str = SECONDARY_COLOR,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot residuals vs predicted values.
    
    Args:
        y_true: Array of true target values
        y_pred: Array of predicted values
        model_name: Name of the model (for title)
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size (width, height)
        color: Color of the points
        title: Plot title (if None, auto-generate)
        ax: Matplotlib Axes object (if None, create new figure)
        
    Returns:
        Tuple containing the Figure and Axes objects
    """
    residuals = y_true - y_pred
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot residuals
    ax.scatter(y_pred, residuals, alpha=0.6, color=color, edgecolor='white', s=70)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Add metrics to the plot
    mae = mean_absolute_error(y_true, y_pred)
    std_residuals = np.std(residuals)
    
    metrics_text = f"{model_name}\n" \
                 f"MAE = {mae:.4f}\n" \
                 f"Std Residuals = {std_residuals:.4f}"
    
    # Position the text in the top-right corner
    ax.text(0.98, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title is None:
        title = f"{model_name}: Residuals Plot"
    ax.set_title(title, fontsize=14, pad=15)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_feature_importance(
    feature_importance: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    color: str = PRIMARY_COLOR,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot feature importance scores.
    
    Args:
        feature_importance: Array of feature importance scores
        feature_names: List of feature names
        model_name: Name of the model (for title)
        top_n: Number of top features to show
        figsize: Figure size (width, height)
        color: Color of the bars
        title: Plot title (if None, auto-generate)
        ax: Matplotlib Axes object (if None, create new figure)
        
    Returns:
        Tuple containing the Figure and Axes objects
    """
    # Sort features by importance
    indices = np.argsort(feature_importance)[-top_n:][::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importance = feature_importance[indices]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create horizontal bar plot
    y_pos = np.arange(len(sorted_features))
    ax.barh(y_pos, sorted_importance, align='center', color=color, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.invert_yaxis()  # Most important on top
    
    # Add value labels on the bars
    for i, v in enumerate(sorted_importance):
        ax.text(v + 0.01 * max(sorted_importance), i, f"{v:.3f}", 
                color='black', va='center', fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f"{model_name}: Top {top_n} Feature Importance" if title is None else title, 
                fontsize=14, pad=15)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    model_name: str = "Model",
    x_label: str = "Training Examples",
    y_label: str = "Score",
    score_name: str = "Score",
    figsize: Tuple[int, int] = (10, 6),
    train_color: str = PRIMARY_COLOR,
    test_color: str = SECONDARY_COLOR,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot learning curves for training and test sets.
    
    Args:
        train_sizes: Array of training set sizes
        train_scores: Array of training scores (mean and std)
        test_scores: Array of test scores (mean and std)
        model_name: Name of the model (for title)
        x_label: Label for x-axis
        y_label: Label for y-axis
        score_name: Name of the scoring metric
        figsize: Figure size (width, height)
        train_color: Color for training curve
        test_color: Color for test curve
        title: Plot title (if None, auto-generate)
        ax: Matplotlib Axes object (if None, create new figure)
        
    Returns:
        Tuple containing the Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Calculate mean and standard deviation for training and test scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curves with error bands
    ax.fill_between(train_sizes, 
                   train_scores_mean - train_scores_std,
                   train_scores_mean + train_scores_std,
                   alpha=0.1, color=train_color)
    ax.fill_between(train_sizes,
                   test_scores_mean - test_scores_std,
                   test_scores_mean + test_scores_std,
                   alpha=0.1, color=test_color)
    
    # Plot mean scores
    ax.plot(train_sizes, train_scores_mean, 'o-', color=train_color,
           label=f'Training {score_name}')
    ax.plot(train_sizes, test_scores_mean, 'o-', color=test_color,
           label=f'Cross-validation {score_name}')
    
    # Add final score annotations
    final_train_score = train_scores_mean[-1]
    final_test_score = test_scores_mean[-1]
    
    ax.annotate(f"{final_train_score:.3f}", 
               xy=(train_sizes[-1], final_train_score),
               xytext=(10, 0), textcoords='offset points',
               ha='left', va='center',
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
               color=train_color)
    
    ax.annotate(f"{final_test_score:.3f}",
               xy=(train_sizes[-1], final_test_score),
               xytext=(10, 0), textcoords='offset points',
               ha='left', va='center',
               bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
               color=test_color)
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title is None:
        title = f"{model_name}: Learning Curves"
    ax.set_title(title, fontsize=14, pad=15)
    
    # Add legend and grid
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax


def plot_gaussian_process(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    model_name: str = "Gaussian Process",
    x_label: str = "X",
    y_label: str = "y",
    figsize: Tuple[int, int] = (12, 8),
    train_color: str = PRIMARY_COLOR,
    pred_color: str = SECONDARY_COLOR,
    uncertainty_alpha: float = 0.2,
    title: Optional[str] = None,
    ax: Optional[Axes] = None
) -> Tuple[Figure, Axes]:
    """Plot Gaussian Process predictions with uncertainty.
    
    Args:
        X: Training data points (1D array)
        y: Training target values
        X_test: Test data points (1D array, sorted)
        y_pred: Predicted mean values
        y_std: Predicted standard deviation
        model_name: Name of the model (for title)
        x_label: Label for x-axis
        y_label: Label for y-axis
        figsize: Figure size (width, height)
        train_color: Color for training points
        pred_color: Color for predictions
        uncertainty_alpha: Alpha value for uncertainty band
        title: Plot title (if None, auto-generate)
        ax: Matplotlib Axes object (if None, create new figure)
        
    Returns:
        Tuple containing the Figure and Axes objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot training data
    ax.scatter(X, y, c=train_color, edgecolors='white', s=70, 
              label='Training Data', zorder=3)
    
    # Plot predicted mean
    ax.plot(X_test, y_pred, '-', color=pred_color, linewidth=2, 
           label='Predicted Mean', zorder=2)
    
    # Plot uncertainty (95% confidence interval)
    ax.fill_between(
        X_test.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color=pred_color, alpha=uncertainty_alpha,
        label='95% Confidence Interval'
    )
    
    # Add metrics to the plot
    mse = mean_squared_error(y, y_pred[:len(y)] if len(y_pred) > len(y) else y_pred)
    r2 = r2_score(y, y_pred[:len(y)] if len(y_pred) > len(y) else y_pred)
    
    metrics_text = f"{model_name}\n" \
                 f"MSE = {mse:.4f}\n" \
                 f"R² = {r2:.4f}"
    
    # Position the text in the top-right corner
    ax.text(0.98, 0.98, metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='lightgray'))
    
    # Set labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    
    if title is None:
        title = f"{model_name}: Predictions with Uncertainty"
    ax.set_title(title, fontsize=14, pad=15)
    
    # Add legend and grid
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax
