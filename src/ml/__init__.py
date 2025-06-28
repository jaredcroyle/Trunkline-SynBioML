"""
ML pipeline components for Trunkline 

This package contains all machine learning related modules including:
- Data preprocessing and feature engineering
- Model training and evaluation
- Ensemble methods
- SHAP visualization
- Design generation
"""

# Import key components 
from .data_preprocessing import load_and_clean_data, prepare_features
from .ml_pipeline import MLPipeline
from .model_evaluation import (
    plot_initial_data,
    plot_feature_importance_rf,
    plot_partial_dependence_rf,
    plot_predicted_vs_true,
    plot_residuals,
    plot_learning_curve,
    plot_predicted_vs_true_with_error
)
from .shap_visualization import shap_explainability
from .design_generator import run_design_generation
from .ensemble import WeightedEnsemble
from .ensemble_utils import EnsembleWrapper
from .feature_selection import shap_feature_selection
from .model_training import evaluate_regression_performance, scale_features

__all__ = [
    'load_and_clean_data',
    'prepare_features',
    'scale_features',
    'evaluate_regression_performance',
    'MLPipeline',
    'plot_initial_data',
    'plot_feature_importance_rf',
    'plot_partial_dependence_rf',
    'plot_predicted_vs_true',
    'plot_residuals',
    'plot_learning_curve',
    'plot_predicted_vs_true_with_error',
    'shap_explainability',
    'run_design_generation',
    'WeightedEnsemble',
    'EnsembleWrapper',
    'shap_feature_selection'
]
