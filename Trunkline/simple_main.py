#!/usr/bin/env python3
"""
Simplified Trunkline ML Pipeline
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core components
try:
    from Trunkline.src.ml_pipeline import MLPipeline
    from Trunkline.src.data_preprocessing import load_and_clean_data, prepare_features, scale_features
    from Trunkline.src.feature_selection import shap_feature_selection
    from Trunkline.src.model_evaluation import (
        plot_initial_data, plot_feature_importance_rf, plot_partial_dependence_rf,
        plot_residuals, plot_learning_curve, plot_predicted_vs_true_with_error
    )
    from Trunkline.src.shap_visualization import shap_explainability
    from Trunkline.src.report_generator import generate_report
    from Trunkline.src.design_generator import run_design_generation
    from Trunkline.src.ensemble import WeightedEnsemble
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Please make sure you're running the script from the TrunklineML directory.")
    sys.exit(1)

def setup_directories(output_dir: str = 'results') -> Tuple[Path, Path, Path]:
    """Create necessary output directories."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / 'plots'
    models_dir = output_dir / 'models'
    
    for directory in [output_dir, plots_dir, models_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        
    return output_dir, plots_dir, models_dir

def main():
    # Configuration
    config = {
        'data_path': 'data/Limonene_data.csv',
        'output_dir': 'results',
        'model_type': 'rf',  # 'rf', 'gb', or 'svm'
        'use_shap_feature_selection': True,
        'num_features': 10,
        'build_model': True,
        'generate_plots': True
    }
    
    # Setup directories
    output_dir, plots_dir, models_dir = setup_directories(config['output_dir'])
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df_clean = load_and_clean_data(config['data_path'])
    X, y, feature_names = prepare_features(df_clean)
    X_scaled, scaler = scale_features(X)
    
    # Feature selection
    if config['use_shap_feature_selection'] and len(feature_names) > 5:
        print("Performing SHAP feature selection...")
        selected_features, mask = shap_feature_selection(
            X_scaled, y, feature_names, num_features=min(config['num_features'], len(feature_names))
        )
        print(f"Selected features: {selected_features}")
        
        if isinstance(X_scaled, np.ndarray):
            X_scaled = X_scaled[:, mask]
        else:
            X_scaled = X_scaled.loc[:, selected_features]
        feature_names = selected_features
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Plot initial data
    if config['generate_plots']:
        plot_initial_data(X_scaled_df, y, save_path=str(plots_dir / "initial_data.png"))
    
    # Model training or loading
    if config['build_model']:
        print("\nTraining model...")
        best_model = pipeline.fit(X_scaled_df, y, feature_names, model_type=config['model_type'])
        
        # Save the model
        model_path = models_dir / f"best_model_{config['model_type']}.joblib"
        joblib.dump(best_model, model_path)
        print(f"Model saved to {model_path}")
        
        # Create and save ensemble
        print("\nCreating ensemble...")
        ensemble = pipeline.create_ensemble(X_scaled_df, y)
        ensemble_path = models_dir / "ensemble_model.joblib"
        joblib.dump(ensemble, ensemble_path)
        print(f"Ensemble model saved to {ensemble_path}")
    else:
        # Load existing models
        model_path = models_dir / f"best_model_{config['model_type']}.joblib"
        ensemble_path = models_dir / "ensemble_model.joblib"
        
        if not model_path.exists() or not ensemble_path.exists():
            raise FileNotFoundError("Model files not found. Set build_model=True to train new models.")
            
        best_model = joblib.load(model_path)
        ensemble = joblib.load(ensemble_path)
        print("Loaded existing models")
    
    # Evaluate models
    print("\nEvaluating models...")
    models_results = {}
    
    # Evaluate base model
    for name, model in [(config['model_type'].upper(), best_model), ('Ensemble', ensemble)]:
        y_pred = model.predict(X_scaled_df)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        print(f"\n{name} Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Generate plots
        if config['generate_plots']:
            prefix = name.lower().replace(' ', '_')
            
            # Basic evaluation plots
            plot_learning_curve(model, X_scaled_df, y, model_name=name, 
                              save_path=str(plots_dir / f"{prefix}_learning_curve.png"))
            plot_residuals(model, X_scaled_df, y, model_name=name,
                         save_path=str(plots_dir / f"{prefix}_residuals.png"))
            
            # Feature importance if available
            try:
                plot_feature_importance_rf(model, feature_names, model_name=name,
                                       save_path=str(plots_dir / f"{prefix}_feature_importance.png"))
            except Exception as e:
                print(f"Could not generate feature importance for {name}: {e}")
            
            # SHAP explainability
            try:
                shap_summary = plots_dir / f"{prefix}_shap_summary.png"
                shap_bar = plots_dir / f"{prefix}_shap_bar.png"
                shap_explainability(model, X_scaled_df, feature_names, 
                                 str(shap_summary), str(shap_bar))
            except Exception as e:
                print(f"Could not generate SHAP plots for {name}: {e}")
        
        models_results[name] = metrics
    
    # Generate HTML report
    print("\nGenerating report...")
    generate_report(
        models_results=models_results,
        output_path=str(output_dir / "report.html"),
        template_folder="templates",
        template_file="model_report_template.html"
    )
    
    # Generate new designs
    print("\nGenerating new designs...")
    top_designs = run_design_generation(
        feature_names=feature_names,
        existing_df=X_scaled_df,
        model={"base_models": [best_model], "ensemble_model": ensemble},
        scaler=scaler,
        output_path=str(output_dir / "generated_designs.csv"),
        PI="Your Name",
        PI_email="your.email@example.com",
        top_n=96,
        include_wild_type=True,
        modif_code_for_NoMod=1,
        is_ensemble=True
    )
    
    print("\nPipeline execution completed successfully!")
    print(f"Results saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
