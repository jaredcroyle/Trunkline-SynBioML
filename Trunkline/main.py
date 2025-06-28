#!/usr/bin/env python3
"""
Trunkline ML Pipeline - Main Entry Point
"""
import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from logging.handlers import RotatingFileHandler

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import from src package
    from Trunkline.src.ml_pipeline import MLPipeline
    from Trunkline.src.config_loader import load_config, validate_config
    from Trunkline.src.data_preprocessing import load_and_clean_data, prepare_features
    from Trunkline.src.model_training import scale_features, evaluate_regression_performance
    from Trunkline.src.feature_selection import shap_feature_selection
    from Trunkline.src.model_evaluation import (
        plot_initial_data, predict_test_points_generic, plot_surface_predictions_generic,
        plot_feature_importance_rf, plot_partial_dependence_rf, plot_predicted_vs_true,
        plot_residuals, plot_learning_curve, plot_predicted_vs_true_with_error
    )
    from Trunkline.src.shap_visualization import shap_explainability
    from Trunkline.src.report_generator import generate_report
    from Trunkline.src.design_generator import run_design_generation
    from Trunkline.src.ensemble import WeightedEnsemble
    from Trunkline.src.ensemble_utils import EnsembleWrapper
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.linear_model import BayesianRidge
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print("Please make sure you're running the script from the TrunklineML directory.")
    sys.exit(1)


def setup_logging(log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Optional path to the log file. If not provided, defaults to 'logs/trunkline.log'
        
    Returns:
        Configured logger instance
    """
    # Set default log file if not provided
    if log_file is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'trunkline.log')
    else:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    
    # Set up file rotation (keep last 5 log files, 1MB each)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=1024*1024,  # 1MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    return logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trunkline ML Pipeline')
    
    # Data arguments
    default_data_path = Path("data") / "Limonene_data.csv"
    parser.add_argument('--data_path', 
                      type=str, 
                      default=str(default_data_path),
                      help=f'Path to input data file (CSV format). Default: {default_data_path}')
    
    parser.add_argument('--output_dir', 
                      type=str, 
                      default='results',
                      help='Directory to save output files. Default: results/')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='rf',
                      choices=['rf', 'gb', 'svm'],
                      help='Type of model to train (rf: Random Forest, gb: Gradient Boosting, svm: Support Vector Machine)')
    parser.add_argument('--build_model', action='store_true', default=True,
                      help='Whether to build a new model')
    parser.add_argument('--save_model', action='store_true', default=True,
                      help='Whether to save the trained model')
    
    # Feature selection arguments
    parser.add_argument('--use_shap', action='store_true', default=True,
                      help='Whether to use SHAP for feature selection')
    parser.add_argument('--num_features', type=int, default=10,
                      help='Number of top features to select')
    
    # Visualization arguments
    parser.add_argument('--generate_plots', action='store_true', default=True,
                      help='Whether to generate visualization plots')
    parser.add_argument('--plot_types', type=str, nargs='+', 
                      default=['learning_curve', 'feature_importance', 'residuals'],
                      help='Types of plots to generate')
    
    # Logging arguments
    parser.add_argument('--log_file', type=str, default=None,
                      help='Path to log file (default: logs/trunkline.log)')
    
    return parser.parse_args()


def main():
    """Main function to run the Trunkline ML Pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Convert to Path objects
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    
    # Setup logging
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"
    
    logger = setup_logging(str(log_file))
    logger.info("Starting Trunkline ML Pipeline")
    logger.info(f"Data path: {data_path.absolute()}")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # Check if data file exists
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path.absolute()}")
        logger.info("Please provide a valid path to the input data file using --data_path")
        sys.exit(1)
    
    # Create output directories
    plots_dir = output_dir / 'plots'
    models_dir = output_dir / 'models'
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = MLPipeline()
    
    # Load and clean data
    logger.info(f"Loading data from {data_path}")
    df_clean = load_and_clean_data(str(data_path))  # Convert Path to string for compatibility
    X, y, feature_names = prepare_features(df_clean)
    
    # Check if we have enough data
    if len(X) < 10:
        logger.error(f"Insufficient data samples: {len(X)}. Need at least 10 samples.")
        sys.exit(1)
        
    X_scaled, scaler = scale_features(X)
    logger.info(f"Loaded data with {X.shape[0]} samples and {X.shape[1]} features")

    # Feature selection
    if args.use_shap and len(feature_names) > 5:  # Only run SHAP if we have enough features
        num_features = min(args.num_features, len(feature_names))
        logger.info(f"Performing SHAP feature selection to select {num_features} features")
        selected_features, mask = shap_feature_selection(
            X_scaled, y, feature_names, num_features=num_features
        )
        logger.info(f"Selected top {num_features} features by SHAP: {selected_features}")
        
        if isinstance(X_scaled, np.ndarray):
            X_scaled = X_scaled[:, mask]
        else:
            X_scaled = X_scaled.loc[:, selected_features]
            feature_names = selected_features
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # Plot initial data if visualization is enabled
    if args.generate_plots:
        try:
            initial_data_plot = plots_dir / "initial_data.png"
            plot_initial_data(X_scaled_df, y, save_path=str(initial_data_plot))
            logger.info(f"Saved initial data plot to {initial_data_plot}")
        except ImportError as e:
            logger.warning(f"Could not import plotting libraries: {e}")
            logger.info("Install matplotlib and seaborn for visualizations")
            args.generate_plots = False  # Disable further plotting
        except Exception as e:
            logger.warning(f"Failed to generate initial data plot: {e}")
            logger.debug("Plot error details:", exc_info=True)

    # Model training or loading
    if args.build_model:
        try:
            # Define which models to train (can be customized)
            model_types = ['rf', 'gb', 'lr']  # Default models to train
            
            # Train all specified models
            logger.info(f"Training models: {', '.join(model_types).upper()}")
            best_model = pipeline.fit(X_scaled_df, y, feature_names, model_type=model_types)
            
            # Save all trained models
            if args.save_model:
                for model_name, model in pipeline.trained_models.items():
                    model_path = models_dir / f"model_{model_name}.joblib"
                    try:
                        joblib.dump(model, model_path)
                        logger.info(f"Saved {model_name} model to {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to save {model_name} model: {e}")
                        continue
            
            # Create and save ensemble
            logger.info("Creating ensemble model...")
            ensemble = pipeline.create_ensemble(X_scaled_df, y, models=model_types)
            
            if args.save_model and ensemble is not None:
                try:
                    ensemble_path = models_dir / "model_ensemble.joblib"
                    joblib.dump(ensemble, ensemble_path)
                    logger.info(f"Saved ensemble model to {ensemble_path}")
                    
                    # Also save predictions from all models for comparison
                    all_predictions = {}
                    
                    # Add predictions from individual models
                    for model_name in model_types:
                        if model_name in pipeline.trained_models:
                            try:
                                y_pred = pipeline.predict(X_scaled_df, model_name=model_name)
                                all_predictions[f"{model_name}_pred"] = y_pred
                            except Exception as e:
                                logger.warning(f"Could not get predictions from {model_name}: {e}")
                    
                    # Add ensemble predictions if available
                    if 'ensemble' in pipeline.trained_models:
                        try:
                            y_pred_ensemble = pipeline.predict(X_scaled_df, model_name='ensemble')
                            all_predictions['ensemble_pred'] = y_pred_ensemble
                        except Exception as e:
                            logger.warning(f"Could not get ensemble predictions: {e}")
                    
                    # Save predictions to CSV
                    if all_predictions:
                        predictions_df = pd.DataFrame({
                            'actual': y,
                            **all_predictions
                        })
                        predictions_path = output_dir / 'all_model_predictions.csv'
                        predictions_df.to_csv(predictions_path, index=False)
                        logger.info(f"Saved all model predictions to {predictions_path}")
                        
                except Exception as e:
                    logger.error(f"Error saving ensemble model or predictions: {e}")
                    logger.debug("Error details:", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error during model training: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Load existing models
        loaded_models = {}
        
        # Load individual models
        for model_type in ['rf', 'gb', 'lr', 'ensemble']:
            model_path = models_dir / f"model_{model_type}.joblib"
            if model_path.exists():
                loaded_models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model from {model_path}")
                
                # Set the best model (prefer RF if available, otherwise first available)
                if model_type == 'rf' or 'best_model' not in locals():
                    best_model = loaded_models[model_type]
                    pipeline.best_model = best_model
                    pipeline.best_model_name = model_type
        
        if not loaded_models:
            logger.error(f"No model files found in {models_dir}")
            sys.exit(1)
            
        # Update pipeline's trained_models
        pipeline.trained_models.update(loaded_models)
        
        # Set ensemble if available
        ensemble = loaded_models.get('ensemble')

    # Evaluate all models
    logger.info("Evaluating all models...")
    
    # Create a comprehensive report
    report_path = output_dir / 'model_report.txt'
    metrics_data = []
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Model Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {data_path.name}\n")
        f.write(f"Number of Samples: {X_scaled_df.shape[0]}\n")
        f.write(f"Number of Features: {len(feature_names)}\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # Evaluate each model
    for model_name in sorted(pipeline.trained_models.keys()):
        # Skip non-model entries
        if model_name in ['preprocessor', 'model']:
            continue
            
        logger.info(f"Evaluating {model_name}...")
        metrics = pipeline.evaluate_model(model_name, X_scaled_df, y)
        
        # Save metrics for later comparison
        metrics_data.append({
            'Model': model_name.upper(),
            'MSE': metrics[0],
            'RMSE': metrics[1],
            'MAE': metrics[2],
            'R²': metrics[3]
        })
        # Write to report
        with open(report_path, 'a') as f:
            f.write(f"\n{model_name.upper()} Model\n")
            f.write("-" * 40 + "\n")
            f.write(f"MSE: {metrics[0]:.4f}\n")
            f.write(f"RMSE: {metrics[1]:.4f}\n")
            f.write(f"MAE: {metrics[2]:.4f}\n")
            f.write(f"R² Score: {metrics[3]:.4f}\n")
            
            # If this is the ensemble, add more details
            if model_name == 'ensemble' and hasattr(pipeline, 'ensemble_model'):
                f.write("\nEnsemble Weights:\n")
                if hasattr(pipeline.ensemble_model, 'meta_model') and \
                   hasattr(pipeline.ensemble_model.meta_model, 'coef_'):
                    weights = pipeline.ensemble_model.meta_model.coef_
                    for i, model in enumerate(pipeline.ensemble_model.base_models):
                        model_type = type(model).__name__
                        f.write(f"  - {model_type}: {weights[i]:.4f}\n")

        # Plot initial data if visualization is enabled
        if args.generate_plots:
            initial_data_plot = plots_dir / "initial_data.png"
            plot_initial_data(X_scaled_df, y, save_path=str(initial_data_plot))
            logger.info(f"Saved initial data plot to {initial_data_plot}")

        # Add comparison table if metrics data exists
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data).set_index('Model')
            with open(report_path, 'a') as f:
                f.write("\nModel Comparison\n")
                f.write("-" * 40 + "\n")
                f.write(df_metrics.to_string(float_format='{:.4f}'.format))

        logger.info(f"Saved comprehensive model report to {report_path}")
        
        # Make predictions with the best model if available
        if hasattr(pipeline, 'best_model_name'):
            best_model_name = pipeline.best_model_name
            logger.info(f"Making predictions with best model ({best_model_name})...")
            y_pred = pipeline.predict(X_scaled_df)
            
            # Generate prediction plots
            plot_path = plots_dir / f'{best_model_name}_predictions.png'
            plot_predicted_vs_true(y, y_pred, model_name=best_model_name, save_path=str(plot_path))
            logger.info(f"Saved prediction plot to {plot_path}")
            
            # Plot residuals
            residuals_path = plots_dir / f'{best_model_name}_residuals.png'
            plot_residuals(y, y_pred, model_name=best_model_name, save_path=str(residuals_path))
            logger.info(f"Saved residuals plot to {residuals_path}")
            
            # If it's a random forest, plot feature importance
            if best_model_name == 'rf' and 'rf' in pipeline.trained_models:
                importance_path = plots_dir / 'feature_importance.png'
                plot_feature_importance_rf(
                    pipeline.trained_models['rf'], 
                    feature_names,
                    save_path=str(importance_path)
                )
                logger.info(f"Saved feature importance plot to {importance_path}")
                    
            # Generate SHAP explainer plot if requested
            if args.shap and best_model_name in pipeline.trained_models:
                logger.info("Generating SHAP explainer plot...")
                shap_path = plots_dir / 'shap_summary.png'
                shap_explainability(
                    pipeline.trained_models[best_model_name],
                    X_scaled_df,
                    feature_names=feature_names,
                    save_path=str(shap_path)
                )
                logger.info(f"Saved SHAP summary plot to {shap_path}")
                
        # Generate residuals plot for each model if plots are enabled
        if args.generate_plots and hasattr(pipeline, 'trained_models'):
            for model_name in pipeline.trained_models:
                if model_name not in ['preprocessor', 'model']:  # Skip non-model entries
                    res_path = plots_dir / f"residuals_{model_name}.png"
                    logger.info(f"Generating residuals plot for {model_name} at {res_path}")
                    plot_residuals(
                        y,
                        pipeline.predict(X_scaled_df, model_name=model_name),
                        model_name=model_name.upper(),
                        save_path=str(res_path)
                    )
                    
            # Generate predicted vs actual plot for each model
            for model_name in pipeline.trained_models:
                if model_name not in ['preprocessor', 'model']:  # Skip non-model entries
                    pvt_path = plots_dir / f"predicted_vs_actual_{model_name}.png"
                    logger.info(f"Generating predicted vs actual plot for {model_name} at {pvt_path}")
                    plot_predicted_vs_true(
                        y,
                        pipeline.predict(X_scaled_df, model_name=model_name),
                        model_name=model_name.upper(),
                        save_path=str(pvt_path)
                    )
            
            # SHAP Summary Plot (if SHAP is installed and enabled)
            if args.shap and hasattr(pipeline, 'model') and hasattr(pipeline.model, 'predict'):
                try:
                    import shap
                    logger.info("Generating SHAP summary plot...")
                    
                    # Create SHAP explainer
                    explainer = shap.Explainer(pipeline.model, X_scaled_df)
                    shap_values = explainer(X_scaled_df)
                    
                    # Summary plot
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_scaled_df, show=False, plot_type="dot")
                    plt.tight_layout()
                    shap_summary_path = os.path.join(plots_dir, "shap_summary.png")
                    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved SHAP summary plot to {shap_summary_path}")

                    # Bar plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values, X_scaled_df, plot_type="bar", show=False)
                    plt.tight_layout()
                    shap_bar_path = os.path.join(plots_dir, "shap_bar.png")
                    plt.savefig(shap_bar_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Saved SHAP bar plot to {shap_bar_path}")

                except ImportError:
                    logger.warning("SHAP is not installed. Install with: pip install shap")
                except Exception as e:
                    logger.error(f"Error generating SHAP plots: {str(e)}")

        # Save the pipeline if requested
        if args.save_model:
            model_path = os.path.join('saved_models', f'trunkline_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib')
            os.makedirs('saved_models', exist_ok=True)
            joblib.dump(pipeline, model_path)
            logger.info(f"Saved trained pipeline to {model_path}")

        logger.info("Trunkline ML Pipeline execution completed successfully!")

        # Generate design if requested
        if args.generate_design:
            logger.info("Generating design...")
            try:
                design = generate_design(pipeline, X_scaled_df, y, feature_names)
                design_path = os.path.join('design', 'generated_design.json')
                os.makedirs('design', exist_ok=True)
                with open(design_path, 'w') as f:
                    json.dump(design, f, indent=2)
                logger.info(f"Generated design saved to {design_path}")
            except Exception as e:
                logger.error(f"Error generating design: {str(e)}")
                return  # Exit the function if there was an error

        # Save predictions for all models to CSV if output directory is specified
        if hasattr(args, 'output_dir') and hasattr(pipeline, 'trained_models'):
            all_predictions = {}
            all_predictions['actual'] = y

            # Add predictions from all individual models
            for model_name, model in pipeline.trained_models.items():
                if model_name not in ['preprocessor', 'model']:  # Skip non-model entries
                    y_pred = pipeline.predict(X_scaled_df, model_name=model_name)
                    all_predictions[f"{model_name}_pred"] = y_pred
                    all_predictions[f"{model_name}_residual"] = y - y_pred

            # Add ensemble predictions if available
            if 'ensemble' in pipeline.trained_models:
                y_pred_ensemble = pipeline.predict(X_scaled_df, model_name='ensemble')
                all_predictions['ensemble_pred'] = y_pred_ensemble
                all_predictions['ensemble_residual'] = y - y_pred_ensemble

                # Save all predictions to CSV
                if all_predictions:
                    predictions_df = pd.DataFrame(all_predictions)
                    predictions_path = os.path.join(args.output_dir, 'all_model_predictions.csv')
                    predictions_df.to_csv(predictions_path, index=False)
                    logger.info(f"Saved all model predictions to {predictions_path}")

                    # Generate a summary of model performance
                    summary = []
                    for col in predictions_df.columns:
                        if col.endswith('_pred') or col == 'ensemble_pred':
                            model_name = col.replace('_pred', '')
                            y_true = predictions_df['actual']
                            y_pred = predictions_df[col]
                            
                            mse = mean_squared_error(y_true, y_pred)
                            mae = mean_absolute_error(y_true, y_pred)
                            r2 = r2_score(y_true, y_pred)
                            
                            summary.append({
                                'Model': model_name,
                                'MSE': mse,
                                'RMSE': np.sqrt(mse),
                                'MAE': mae,
                                'R²': r2
                            })
                    
                    # Save summary to CSV
                    if summary:
                        try:
                            summary_df = pd.DataFrame(summary).sort_values('R²', ascending=False)
                            summary_path = output_dir / 'model_performance_summary.csv'
                            summary_df.to_csv(summary_path, index=False)
                            logger.info(f"Saved model performance summary to {summary_path}")
                            
                            # Print top 3 models
                            print("\nTop performing models:")
                            print(summary_df.head(3).to_string(index=False))
                        except Exception as e:
                            logger.error(f"Error saving performance summary: {e}")

            logger.info("Pipeline execution completed!")
            print("\nPipeline execution completed!")
            print(f"Results saved to: {output_dir.absolute()}")
            
            # Print path to the model report
            report_path = output_dir / 'model_report.txt'
            if report_path.exists():
                print(f"\nModel report: {report_path.absolute()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)