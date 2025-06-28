import json
import os
import sys
import yaml
from pathlib import Path

# Add the src directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Load configuration
config_path = os.path.join(project_root, 'biolab', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

from src.ml import (
    MLPipeline,
    plot_initial_data,
    plot_feature_importance_rf,
    plot_partial_dependence_rf,
    plot_predicted_vs_true,
    plot_residuals,
    plot_learning_curve,
    plot_predicted_vs_true_with_error,
    shap_explainability,
    run_design_generation,
    WeightedEnsemble,
    EnsembleWrapper,
    load_and_clean_data,
    prepare_features,
    scale_features,
    evaluate_regression_performance,
    shap_feature_selection
)
from src.report_generator import generate_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    # Ensure logs directory exists
    log_file = config.get('logging.file', 'logs/trunkline.log')
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=config.get('logging.level', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def create_directories():
    """Create required directories."""
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # List of directories to create
    dirs = [
        os.path.join(project_root, config['data']['model_dir']),
        os.path.join(project_root, config['data']['plots_dir']),
        os.path.join(project_root, config['data']['report_dir']),
        os.path.join(project_root, 'logs')
    ]
    
    # Create each directory if it doesn't exist
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def validate_config():
    """Validate configuration values."""
    # Check required top-level sections
    for section in ['model', 'data', 'logging', 'ensemble', 'feature_selection']:
        if section not in config:
            config[section] = {}
    
    # Set default values for required fields
    defaults = {
        'model': {
            'type': 'rf',
            'num_features': 10,
            'build_new': True
        },
        'data': {
            'input_file': 'data/Limonene_data.csv',
            'output_dir': 'results/',
            'model_dir': 'results/saved_models/',
            'plots_dir': 'results/plots/',
            'report_dir': 'results/'
        },
        'logging': {
            'level': 'INFO',
            'file': 'results/pipeline.log'
        },
        'ensemble': {
            'use_ensemble': True,
            'models': ['rf', 'gp', 'lr']
        },
        'feature_selection': {
            'method': 'shap',
            'num_features': 10
        }
    }
    
    # Apply defaults for any missing values
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = section_defaults
        else:
            for key, value in section_defaults.items():
                if key not in config[section]:
                    config[section][key] = value

def main():
    try:
        # Set up project root and paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create required directories first
        create_directories()
        
        # Set up logging
        log_file = os.path.join(project_root, config['logging']['file'])
        logging.basicConfig(
            level=config['logging']['level'],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info("=" * 50)
        logger.info("Starting NonaTalks25ML Pipeline")
        logger.info("=" * 50)
        
        # Validate and update configuration
        validate_config()
        
        # Log configuration
        logger.info("Configuration:")
        for section, values in config.items():
            logger.info(f"  {section}:")
            for key, value in values.items():
                logger.info(f"    {key}: {value}")
        
        # Load and clean data
        input_file = os.path.join(project_root, config['data']['input_file'])
        logger.info(f"Loading data from: {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        df_clean = load_and_clean_data(input_file)
        X, y, feature_names = prepare_features(df_clean)
        X_scaled, scaler = scale_features(X)

        # Feature selection
        selected_features = None
        if config.get('feature_selection', {}).get('method') == 'shap':
            num_features = config.get('feature_selection', {}).get('num_features', 10)
            logger.info(f"Performing SHAP feature selection to select {num_features} features...")
            selected_features, feature_mask = shap_feature_selection(X_scaled, y, feature_names=feature_names, num_features=num_features)
            logger.info(f"Selected features: {selected_features}")
            
            # Apply feature mask to X_scaled and convert back to DataFrame
            X_scaled = pd.DataFrame(X_scaled[:, feature_mask], columns=selected_features)
            feature_names = selected_features  # Update feature names to only include selected features
        
        # Model training
        model_type = config['model']['type']
        model_path = os.path.join(project_root, config['data']['model_dir'], f'best_model_{model_type}.joblib')
        
        if config['model']['build_new']:
            logger.info(f"Training new {model_type.upper()} model with hyperparameter tuning...")
            pipeline = MLPipeline()
            best_model = pipeline.fit(X_scaled, y, selected_features, model_type=model_type)
            
            # Save the best model
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(best_model, model_path)
            logger.info(f"Saved best model to {model_path}")
            
            # Create and save ensemble model if enabled
            if config['ensemble']['use_ensemble']:
                logger.info("Creating ensemble model...")
                # Create base models
                from src.ml.model_training import train_random_forest, train_gradient_boosting, train_linear_regression
                
                # Train base models
                rf_model = train_random_forest(X_scaled, y)
                gb_model = train_gradient_boosting(X_scaled, y)
                lr_model = train_linear_regression(X_scaled, y)
                
                # Create ensemble with base models
                ensemble = WeightedEnsemble(
                    base_models=[rf_model, gb_model, lr_model],
                    weights=None  # Equal weights
                )
                
                # Fit the ensemble
                ensemble.fit(X_scaled, y)
                
                # Save the ensemble
                ensemble_path = os.path.join(project_root, config['data']['model_dir'], 'ensemble_model.joblib')
                os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
                joblib.dump(ensemble, ensemble_path)
                logger.info(f"Saved ensemble model to {ensemble_path}")
                
                # Also save the best model from the pipeline
                best_model_path = os.path.join(project_root, config['data']['model_dir'], 'best_rf_model.joblib')
                joblib.dump(best_model, best_model_path)
                logger.info(f"Saved best model to {best_model_path}")
        else:
            # Load existing model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
                
            logger.info(f"Loading existing model from {model_path}")
            best_model = joblib.load(model_path)
            
            # Load ensemble if enabled
            if config['ensemble']['use_ensemble']:
                ensemble_path = os.path.join(project_root, config['data']['model_dir'], 'ensemble_model.joblib')
                if not os.path.exists(ensemble_path):
                    logger.warning(f"Ensemble model not found at {ensemble_path}")
                else:
                    ensemble = joblib.load(ensemble_path)
                    logger.info(f"Loaded ensemble model from {ensemble_path}")

        # Model evaluation
        logger.info("Evaluating model performance...")
        metrics = pipeline.evaluate(X_scaled, y)
        logger.info(f"Model evaluation metrics: {metrics}")

        # Save metrics
        metrics_dir = os.path.join(project_root, config['data']['results_dir'])
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, 'model_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'mse': float(metrics[0]),
                'rmse': float(metrics[1]),
                'mae': float(metrics[2]),
                'r2': float(metrics[3])
            }, f, indent=2)
        logger.info(f"Saved model metrics to {metrics_path}")

        # Generate report
        report_path = os.path.join(project_root, config['data']['report_dir'], 'model_report.html')
        metrics_dict = {
            'best_model': {
                'mse': metrics[0],
                'rmse': metrics[1],
                'mae': metrics[2],
                'r2': metrics[3],
                'model_type': model_type,
                'features': selected_features
            }
        }
        
        logger.info(f"Generating report at {report_path}")
        generate_report(metrics_dict, report_path)
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise