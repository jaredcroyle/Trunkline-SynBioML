import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.exceptions import NotFittedError
from .data_preprocessing import load_and_clean_data, prepare_features
from .model_training import (
    train_gaussian_process, train_random_forest, train_gradient_boosting,
    train_linear_regression, train_ridge, train_svr, train_mlp, train_knn,
    evaluate_regression_performance, scale_features
)
from .feature_selection import shap_feature_selection as select_features
from .ensemble import WeightedEnsemble
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EstimatorWrapper:
    def __init__(self, model_func=None, **params):
        """Wrapper to make function-based models compatible with scikit-learn."""
        self.model_func = model_func
        self.params = params
        self.model = None
        self._is_fitted = False

    def fit(self, X, y):
        """Fit the model."""
        if self.model_func is None:
            raise ValueError("model_func must be provided")
        self.model = self.model_func(X, y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Make predictions."""
        if not self._is_fitted or self.model is None:
            raise NotFittedError("Model has not been fitted yet")
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.params.update(params)
        return self
        
    @property
    def _estimator_type(self):
        return "regressor"
        
    def score(self, X, y):
        """Return the coefficient of determination R^2 of the prediction."""
        if not self._is_fitted or self.model is None:
            raise NotFittedError("Model has not been fitted yet")
        return r2_score(y, self.predict(X))

    def __sklearn_clone__(self):
        """Clone the estimator."""
        return self.__class__(model_func=self.model_func, **self.params)

class MLPipeline:
    def __init__(self, config=None):
        """
        Initialize the ML pipeline with configuration options.
        
        Parameters:
        - config: dict, optional configuration parameters
        """
        self.config = config or {}
        self.models = {
            'gp': train_gaussian_process,
            'rf': train_random_forest,
            'gb': train_gradient_boosting,
            'lr': train_linear_regression,
            'ridge': train_ridge,
            'svr': train_svr,
            'mlp': train_mlp,
            'knn': train_knn
        }
        self.best_model = None
        self.best_params = None
        self.feature_selector = None
        self.pipeline = None
        self.scaler = None
        self._is_fitted = False

    def _get_estimator(self, model_type):
        """Get the EstimatorWrapper for the specified model type."""
        return EstimatorWrapper(self.models[model_type])
        
    def _setup_pipeline(self, X, feature_names):
        """
        Set up the preprocessing pipeline.
        """
        # Identify categorical and numerical features
        categorical_cols = [col for col in feature_names if '_code' in col]
        numerical_cols = [col for col in feature_names if col not in categorical_cols]
        
        # Create transformers
        numerical_transformer = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2))
        ])
        
        # For categorical features, use OneHotEncoder instead of LabelEncoder
        from sklearn.preprocessing import OneHotEncoder
        categorical_transformer = Pipeline([
            ('one_hot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        return preprocessor
    
    def fit(self, X, y, feature_names, model_type='rf', cv=5):
        """
        Train the pipeline with hyperparameter tuning and cross-validation.
        
        Parameters:
        - X: feature matrix
        - y: target variable
        - feature_names: list of feature names
        - model_type: str, type of model to train
        - cv: int, number of cross-validation folds
        """
        try:
            # Feature selection
            selected_features, mask = select_features(X, y, feature_names)
            logger.info(f"Selected features: {selected_features}")
            
            # Apply mask to X
            if isinstance(X, np.ndarray):
                X_selected = X[:, mask]
            else:  # pandas DataFrame
                X_selected = X.loc[:, selected_features]
            
            # Set up preprocessing pipeline
            preprocessor = self._setup_pipeline(X_selected, selected_features)
            
            # Create model with hyperparameter grid
            model = self._get_estimator(model_type)
            
            # Create full pipeline
            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Define scoring metrics
            scoring = {
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
            
            # Perform cross-validation
            cv_results = cross_validate(
                self.pipeline, 
                X_selected, 
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True,
                n_jobs=-1
            )
            
            # Store the best model from cross-validation
            best_idx = np.argmax(cv_results['test_r2'])
            self.best_model = cv_results['estimator'][best_idx]
            self._is_fitted = True
            
            # Log cross-validation results
            logger.info("\nCross-validation results:")
            logger.info(f"  Mean CV R²: {np.mean(cv_results['test_r2']):.4f} (±{np.std(cv_results['test_r2']):.4f})")
            logger.info(f"  Mean CV MSE: {-np.mean(cv_results['test_mse']):.4f} (±{np.std(cv_results['test_mse']):.4f})")
            logger.info(f"  Mean CV MAE: {-np.mean(cv_results['test_mae']):.4f} (±{np.std(cv_results['test_mae']):.4f})")
            
            # Train final model on full dataset
            self.best_model.fit(X_selected, y)
            
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error during pipeline fitting: {str(e)}")
            raise
    
    def _get_param_grid(self, model_type):
        """
        Get hyperparameter grid for different models.
        """
        param_grids = {
            'rf': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30],
                'model__min_samples_split': [2, 5, 10]
            },
            'gb': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            },
            'lr': {
                'model__fit_intercept': [True, False]
            },
            'ridge': {
                'model__alpha': [0.1, 1.0, 10.0]
            }
        }
        
        return param_grids.get(model_type, {})
    
    def evaluate(self, X, y, cv=5):
        """
        Evaluate the model using cross-validation with multiple metrics.
        
        Returns:
        - dict: Dictionary containing train and test metrics with their standard deviations
        """
        if not self._is_fitted or self.best_model is None:
            raise NotFittedError("Model has not been fitted yet")
            
        try:
            # Define scoring metrics
            scoring = {
                'mse': 'neg_mean_squared_error',
                'mae': 'neg_mean_absolute_error',
                'r2': 'r2'
            }
            
            # Perform cross-validation
            cv_results = cross_validate(
                self.best_model, 
                X, 
                y,
                cv=cv,
                scoring=scoring,
                return_train_score=True
            )
            
            # Calculate metrics
            metrics = {
                'train': {
                    'r2': np.mean(cv_results['train_r2']),
                    'mse': -np.mean(cv_results['train_mse']),
                    'mae': -np.mean(cv_results['train_mae']),
                    'r2_std': np.std(cv_results['train_r2']),
                    'mse_std': np.std(cv_results['train_mse']),
                    'mae_std': np.std(cv_results['train_mae'])
                },
                'test': {
                    'r2': np.mean(cv_results['test_r2']),
                    'mse': -np.mean(cv_results['test_mse']),
                    'mae': -np.mean(cv_results['test_mae']),
                    'r2_std': np.std(cv_results['test_r2']),
                    'mse_std': np.std(cv_results['test_mse']),
                    'mae_std': np.std(cv_results['test_mae'])
                }
            }
            
            # Log metrics
            logger.info("\nModel evaluation metrics (mean ± std):")
            for split in ['train', 'test']:
                logger.info(f"  {split.capitalize()}:")
                logger.info(f"    R²: {metrics[split]['r2']:.4f} ± {metrics[split]['r2_std']:.4f}")
                logger.info(f"    MSE: {metrics[split]['mse']:.4f} ± {metrics[split]['mse_std']:.4f}")
                logger.info(f"    MAE: {metrics[split]['mae']:.4f} ± {metrics[split]['mae_std']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
    
    def predict(self, X):
        """
        Make predictions using the best model.
        
        Parameters:
        - X: Feature matrix to make predictions on
        
        Returns:
        - array-like: Predicted values
        """
        if not self._is_fitted or self.best_model is None:
            raise NotFittedError("Model has not been fitted yet")
            
        try:
            return self.best_model.predict(X)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise
    
    def create_ensemble(self, X, y, models=None):
        """
        Create a weighted ensemble model from multiple base models.
        """
        if models is None:
            models = ['rf', 'gb', 'lr']
            
        try:
            # Get the base models
            base_models = [self.models[m](X, y) for m in models]
            
            # Create meta-model
            from sklearn.linear_model import BayesianRidge
            meta_model = BayesianRidge()
            
            # Get base model predictions
            base_preds = np.column_stack([m.predict(X) for m in base_models])
            
            # Train meta-model
            meta_model.fit(base_preds, y)
            
            # Create ensemble
            ensemble = WeightedEnsemble(base_models, meta_model=meta_model)
            return ensemble
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            raise
