import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.exceptions import NotFittedError
from Trunkline.src.data_preprocessing import load_and_clean_data, prepare_features
from Trunkline.src.model_training import (
    train_gaussian_process, train_random_forest, train_gradient_boosting,
    train_linear_regression, train_ridge, train_svr, train_mlp, train_knn,
    evaluate_regression_performance
)
from Trunkline.src.feature_selection import shap_feature_selection as select_features
from Trunkline.src.ensemble import WeightedEnsemble
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

    def fit(self, X, y):
        """Fit the model."""
        if self.model_func is None:
            raise ValueError("model_func must be provided")
        self.model = self.model_func(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model has not been trained")
        return self.model.predict(X)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.params.update(params)
        return self

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
        self.model_factories = {
            'gp': train_gaussian_process,
            'rf': train_random_forest,
            'gb': train_gradient_boosting,
            'lr': train_linear_regression,
            'ridge': train_ridge,
            'svr': train_svr,
            'mlp': train_mlp,
            'knn': train_knn
        }
        self.trained_models = {}  # Store all trained models
        self.best_model = None
        self.best_model_name = None
        self.best_params = None
        self.feature_selector = None
        self.pipeline = None
        self.ensemble_model = None

    def _get_estimator(self, model_type):
        """Get the EstimatorWrapper for the specified model type."""
        return EstimatorWrapper(self.model_factories[model_type])
        
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
        Train the pipeline with hyperparameter tuning.
        
        Parameters:
        - X: feature matrix
        - y: target variable
        - feature_names: list of feature names
        - model_type: str or list, type(s) of model(s) to train
        - cv: int, number of cross-validation folds
        """
        try:
            # Feature selection
            selected_features, mask = select_features(X, y, feature_names)
            print(f"Selected features: {selected_features}")
            
            # Apply mask to X
            if isinstance(X, np.ndarray):
                X_selected = X[:, mask]
            else:  # pandas DataFrame
                X_selected = X.loc[:, selected_features]
            
            # Set up preprocessing
            preprocessor = self._setup_pipeline(X_selected, selected_features)
            
            # Handle single model or list of models
            model_types = [model_type] if isinstance(model_type, str) else model_type
            
            for current_model_type in model_types:
                logger.info(f"Training model: {current_model_type}")
                
                # Create model with hyperparameter grid
                model = self._get_estimator(current_model_type)
                param_grid = self._get_param_grid(current_model_type)
                
                # Create full pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                
                # Update parameter grid to work with EstimatorWrapper
                current_param_grid = {f'model__{k}': v for k, v in param_grid.items()}
                
                # Set up grid search
                scoring = make_scorer(mean_squared_error, greater_is_better=False)
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid=current_param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )
                
                # Fit with grid search
                grid_search.fit(X_selected, y)
                
                # Store the trained model
                self.trained_models[current_model_type] = grid_search.best_estimator_
                
                # If this is the first model or better than current best, update best model
                if (self.best_model is None or 
                    grid_search.best_score_ > (self.best_model.best_score_ if hasattr(self.best_model, 'best_score_') else -np.inf)):
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = current_model_type
                    self.best_params = grid_search.best_params_
                
                logger.info(f"Trained {current_model_type} with params: {grid_search.best_params_}")
                
                # Evaluate model
                metrics = self.evaluate_model(current_model_type, X_selected, y, cv)
                logger.info(f"{current_model_type.upper()} evaluation metrics: {metrics}")
            
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
    
    def evaluate_model(self, model_name, X, y, cv=5):
        """
        Evaluate a specific model by name.
        
        Parameters:
        - model_name: str, name of the model to evaluate
        - X: feature matrix
        - y: target variable
        - cv: int, number of cross-validation folds
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained")
            
        model = self.trained_models[model_name]
        
        try:
            metrics = evaluate_regression_performance(
                model,
                X,
                y,
                cv=cv
            )
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name}: {str(e)}")
            raise
            
    def evaluate(self, X, y, cv=5):
        """
        Evaluate the best model using multiple metrics.
        """
        if self.best_model is None:
            raise NotFittedError("No models have been trained yet")
            
        return self.evaluate_model(self.best_model_name, X, y, cv)
    
    def predict(self, X, model_name=None):
        """
        Make predictions using a specific model or the best model.
        
        Parameters:
        - X: feature matrix
        - model_name: str, optional name of the model to use for prediction
                     If None, uses the best model
        """
        if model_name is None:
            if self.best_model is None:
                raise NotFittedError("No models have been trained yet")
            model = self.best_model
        else:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} has not been trained")
            model = self.trained_models[model_name]
            
        try:
            return model.predict(X)
            
        except Exception as e:
            logger.error(f"Error during prediction with model {model_name or 'best'}: {str(e)}")
            raise
    
    def create_ensemble(self, X, y, models=None):
        """
        Create a weighted ensemble model from multiple base models.
        
        Returns:
            WeightedEnsemble: The trained ensemble model
        """
        if models is None:
            models = ['rf', 'gb', 'lr']
            
        try:
            # Get the base models (train them if not already trained)
            base_models = []
            for model_name in models:
                if model_name in self.trained_models:
                    base_models.append(self.trained_models[model_name])
                else:
                    logger.info(f"Training base model: {model_name}")
                    model = self.model_factories[model_name](X, y)
                    self.trained_models[model_name] = model
                    base_models.append(model)
            
            # Create meta-model
            from sklearn.linear_model import BayesianRidge
            meta_model = BayesianRidge()
            
            # Get base model predictions
            base_preds = np.column_stack([m.predict(X) for m in base_models])
            
            # Train meta-model
            meta_model.fit(base_preds, y)
            
            # Create and store ensemble
            self.ensemble_model = WeightedEnsemble(base_models, meta_model=meta_model)
            self.trained_models['ensemble'] = self.ensemble_model
            
            # Evaluate ensemble
            y_pred = self.ensemble_model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            logger.info(f"Ensemble model - MSE: {mse:.4f}, R²: {r2:.4f}")
            
            return self.ensemble_model
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            raise
