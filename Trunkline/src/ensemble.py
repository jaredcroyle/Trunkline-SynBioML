import numpy as np

class WeightedEnsemble:
    def __init__(self, base_models=None, weights=None, meta_model=None):
        """
        base_models: list of trained models (optional)
        weights: array-like, weights for each base model (default equal weights)
        meta_model: ensemble meta-model (e.g., BayesianRidge)
        """
        self.base_models = base_models if base_models is not None else []
        if weights is None:
            self.weights = np.ones(len(self.base_models)) / len(self.base_models) if self.base_models else np.array([1.0])
        else:
            if len(weights) != len(self.base_models):
                raise ValueError("Weights length must match number of base models")
            self.weights = np.array(weights)
        self.meta_model = meta_model

    def predict(self, X):
        """Weighted mean prediction from base models or meta-model."""
        if self.base_models:
            preds = np.column_stack([m.predict(X) for m in self.base_models])
            weighted_mean = np.dot(preds, self.weights)
            return weighted_mean
        else:
            # If no base models, use meta-model directly
            return self.meta_model.predict(X)

    def predict_with_uncertainty(self, X):
        """
        Returns weighted mean and std deviation of base models predictions.
        Treats std dev as uncertainty estimate.
        """
        preds = np.column_stack([m.predict(X) for m in self.base_models])
        weighted_mean = np.dot(preds, self.weights)
        weighted_var = np.dot(self.weights, (preds - weighted_mean[:, None]) ** 2)
        weighted_std = np.sqrt(weighted_var)
        return weighted_mean, weighted_std

    def predict_meta(self, X):
        """
        Uses meta-model to predict based on base model predictions.
        """
        base_preds = np.column_stack([m.predict(X) for m in self.base_models])
        if self.meta_model is None:
            raise ValueError("Meta model not set")
        ensemble_pred = self.meta_model.predict(base_preds)
        return ensemble_pred

    def fit(self, X, y):
        """
        Fits both base models and meta-model if available.
        """
        if self.base_models:
            # Train base models
            for model in self.base_models:
                model.fit(X, y)
            
            # Train meta-model if available
            if self.meta_model is not None:
                # Get predictions from base models
                base_preds = np.column_stack([m.predict(X) for m in self.base_models])
                self.meta_model.fit(base_preds, y)