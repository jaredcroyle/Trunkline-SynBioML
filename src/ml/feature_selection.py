import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def shap_feature_selection(X, y, feature_names, num_features=10, model=None, random_state=42):
    """
    Select top N most important features using SHAP values from a tree-based model.

    Parameters:
    - X: np.ndarray or pd.DataFrame, input features
    - y: np.ndarray or pd.Series, target values
    - feature_names: list of str, feature names in order
    - num_features: int, number of top features to select
    - model: optional pre-trained model; if None, a RandomForestRegressor is used
    - random_state: for reproducibility if training a new model

    Returns:
    - selected_features: list of selected feature names (top SHAP-ranked)
    - mask: boolean array, True for selected features
    """

    # Train default model if none provided
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X, y)

    # Create SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values and disable additivity check
    shap_values = explainer.shap_values(X, check_additivity=False)

    # Compute average magnitude of SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Rank features by SHAP importance
    top_indices = np.argsort(mean_abs_shap)[::-1][:num_features]
    selected_features = [feature_names[i] for i in top_indices]

    # Create mask for selected features (useful for slicing arrays)
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[top_indices] = True

    return selected_features, mask