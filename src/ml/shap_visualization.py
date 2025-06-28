import os
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def shap_explainability(model, X, feature_names, model_name=None):
    """
    Generate SHAP explainability plots for a given model and input features.

    Parameters:
    - model: trained ML model (tree-based or otherwise)
    - X: input features (NumPy array or DataFrame)
    - feature_names: list of feature names
    - model_name: optional string to tag plots
    """

    # Convert X to DataFrame if needed
    X_df = pd.DataFrame(X, columns=feature_names)

    print("\nFeature columns for SHAP explainability:")
    print(X_df.columns.tolist())

    # Sample a background set (required for SHAP) from the dataset
    background = X_df.sample(min(100, len(X_df)), random_state=42)

    # --- Use different explainers based on model type ---
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        # TreeExplainer is fast and accurate for tree-based models
        explainer = shap.TreeExplainer(model, data=background, check_additivity=False)
        shap_values = explainer.shap_values(X_df.iloc[:10])
    else:
        # KernelExplainer is a generic (slower) fallback for non-tree models
        def model_predict(data):
            return model.predict(data)

        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_df.iloc[:10], nsamples=100)

    # Create output directory
    os.makedirs("report", exist_ok=True)

    # === SHAP Summary Plot ===
    plt.figure()
    shap.summary_plot(shap_values, X_df.iloc[:10], feature_names=feature_names, show=False)
    summary_path = f"report/shap_summary_plot.png" if model_name is None else f"report/shap_summary_plot_{model_name.replace(' ', '_')}.png"
    plt.savefig(summary_path, bbox_inches='tight')
    print(f"Saved SHAP summary plot to: {summary_path}")

    # === SHAP Bar Plot (average impact) ===
    plt.figure()
    shap.summary_plot(shap_values, X_df.iloc[:10], feature_names=feature_names, plot_type="bar", show=False)
    bar_path = f"report/shap_bar_plot.png" if model_name is None else f"report/shap_bar_plot_{model_name.replace(' ', '_')}.png"
    plt.savefig(bar_path, bbox_inches='tight')
    print(f"Saved SHAP bar plot to: {bar_path}")