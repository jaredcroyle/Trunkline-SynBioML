import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from .ensemble import WeightedEnsemble

# Set a consistent style for all plots
sns.set_theme(style="whitegrid")


def predict_test_points_generic(model, X_test, model_name="Model"):
    """
    Predict and print results for given test inputs using the model.
    Supports uncertainty if model provides it (e.g., GP).
    """
    try:
        y_pred, y_std = model.predict(X_test, return_std=True)
    except TypeError:
        y_pred = model.predict(X_test)
        y_std = None

    for i, mean in enumerate(y_pred):
        if y_std is not None:
            print(f"{model_name} - Test point {i+1}: Mean={mean:.2f}, Std={y_std[i]:.2f}")
        else:
            print(f"{model_name} - Test point {i+1}: Predicted={mean:.2f}")


def plot_initial_data(X, y, save_path=None):
    """
    Plot initial 2D data colored by target variable.
    Compatible with both DataFrame and NumPy input.
    """
    if hasattr(X, "iloc"):
        x0, x1 = X.iloc[:, 0], X.iloc[:, 1]
    else:
        x0, x1 = X[:, 0], X[:, 1]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x0, x1, c=y, cmap='viridis', edgecolor='k', alpha=0.85)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Initial Data Distribution Colored by Target', fontsize=15, weight='bold')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Target Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_surface_predictions_generic(model, df_clean, X, model_name="Model", save_path=None):
    """
    Visualize surface predictions and uncertainty across 2 features (Starting OD, Line Name).
    """
    od_vals = np.linspace(df_clean["Starting OD"].min(), df_clean["Starting OD"].max(), 50)
    line_vals = np.linspace(df_clean["Line Name_code"].min(), df_clean["Line Name_code"].max(), 50)
    X1, X2 = np.meshgrid(od_vals, line_vals)

    X_grid = np.zeros((50 * 50, X.shape[1]))
    X_grid[:, 0] = X1.ravel()
    X_grid[:, 1] = X2.ravel()

    try:
        y_mean, y_std = model.predict(X_grid, return_std=True)
    except TypeError:
        y_mean = model.predict(X_grid)
        y_std = None

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    cp = plt.contourf(X1, X2, y_mean.reshape(50, 50), cmap='viridis')
    plt.colorbar(cp, label='Predicted Mean')
    plt.xlabel('Starting OD', fontsize=12)
    plt.ylabel('Line Name Code', fontsize=12)
    plt.title(f'{model_name} Mean Prediction Surface', fontsize=14, weight='bold')

    if y_std is not None:
        plt.subplot(1, 2, 2)
        cp2 = plt.contourf(X1, X2, y_std.reshape(50, 50), cmap='inferno')
        plt.colorbar(cp2, label='Prediction Std. Dev.')
        plt.xlabel('Starting OD', fontsize=12)
        plt.ylabel('Line Name Code', fontsize=12)
        plt.title(f'{model_name} Prediction Uncertainty Surface', fontsize=14, weight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_feature_importance_rf(model, feature_names, model_name="Random Forest", save_path=None):
    """
    Plot ranked feature importances from a Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.title(f"{model_name} Feature Importances", fontsize=16, weight='bold')
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_partial_dependence_rf(model, X, feature_names, model_name="Random Forest", save_path=None):
    """
    Plot partial dependence plots for all features in the model.
    Shows average effect of each feature on prediction.
    """
    display = PartialDependenceDisplay.from_estimator(
        model, X, features=range(X.shape[1]),
        feature_names=feature_names,
        grid_resolution=50, kind='average'
    )
    display.figure_.suptitle(f"{model_name} Partial Dependence Plots", fontsize=16, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_predicted_vs_true(model, X, y, model_name="Model", save_path=None):
    """
    Scatter plot comparing predicted values to true values.
    """
    y_pred = model.predict(X)

    plt.figure(figsize=(8, 8))
    plt.scatter(y, y_pred, alpha=0.75, edgecolor='k', s=60)
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=2)
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.title(f"{model_name} - Predicted vs True", fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_residuals(model, X, y, model_name="Model", save_path=None):
    """
    Plot residuals (true - predicted) against predictions.
    Helps diagnose bias and variance in model predictions.
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals, edgecolor='k', s=60, alpha=0.75)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel("Predicted Values", fontsize=14)
    plt.ylabel("Residuals (True - Predicted)", fontsize=14)
    plt.title(f"{model_name} - Residual Plot", fontsize=16, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_learning_curve(model, X, y, model_name="Model", cv=5, save_path=None):
    """
    Plot learning curve: training vs cross-validation error as dataset size increases.
    Handles both scikit-learn models and WeightedEnsemble.
    """
    if isinstance(model, WeightedEnsemble):
        # For WeightedEnsemble, use direct evaluation
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_errors = []
        val_errors = []
        
        for size in train_sizes:
            train_size = int(size * len(X))
            # Ensure we have at least one sample for training
            if train_size == 0:
                train_size = 1
            # Ensure we have at least one sample for validation
            val_size = len(X) - train_size
            if val_size == 0:
                val_size = 1
                train_size = len(X) - 1
            
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            
            # Calculate errors
            train_error = mean_squared_error(y_train, y_train_pred)
            val_error = mean_squared_error(y_val, y_val_pred)
            
            train_errors.append(train_error)
            val_errors.append(val_error)
        
        train_errors = np.array(train_errors)
        val_errors = np.array(val_errors)
    else:
        # For scikit-learn models, use cross-validation
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring='neg_mean_squared_error',
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )
        train_errors = -train_scores.mean(axis=1)
        val_errors = -test_scores.mean(axis=1)

    # Plot the learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, 'o-', label='Training Error', color='navy', linewidth=2, markersize=6)
    plt.plot(train_sizes, val_errors, 's--', label='Validation Error', color='orange', linewidth=2, markersize=6)
    plt.xlabel("Training Set Size", fontsize=12)
    plt.ylabel("Mean Squared Error", fontsize=12)
    plt.title(f"Learning Curve for {model_name}", fontsize=16, weight='bold')
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


def plot_predicted_vs_true_with_error(model, X, y_true, model_name="Model", is_gp=False, save_path=None):
    """
    Plot predicted vs true with uncertainty/error bars if available.

    Supports Gaussian Processes and Random Forests with uncertainty.
    """
    if is_gp:
        y_pred, y_std = model.predict(X, return_std=True)
    elif hasattr(model, "estimators_"):  # For Random Forest
        all_tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        y_pred = np.mean(all_tree_preds, axis=0)
        y_std = np.std(all_tree_preds, axis=0)
    else:
        y_pred = model.predict(X)
        y_std = None

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolor='k', s=60, label="Predictions")
    if y_std is not None:
        plt.errorbar(y_true, y_pred, yerr=y_std, fmt='o', alpha=0.3, label="Uncertainty", color='gray')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=2, label="Ideal Fit")
    plt.xlabel("True Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.title(f"{model_name} - Predicted vs True (with Error Bars)", fontsize=16, weight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()