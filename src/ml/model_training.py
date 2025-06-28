import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .ensemble import WeightedEnsemble
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Individual Model Trainers ===

def train_gaussian_process(X, y):
    """
    Trains a Gaussian Process Regressor with RBF kernel.

    Returns:
    - Fitted GaussianProcessRegressor
    """
    kernel = C(1.0, (1e-3, 1e3)) * RBF([1.0]*X.shape[1], (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1.0, n_restarts_optimizer=10, normalize_y=True)
    gp.fit(X, y)
    return gp

def train_random_forest(X, y, n_estimators=100, random_state=42):
    """
    Trains a Random Forest Regressor.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    return rf

def train_gradient_boosting(X, y, random_state=42):
    """
    Trains a Gradient Boosting Regressor.
    """
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=random_state
    )
    gbr.fit(X, y)
    return gbr

def train_linear_regression(X, y):
    """
    Trains a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def train_ridge(X, y, alpha=1.0):
    """
    Trains a Ridge Regression model.
    """
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

def train_svr(X, y):
    """
    Trains a Support Vector Regressor with RBF kernel.
    """
    model = SVR(kernel='rbf')
    model.fit(X, y)
    return model

def train_mlp(X, y):
    """
    Trains a Multi-layer Perceptron (Neural Network) Regressor.
    """
    model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y)
    return model

def train_knn(X, y, n_neighbors=5):
    """
    Trains a K-Nearest Neighbors Regressor.
    """
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X, y)
    return model

# === Utility Functions ===

def cross_validate_model(model, X, y, cv=5, scoring="neg_mean_squared_error"):
    """
    Perform k-fold cross-validation and print mean and std of the chosen score.

    Returns:
    - Array of cross-validation scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"\nCross-Validation Scores (MSE): {-scores}")
    print(f"Mean MSE: {-scores.mean():.4f}, Std: {scores.std():.4f}")
    return scores

def scale_features(X):
    """
    Standardizes features to zero mean and unit variance.

    Returns:
    - X_scaled: scaled feature matrix
    - scaler: fitted StandardScaler object
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def add_polynomial_features(X, degree=2):
    """
    Adds polynomial features of a given degree.

    Returns:
    - X_poly: transformed features
    - poly: fitted PolynomialFeatures transformer
    """
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    return X_poly, poly

def evaluate_regression_performance(model, X, y, cv=5):
    """
    Evaluates model performance using cross-validation.

    Returns:
    - mse, rmse, mae, r2
    """
    if isinstance(model, WeightedEnsemble):
        # For WeightedEnsemble, use direct prediction since it's not scikit-learn compatible
        y_pred = model.predict(X)
    else:
        # For scikit-learn models, use cross-validation
        y_pred = cross_val_predict(model, X, y, cv=cv)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, rmse, mae, r2