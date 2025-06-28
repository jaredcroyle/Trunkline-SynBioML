import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(filepath):
    """
    Loads and preprocesses raw CSV data for modeling.

    Parameters:
    - filepath: str, path to the CSV data file

    Returns:
    - df_clean: pd.DataFrame, cleaned and encoded dataset
    """
    # Load CSV/file into DataFrame
    df = pd.read_csv(filepath)

    # Drop rows with missing values in critical columns
    df_clean = df.dropna(subset=["Starting OD", "Line Name", "24"]).copy()

    # Convert selected categorical columns to string type and encode them
    for col in ["Line Name", "Media", "Protocol Name", "Type"]:
        df_clean[col] = df_clean[col].astype(str)
        df_clean[f"{col}_code"] = LabelEncoder().fit_transform(df_clean[col])

    return df_clean

def prepare_features(df_clean):
    """
    Extracts features and target values from cleaned dataset.

    Parameters:
    - df_clean: pd.DataFrame, cleaned and encoded data

    Returns:
    - X: np.ndarray, feature matrix
    - y: np.ndarray, target variable
    - feature_cols: list of str, names of features used
    """
    # Columns used as features (placeholder - EDIT FOR FEATURE ENGINEERING ON CUSTOM)
    feature_cols = [
        "Starting OD",
        "Line Name_code",
        "Media_code",
        "Protocol Name_code",
        "Type_code"
    ]

    # Extract feature matrix (X) and target values (y)
    X = df_clean[feature_cols].values
    y = df_clean["24"].values  # '24' is assumed to be the target column

    return X, y, feature_cols