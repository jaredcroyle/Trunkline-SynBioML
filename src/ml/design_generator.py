import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def predict_with_uncertainty_rf(rf_model, X):
    all_tree_preds = np.array([tree.predict(X) for tree in rf_model.estimators_])
    mean_pred = np.mean(all_tree_preds, axis=0)
    std_pred = np.std(all_tree_preds, axis=0)
    return mean_pred, std_pred


def predict_with_uncertainty_ensemble(base_models, ensemble_model, X):
    base_means = []
    base_stds = []

    if not base_models:  # If no base models, use meta-model directly
        ensemble_pred = ensemble_model.predict(X)
        ensemble_std = np.zeros_like(ensemble_pred)
        return ensemble_pred, ensemble_std

    for model in base_models:
        if hasattr(model, "estimators_"):  # Random Forest
            mean, std = predict_with_uncertainty_rf(model, X)
        elif hasattr(model, "predict"):
            try:
                mean, std = model.predict(X, return_std=True)
            except TypeError:
                mean = model.predict(X)
                std = np.zeros_like(mean)
        else:
            mean = model.predict(X)
            std = np.zeros_like(mean)

        base_means.append(mean.reshape(-1, 1))
        base_stds.append(std.reshape(-1, 1))

    if not base_means:  # Fallback if no predictions were made
        ensemble_pred = ensemble_model.predict(X)
        ensemble_std = np.zeros_like(ensemble_pred)
        return ensemble_pred, ensemble_std

    base_means = np.hstack(base_means)
    base_stds = np.hstack(base_stds)

    ensemble_pred = ensemble_model.predict(base_means)
    weights = ensemble_model.coef_.reshape(-1, 1)
    ensemble_var = np.sum((weights ** 2) * (base_stds ** 2), axis=1)
    ensemble_std = np.sqrt(ensemble_var)

    return ensemble_pred, ensemble_std


# === MAIN DESIGN GENERATION PIPELINE ===

def run_design_generation(
    feature_names,
    existing_df,
    model,
    scaler,
    output_path,
    PI,
    PI_email,
    top_n=96,
    include_wild_type=True,
    modif_code_for_NoMod=1,
    is_ensemble=False,
    bounds_file=None,
    verbose=False
):
    # Generate design space using the same features as preprocessing
    full_space = pd.DataFrame(
        np.random.uniform(0, 1, (1000, len(feature_names))),
        columns=feature_names
    )
    if verbose:
        print(f"Generated full design space: {len(full_space)} designs with {len(feature_names)} features")

    remaining_space = filter_existing_designs(full_space, existing_df, feature_names)
    if verbose:
        print(f"Designs after filtering existing: {len(remaining_space)}")

    if bounds_file:
        bounds = load_bounds(bounds_file)
        remaining_space = filter_designs_by_bounds(remaining_space, bounds)
        if verbose:
            print(f"Designs after applying bounds: {len(remaining_space)}")

    X_scaled = scaler.transform(remaining_space)
    stds = None

    if is_ensemble:
        predictions, stds = predict_with_uncertainty_ensemble(
            model["base_models"], model["ensemble_model"], X_scaled
        )
        ranking_scores = predictions - 1.96 * stds  # Lower Confidence Bound
        if verbose:
            print("Used ensemble model with uncertainty (LCB).")
    else:
        try:
            predictions, stds = model.predict(X_scaled, return_std=True)
            ranking_scores = predictions + 1.96 * stds  # Upper Confidence Bound
            if verbose:
                print("Used model with uncertainty (UCB).")
        except TypeError:
            predictions = model.predict(X_scaled)
            ranking_scores = predictions
            if verbose:
                print("Used model without uncertainty.")

    remaining_space = remaining_space.copy()
    remaining_space["predicted_output"] = predictions
    if stds is not None:
        remaining_space["predicted_std"] = stds
    remaining_space["ranking_score"] = ranking_scores

    # Sort and select top N
    ranked_space = remaining_space.sort_values(by="ranking_score", ascending=False)
    ranked_space.insert(0, "Line Name", [f"Strain {i + 1}" for i in range(len(ranked_space))])
    top_designs = ranked_space.head(top_n).copy()

    # Add wild type if requested
    if include_wild_type:
        top_designs = add_wild_type(top_designs, feature_names, modif_code=modif_code_for_NoMod)
        ranked_space = add_wild_type(ranked_space, feature_names, modif_code=modif_code_for_NoMod)
        if verbose:
            print(f"Added wild type design")

    # Save ICE-compatible top designs
    save_ice_csv(top_designs, feature_names, output_path, PI, PI_email)

    # Optional: save full predictions
    full_output_path = output_path.replace(".csv", "_FULL.csv")
    save_ice_csv(ranked_space, feature_names, full_output_path, PI, PI_email)
    if verbose:
        print(f"Also saved full ranked designs to {full_output_path}")

    # Save static and interactive plots
    plot_path = output_path.replace(".csv", "_PLOT.png")
    plot_top_design_predictions(top_designs, output_path=plot_path)

    interactive_path = output_path.replace(".csv", "_PLOT.html")
    plot_top_design_predictions_interactive(top_designs, output_html=interactive_path)

    return top_designs


# === HELPER FUNCTIONS ===

def generate_design_space(feature_names, n_samples=1000):
    """
    Generate a design space with random values for each feature.
    
    Parameters:
    - feature_names: list of feature names
    - n_samples: number of samples to generate
    
    Returns:
    - pd.DataFrame with random values for each feature
    """
    return pd.DataFrame(
        np.random.uniform(0, 1, (n_samples, len(feature_names))),
        columns=feature_names
    )


def filter_existing_designs(full_design_df, existing_df, feature_names):
    # If existing_df is empty, return all designs
    if existing_df.empty:
        return full_design_df
    
    # Get common columns between full_design_df and existing_df
    common_columns = list(set(full_design_df.columns) & set(existing_df.columns))
    
    # If no common columns found, return all designs
    if not common_columns:
        return full_design_df
    
    # Filter designs that don't match existing ones
    merged = pd.merge(full_design_df, existing_df[common_columns], on=common_columns, how='left', indicator=True)
    filtered_df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    return filtered_df.reset_index(drop=True)


def load_bounds(bounds_file):
    df = pd.read_csv(bounds_file)
    return {row['gene']: (row['min'], row['max']) for _, row in df.iterrows()}


def filter_designs_by_bounds(design_df, bounds):
    def in_bounds(row):
        for gene, (min_val, max_val) in bounds.items():
            if gene in row and not (min_val <= row[gene] <= max_val):
                return False
        return True
    return design_df[design_df.apply(in_bounds, axis=1)]


def add_wild_type(designs_df, feature_names, modif_code=1):
    """
    Add a wild type design with all features set to NoMod.
    
    Parameters:
    - designs_df: DataFrame of designs
    - feature_names: list of feature names
    - modif_code: code for NoMod modification
    
    Returns:
    - DataFrame with wild type added
    """
    wild_type = pd.DataFrame({feature: [modif_code] for feature in feature_names})
    wild_type.insert(0, "Line Name", ["Wild Type"])
    return pd.concat([wild_type, designs_df], ignore_index=True)


def save_ice_csv(designs_df, feature_names, output_path, PI, PI_email):
    header_info = f"# PI: {PI}\n# PI Email: {PI_email}\n"

    base_cols = ["Line Name"] + feature_names
    extra_cols = []
    for col in ["predicted_output", "predicted_std", "ranking_score"]:
        if col in designs_df.columns:
            extra_cols.append(col)

    all_cols = base_cols + extra_cols
    designs_df[all_cols].to_csv(output_path, index=False, header=True)
    with open(output_path, 'r') as f:
        content = f.read()
    with open(output_path, 'w') as f:
        f.write(header_info + content)

    print(f"Saved ICE-compatible CSV to {output_path} with {len(designs_df)} rows.")


def plot_top_design_predictions(designs_df, output_path="plots/top_design_predictions.png"):
    if "predicted_output" not in designs_df.columns:
        print("No predictions found — skipping design plot.")
        return

    plot_df = designs_df.copy()
    plot_df = plot_df.sort_values(by="ranking_score", ascending=False).head(20)
    plot_df = plot_df[::-1]

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_df,
        y="Line Name",
        x="predicted_output",
        xerr=plot_df["predicted_std"] if "predicted_std" in plot_df.columns else None,
        palette="viridis"
    )

    ax.set_xlabel("Predicted Output")
    ax.set_ylabel("Strain")
    ax.set_title("Top Predicted Designs")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved top design prediction plot to {output_path}")


def plot_top_design_predictions_interactive(designs_df, output_html="report/top_design_predictions.html"):
    if "predicted_output" not in designs_df.columns:
        print("ERROR: No predictions found — skipping interactive plot.")
        return

    plot_df = designs_df.copy()
    plot_df = plot_df.sort_values(by="ranking_score", ascending=False).head(20)
    plot_df = plot_df[::-1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=plot_df["predicted_output"],
        y=plot_df["Line Name"],
        orientation='h',
        error_x=dict(
            type='data',
            array=plot_df["predicted_std"] if "predicted_std" in plot_df.columns else None,
            visible=True
        ),
        marker=dict(color='mediumseagreen'),
        name="Predicted Output"
    ))

    fig.update_layout(
        title="Top 20 Predicted Designs (Interactive)",
        xaxis_title="Predicted Output",
        yaxis_title="Strain",
        yaxis=dict(tickmode='linear'),
        template="plotly_white",
        height=600
    )

    fig.write_html(output_html, include_plotlyjs='cdn')
    print(f"Interactive plot saved to {output_html}")