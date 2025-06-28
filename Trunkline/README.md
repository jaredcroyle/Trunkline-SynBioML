# Trunkline: ML Pipeline for Biological Data Analysis

A comprehensive machine learning pipeline designed for biological data analysis, featuring automated data preprocessing, model training, evaluation, and visualization.

## Features

- **Data Preprocessing**: Automated cleaning and preparation of biological data
- **Feature Selection**: SHAP-based feature importance analysis
- **Model Training**: Supports multiple regression models:
  - **Random Forest**: Ensemble of decision trees for robust predictions
  - **Gradient Boosting**: Sequential building of decision trees to minimize errors
  - **Support Vector Regressor (SVR)**: Effective for high-dimensional spaces
  - **Multi-layer Perceptron (MLP)**: Neural network implementation
  - **Linear Regression**: Basic linear model for continuous predictions
  - **Ridge Regression**: Linear regression with L2 regularization
  - **K-Nearest Neighbors (KNN)**: Instance-based learning
  - **Gaussian Process**: Probabilistic approach with uncertainty estimates
  - **Weighted Ensemble**: Custom implementation combining multiple models
- **Model Evaluation**: Comprehensive metrics including MSE, RMSE, MAE, and R²
- **Visualization**: Automatic generation of various plots for model analysis
- **Easy to Use**: Simple command-line interface with sensible defaults

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/TrunklineML.git
   cd TrunklineML/Trunkline
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. **Prepare your data**:
   - Place your CSV file in the `data/` directory
   - The default expected file is `data/Limonene_data.csv`

2. **Run the pipeline**:
   ```bash
   python main.py --data_path data/your_data.csv --output_dir results
   ```

## Usage

```bash
python main.py [--data_path DATA_PATH] [--output_dir OUTPUT_DIR] [--model_type {rf,gb,svm}]
               [--use_shap] [--num_features NUM_FEATURES] [--generate_plots]
```

### Arguments

- `--data_path`: Path to input data file (CSV format)
- `--output_dir`: Directory to save output files (default: 'results')
- `--model_type`: Type of model to train (default: 'rf'). Options:
  - `rf`: Random Forest
  - `gb`: Gradient Boosting
  - `svr`: Support Vector Regressor
  - `mlp`: Multi-layer Perceptron
  - `lr`: Linear Regression
  - `ridge`: Ridge Regression
  - `knn`: K-Nearest Neighbors
  - `gp`: Gaussian Process
  - `ensemble`: Weighted Ensemble
- `--use_shap`: Enable SHAP for feature selection (default: True)
- `--num_features`: Number of top features to select (default: 10)
- `--generate_plots`: Generate visualization plots (default: True)

## Output

The pipeline generates the following outputs in the specified output directory:

- `models/`: Trained model files
- `plots/`: Visualization plots
- `logs/`: Log files
- `model_report.txt`: Summary of model performance
- `all_model_predictions.csv`: Predictions from all models
- `model_performance_summary.csv`: Performance metrics for all models

## Example

```bash
# Run with custom parameters
python main.py \
    --data_path data/experiment_data.csv \
    --output_dir results/experiment_1 \
    --model_type gb \
    --num_features 15 \
    --generate_plots
```

## Dependencies

- Python 3.8+
- Core: numpy, pandas, scikit-learn, scipy
- Visualization: matplotlib, seaborn, shap
- Jupyter: jupyter, ipykernel, ipywidgets (for tutorials)
- Utilities: joblib, PyYAML

## Tutorial

The project includes a Jupyter notebook tutorial to help you get started with the ML pipeline:

```bash
# Install Jupyter if you haven't already
pip install jupyter

# Launch Jupyter Notebook
jupyter notebook notebooks/Tutorial.ipynb
```

The tutorial covers:
- Loading and exploring the dataset
- Running the ML pipeline
- Interpreting model results
- Visualizing feature importance
- Making predictions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Key Features

- **Lightweight**: Based on Python 3.9-slim
- **Persistent Storage**: Volumes for data, results, and reports
- **Resource Management**: Configurable CPU and memory limits
- **Development Ready**: Optional Jupyter Lab service

For detailed container usage, see [DOCKER.md](DOCKER.md).

## Output Files

The pipeline generates several output files in the specified output directory:

### Plots Directory (`plots/`)
- `ICE_MOstrains.csv`: Contains the top strain designs with their predicted values
- `ICE_MOstrains_FULL.csv`: Comprehensive dataset of all generated strain designs
- `ICE_MOstrains_PLOT.html`: Interactive visualization of strain predictions
- `ICE_MOstrains_PLOT.png`: Static visualization of strain predictions

### Model Outputs
- `model_rf.joblib`: Trained Random Forest model
- `model_gb.joblib`: Trained Gradient Boosting model
- `model_lr.joblib`: Trained Linear Regression model
- `model_ensemble.joblib`: Trained ensemble model
- `model_performance_summary.csv`: Performance metrics for all models
- `all_model_predictions.csv`: Predictions from all models

### Reports
- `model_report.txt`: Detailed report of model training and evaluation
- `top_design_predictions.html`: Interactive visualization of top designs

## Usage

### Basic Usage

Run the pipeline with default settings:
```bash
python main.py
```

### Command Line Options

```
usage: main.py [-h] [--data_path DATA_PATH] [--output_dir OUTPUT_DIR]
               [--model_type {rf,gb,svm}] [--build_model] [--save_model]
               [--use_shap] [--num_features NUM_FEATURES] [--generate_plots]
               [--plot_types PLOT_TYPES [PLOT_TYPES ...]] [--log_file LOG_FILE]

Trunkline ML Pipeline

options:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to input data file (CSV format)
  --output_dir OUTPUT_DIR
                        Directory to save output files
  --model_type {rf,gb,svm}
                        Type of model to train (rf: Random Forest, gb: Gradient Boosting, svm: Support Vector Machine)
  --build_model         Whether to build a new model
  --save_model          Whether to save the trained model
  --use_shap            Whether to use SHAP for feature selection
  --num_features NUM_FEATURES
                        Number of top features to select
  --generate_plots      Whether to generate visualization plots
  --plot_types PLOT_TYPES [PLOT_TYPES ...]
                        Types of plots to generate
  --log_file LOG_FILE   Path to log file
```

### Example Commands

1. Run with a specific model and feature selection:
   ```bash
   python main.py --model_type gb --num_features 15
   ```

2. Specify custom input and output paths:
   ```bash
   python main.py --data_path data/my_data.csv --output_dir results/experiment1
   ```

3. Disable visualizations:
   ```bash
   python main.py --generate_plots False
   ```

## Output

The pipeline generates the following outputs:

- `results/models/`: Contains trained model files
- `results/plots/`: Contains visualization plots
- `pipeline.log`: Log file with detailed execution information
- `results/model_report.txt`: Summary of model performance

## Plot Types

The following plots are generated by default:

1. **Learning Curve**: Shows model performance vs. training set size
2. **Feature Importance**: Displays the most important features
3. **Residuals**: Plots prediction errors vs. actual values

## Dependencies

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- shap
- joblib

## License

MIT License

Copyright (c) 2025 Jared Croyle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.