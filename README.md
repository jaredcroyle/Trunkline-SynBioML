# Trunkline - Advanced ML Pipeline for Predictive Biological Data Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Trunkline is a comprehensive machine learning pipeline designed for biological data analysis, featuring automated feature selection, model training, and result visualization. The project offers two versions:

- **Full Version**: Complete implementation with all features and customization options
- **Trunkline**: Streamlined version for quick setup and execution

## Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)   
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## Features

- **Automated Feature Selection**
  - SHAP-based feature importance
  - Recursive feature elimination
  - Correlation-based filtering

- **Machine Learning Models**
  - **Random Forest Regressor**: Ensemble of decision trees for robust predictions
  - **Gradient Boosting Regressor**: Sequential building of decision trees to minimize errors
  - **Support Vector Regressor (SVR)**: Effective for high-dimensional spaces with kernel trick
  - **Multi-layer Perceptron (MLP)**: Neural network implementation for regression
  - **Linear Regression**: Basic linear model for continuous predictions
  - **Ridge Regression**: Linear regression with L2 regularization
  - **K-Nearest Neighbors (KNN)**: Instance-based learning using k-nearest neighbors
  - **Gaussian Process Regressor**: Non-parametric, probabilistic approach
  - **Weighted Ensemble**: Custom implementation combining multiple models with optimized weights

- **Model Evaluation**
  - Cross-validation
  - Multiple metrics (R², MSE, MAE)
  - Learning curves
  - Feature importance visualization

- **Data Visualization**
  - Interactive plots
  - Performance metrics visualization
  - SHAP value explanations

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

### Clone the Repository

```bash
git clone https://github.com/jaredcroyle/TrunklineML.git
cd Trunkline
```

### Set Up Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (Trunkline Version)

1. Navigate to the Trunkline directory:
   ```bash
   cd Trunkline
   ```

2. Run the pipeline with default settings:
   ```bash
   python main.py
   ```

3. View the generated reports in the `results/` directory.

## Project Structure

```
TrunklineML/
├── Trunkline/                  # Main package directory
│   ├── config/                 # Configuration files
│   │   ├── default.yaml        # Default configuration
│   │   └── test_config.yaml    # Test configuration
│   │
│   ├── data/                   # Data storage
│   │   ├── input/              # Raw input data
│   │   ├── output/             # Processed data outputs
│   │   └── processed/          # Intermediate processed data
│   │
│   ├── plots/                  # Generated visualization plots
│   │   ├── feature_importance_*.png
│   │   ├── learning_curve_*.png
│   │   ├── partial_dependence_*.png
│   │   └── residuals_*.png
│   │
│   ├── report/                 # Generated reports and visualizations
│   │   ├── report.html
│   │   ├── shap_*.png
│   │   └── predicted_vs_true_*.png
│   │
│   ├── src/                    # Source code
│   │   ├── __init__.py
│   │   ├── config_loader.py    # Configuration management
│   │   ├── data_preprocessing.py
│   │   ├── design_generator.py
│   │   ├── ensemble.py         # Ensemble model implementation
│   │   ├── ensemble_utils.py
│   │   ├── feature_selection.py
│   │   ├── ml_pipeline.py      # Main ML pipeline
│   │   ├── model_evaluation.py
│   │   ├── model_training.py   # Model implementations
│   │   ├── report_generator.py
│   │   ├── shap_visualization.py
│   │   └── test_pipeline.py
│   │
│   ├── .gitignore
│   ├── README.md              # Package documentation
│   ├── USER_GUIDE.md          # User guide
│   ├── main.py                # Entry point
│   └── requirements.txt       # Dependencies
│
├── .gitignore
├── LICENSE
└── README.md                  # Main project documentation
```

## Configuration

Customize the pipeline behavior by editing the configuration file:

```yaml
# biolab/config/config.yaml
data:
  input_file: "data/example_dataset.csv"
  test_size: 0.2
  random_state: 42

model:
  type: "rf"  # rf, gb, svm, etc.
  params:
    n_estimators: 100
    max_depth: 10

feature_selection:
  method: "shap"
  num_features: 20

evaluation:
  cv_folds: 5
  metrics: ["r2", "mse", "mae"]
```

## Usage

### Running the Full Version

```bash
# Run with custom configuration
python biolab/main.py --config biolab/config/config.yaml

# Run with custom data
python biolab/main.py --input data/your_data.csv
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to config file | `config/config.yaml` |
| `--input` | Input data file path | `data/example.csv` |
| `--output` | Output directory | `results/` |
| `--model` | Model type (rf, gb, svm) | `rf` |
| `--cv` | Number of cross-validation folds | `5` |

## Examples

### Basic Usage

```python
from src.ml.pipeline import MLPipeline
import pandas as pd

# Load your data
data = pd.read_csv("data/your_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Initialize and run pipeline
pipeline = MLPipeline()
model = pipeline.fit(X, y)
predictions = model.predict(X_test)
```

### Custom Model Training

```python
from src.ml.model_training import train_random_forest

# Train a custom model
model = train_random_forest(
    X_train, 
    y_train,
    n_estimators=200,
    max_depth=15,
    random_state=42
)
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out:

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

Made with love by Jared Croyle

- **Data Preprocessing**: Handles missing values and categorical variables
- **Feature Selection**: Uses SHAP values for interpretable feature importance
- **Model Training**: Supports multiple algorithms (Random Forest, XGBoost, etc.)
- **Ensemble Methods**: Combines models for improved performance
- **Visualization**: Generates plots and reports

## Customization

The pipeline is designed to be adaptable to different datasets. Key files to modify:

1. **Data Input**: Place your CSV file in the `data/` directory
2. **Configuration**: Adjust parameters in `config/config.yaml`
3. **Feature Engineering**: Modify the preprocessing steps in `src/ml/pipeline.py`

## Full Version

For advanced features and more customization options, use the full version in the `biolab/` directory:

```bash
cd ../biolab
python main.py
```

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

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

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

Jared Croyle - jcroyle@berkeley.edu