# Trunkline ML - User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Running the Pipeline](#running-the-pipeline)
6. [Generating Reports](#generating-reports)
7. [Troubleshooting](#troubleshooting)
8. [FAQ](#faq)

## Introduction

Trunkline ML is a machine learning pipeline for predictive modeling with a focus on interpretability and ease of use. It includes multiple model types, ensemble methods, and comprehensive reporting capabilities.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git (for version control)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TrunklineML.git
   cd TrunklineML/Trunkline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

1. Prepare your data in CSV format and place it in the `data/` directory
2. Update the configuration file (`config/default.yaml`) with your data paths and parameters
3. Run the main pipeline:
   ```bash
   python main.py --config config/default.yaml
   ```
4. View the generated reports in the `report/` directory

## Configuration

The pipeline is configured using YAML files. The main configuration options are:

```yaml
data:
  input_path: "data/your_data.csv"
  output_dir: "results/"
  test_size: 0.2
  random_state: 42

models:
  - name: "Random Forest"
    type: "random_forest"
    params:
      n_estimators: 100
      max_depth: 10
  - name: "Gradient Boosting"
    type: "gradient_boosting"
    params:
      n_estimators: 100
      learning_rate: 0.1

reporting:
  output_dir: "report/"
  generate_plots: true
  generate_shap: true
```

## Running the Pipeline

### Full Pipeline
```bash
python main.py --config config/default.yaml
```

### Specific Components

1. **Data Preprocessing**
   ```python
   from src.data_preprocessing import load_and_clean_data
   
   df = load_and_clean_data("data/your_data.csv")
   ```

2. **Model Training**
   ```python
   from src.model_training import train_model
   
   model = train_model(X_train, y_train, model_type="random_forest")
   ```

3. **Generating Reports**
   ```python
   from src.report_generator import generate_report
   
   generate_report(model, X_test, y_test, output_dir="report/")
   ```

## Generating Reports

The pipeline generates several types of reports:

1. **HTML Report**: Comprehensive model analysis in `report/report.html`
2. **Performance Metrics**: CSV files with model metrics
3. **Visualizations**: Plots in the `plots/` directory

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **File Not Found**
   - Verify file paths in the configuration
   - Ensure the data directory exists

3. **Memory Issues**
   - Reduce dataset size
   - Use fewer features
   - Increase system memory

## FAQ

**Q: How do I add a new model?**
A: Add the model class to `src/models/` and update the model factory in `src/model_training.py`

**Q: How do I customize the report?**
A: Modify the templates in `src/reporting/templates/`

**Q: How do I handle large datasets?**
A: Use the `chunk_size` parameter in the data loading functions

## Support

For additional help, please open an issue on the GitHub repository.