from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64
from datetime import datetime

def generate_report(
    models_results: Dict[str, Any], 
    output_path: str = "reports/report.html", 
    template_folder: str = "./templates",
    template_file: str = "model_report_template.html",
    ensemble_plot_path: Optional[str] = None,
    timestamp: Optional[str] = None
) -> str:
    """
    Generate HTML report from models_results dictionary using Jinja2 template.

    Args:
        models_results (dict): Dictionary containing model names as keys and their results as values.
        output_path (str): Path to save the rendered HTML report.
        template_folder (str): Path to the folder containing Jinja2 templates.
        template_file (str): Name of the template file to use.
        ensemble_plot_path (str, optional): Path to the ensemble plot image to include.
        timestamp (str, optional): Timestamp for the report. Defaults to current time.

    Returns:
        str: Path to the generated report
    """
    if not models_results:
        raise ValueError("models_results dictionary is empty. Nothing to generate report.")

    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Set up template environment
    abs_template_folder = Path(template_folder).absolute()
    env = Environment(
        loader=FileSystemLoader(searchpath=str(abs_template_folder)),
        autoescape=select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters
    def format_float(value, precision=4):
        """Format a float to a string with specified precision."""
        if isinstance(value, (int, float)):
            return f"{value:.{precision}f}"
        return str(value)
    
    env.filters['format_float'] = format_float
    
    # Load the template
    template = env.get_template(template_file)
    
    # Prepare the context
    context = {
        'models': models_results,
        'ensemble_plot_path': ensemble_plot_path,
        'timestamp': timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'title': 'Trunkline ML Model Analysis Report'
    }
    
    # Render the template
    html_out = template.render(**context)
    
    # Save the report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_out)
    
    print(f"[SUCCESS] Report generated at: {output_path}")
    return str(output_path)


def generate_model_report(
    model_name: str,
    model_description: str,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    plots: List[Dict[str, str]],
    output_dir: str = "reports",
    template_folder: str = "./templates",
    template_file: str = "model_report_template.html"
) -> str:
    """
    Generate a report for a single model.
    
    Args:
        model_name: Name of the model
        model_description: Description of the model
        metrics: Dictionary of metrics (mse, rmse, mae, r2)
        params: Dictionary of model parameters
        plots: List of plot dictionaries with 'title', 'path', and 'caption' keys
        output_dir: Directory to save the report
        template_folder: Path to the folder containing Jinja2 templates
        template_file: Name of the template file to use
        
    Returns:
        str: Path to the generated report
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare the models results dictionary
    models_results = {
        model_name: {
            'description': model_description,
            **metrics,
            'params': params,
            'plots': plots
        }
    }
    
    # Generate a timestamp for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{model_name.lower().replace(' ', '_')}_report_{timestamp}.html"
    
    # Generate the report
    return generate_report(
        models_results=models_results,
        output_path=str(output_path),
        template_folder=template_folder,
        template_file=template_file,
        timestamp=timestamp
    )