from jinja2 import Environment, FileSystemLoader
import os
from typing import Dict, Any

def generate_report(
    models_results: Dict[str, Any], 
    output_path: str = "report/report.html", 
    template_folder: str = "./templates",
    template_file: str = "model_report_template.html",
    ensemble_plot_html: str = None  # <- NEW ARGUMENT
) -> None:
    """
    Generate HTML report from models_results dictionary using Jinja2 template.

    Args:
        models_results (dict): Dictionary containing model names as keys and their results as values.
        output_path (str): Path to save the rendered HTML report.
        template_folder (str): Path to the folder containing Jinja2 templates.
        template_file (str): Name of the template file to use.
        ensemble_plot_html (str): Path to the interactive ensemble HTML plot to embed.
    """

    if not models_results:
        raise ValueError("models_results dictionary is empty. Nothing to generate report.")

    abs_template_folder = os.path.abspath(template_folder)
    env = Environment(loader=FileSystemLoader(searchpath=abs_template_folder))
    template = env.get_template(template_file)

    html_out = template.render(
        models=models_results,
        ensemble_plot_path=ensemble_plot_html  # <- Pass to template
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"Report saved to {output_path}")