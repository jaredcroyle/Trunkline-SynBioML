"""Configuration loader for Trunkline ML Pipeline."""
import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, loads default config.
        
    Returns:
        dict: Configuration dictionary
    """
    if config_path is None:
        # Use default config path
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "default.yaml"
        )
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Set up paths relative to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Make paths absolute
        def make_absolute(path):
            if path and not os.path.isabs(path):
                return os.path.join(project_root, path)
            return path
        
        # Update paths in the config
        config['data']['input_file'] = make_absolute(config['data']['input_file'])
        config['data']['output_dir'] = make_absolute(config['data']['output_dir'])
        config['model']['model_dir'] = make_absolute(config['model']['model_dir'])
        config['visualization']['plots_dir'] = make_absolute(config['visualization']['plots_dir'])
        config['report']['output_dir'] = make_absolute(config['report']['output_dir'])
        config['design_generation']['output_path'] = make_absolute(config['design_generation']['output_path'])
        
        # Create necessary directories
        os.makedirs(config['data']['output_dir'], exist_ok=True)
        os.makedirs(config['model']['model_dir'], exist_ok=True)
        os.makedirs(config['visualization']['plots_dir'], exist_ok=True)
        os.makedirs(config['report']['output_dir'], exist_ok=True)
        
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = [
        'data', 'model', 'feature_selection', 'evaluation',
        'visualization', 'report', 'logging', 'design_generation'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section in config: {section}")
    
    # Check required fields in each section
    required_fields = {
        'data': ['input_file', 'output_dir'],
        'model': ['type', 'build_model', 'save_model', 'model_dir'],
        'feature_selection': ['use_shap', 'num_features'],
        'evaluation': ['cv_folds', 'metrics'],
        'visualization': ['plots_dir', 'generate_plots', 'plot_types'],
        'report': ['generate', 'output_dir', 'template_folder', 'template_file'],
        'logging': ['level', 'file'],
        'design_generation': ['output_path', 'pi_name', 'pi_email', 'top_n', 'include_wild_type', 'modif_code_for_NoMod']
    }
    
    for section, fields in required_fields.items():
        for field in fields:
            if field not in config[section]:
                raise ValueError(f"Missing required field in config[{section}]: {field}")
    
    # Check if input file exists
    if not os.path.exists(config['data']['input_file']):
        raise FileNotFoundError(f"Input file not found: {config['data']['input_file']}")
    
    return True
