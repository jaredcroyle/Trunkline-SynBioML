import os
from pathlib import Path
from typing import Dict, Any
import yaml

class Config:
    def __init__(self):
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or default values."""
        try:
            config_path = Path(os.getenv('TRUNKLINE_CONFIG', 'config.yaml'))
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            return self._default_config()
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            'model': {
                'type': 'rf',
                'num_features': 10,
                'build_new': True
            },
            'data': {
                'input_file': 'data/Limonene_data.csv',
                'output_dir': 'output/',
                'model_dir': 'saved_models/',
                'plots_dir': 'plots/',
                'report_dir': 'report/'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/monatalks.log'
            },
            'ensemble': {
                'use_ensemble': True,
                'models': ['rf', 'gp', 'lr']
            },
            'feature_selection': {
                'method': 'shap',
                'num_features': 10
            },
            'hardware': {
                'use_gpu': False,
                'cpu_cores': 'auto',  # Can be 'auto' or a specific number
                'gpu_devices': [],    # List of GPU devices to use
                'memory_limit': None, # Memory limit in GB
                'batch_size': 32,     # Batch size for processing
                'parallel_jobs': -1   # Number of parallel jobs (-1 means all available cores)
            }
        }

    def get(self, key: str, default=None) -> Any:
        """Get a configuration value with optional default."""
        try:
            keys = key.split('.')
            value = self._config
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

    def update(self, key: str, value: Any) -> None:
        """Update a configuration value."""
        keys = key.split('.')
        current = self._config
        for k in keys[:-1]:
            current = current.setdefault(k, {})
        current[keys[-1]] = value

def get_config() -> Config:
    """Get the global configuration instance."""
    if not hasattr(get_config, "_instance"):
        get_config._instance = Config()
    return get_config._instance

# Initialize config on module import
config = get_config()