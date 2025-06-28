"""
Utility functions for file operations in Trunkline ML.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import json
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory: Path to the directory
        
    Returns:
        Path: Absolute path to the directory
    """
    dir_path = Path(directory).expanduser().resolve()
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_timestamped_dir(base_dir: Union[str, Path], 
                      prefix: str = "", 
                      suffix: str = "",
                      fmt: str = "%Y%m%d_%H%M%S") -> Path:
    """Create a timestamped subdirectory within the base directory.
    
    Args:
        base_dir: Base directory path
        prefix: Optional prefix for the directory name
        suffix: Optional suffix for the directory name
        fmt: Datetime format string
        
    Returns:
        Path: Path to the created directory
    """
    timestamp = datetime.now().strftime(fmt)
    dir_name = f"{prefix}{timestamp}{suffix}".strip("_")
    dir_path = ensure_dir_exists(Path(base_dir) / dir_name)
    logger.debug(f"Created timestamped directory: {dir_path}")
    return dir_path


def save_plot(fig, path: Union[str, Path], dpi: int = 300, 
             bbox_inches: str = 'tight', **kwargs) -> Path:
    """Save a matplotlib figure to a file.
    
    Args:
        fig: Matplotlib figure object
        path: Output file path (can be relative to project root)
        dpi: Dots per inch for the output image
        bbox_inches: Bounding box in inches
        **kwargs: Additional arguments to pass to savefig()
        
    Returns:
        Path: Absolute path to the saved file
    """
    path = Path(path).resolve()
    ensure_dir_exists(path.parent)
    
    # Ensure the file extension is valid
    valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}
    if path.suffix.lower() not in valid_extensions:
        path = path.with_suffix('.png')
    
    fig.savefig(
        path, 
        dpi=dpi, 
        bbox_inches=bbox_inches,
        **kwargs
    )
    logger.info(f"Saved plot to {path}")
    return path


def save_dict_to_json(data: Dict[str, Any], path: Union[str, Path], 
                    indent: int = 2, **kwargs) -> Path:
    """Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        path: Output file path
        indent: Indentation level for the JSON file
        **kwargs: Additional arguments to pass to json.dump()
        
    Returns:
        Path: Absolute path to the saved file
    """
    path = Path(path).resolve()
    ensure_dir_exists(path.parent)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, **kwargs)
    
    logger.debug(f"Saved dictionary to {path}")
    return path


def copy_file(src: Union[str, Path], dst: Union[str, Path], 
             overwrite: bool = False) -> Path:
    """Copy a file from source to destination.
    
    Args:
        src: Source file path
        dst: Destination path (can be directory or file)
        overwrite: Whether to overwrite if destination exists
        
    Returns:
        Path: Path to the copied file
    """
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    if dst_path.is_dir():
        dst_path = dst_path / src_path.name
    
    if dst_path.exists() and not overwrite:
        raise FileExistsError(f"Destination file exists and overwrite=False: {dst_path}")
    
    ensure_dir_exists(dst_path.parent)
    shutil.copy2(src_path, dst_path)
    logger.debug(f"Copied {src_path} to {dst_path}")
    return dst_path


def get_project_root() -> Path:
    """Get the project root directory.
    
    Returns:
        Path: Path to the project root directory
    """
    # This assumes this file is in the project's src/utils directory
    return Path(__file__).parent.parent.parent


def get_reports_dir() -> Path:
    """Get the reports directory, creating it if it doesn't exist.
    
    Returns:
        Path: Path to the reports directory
    """
    reports_dir = get_project_root() / 'reports'
    return ensure_dir_exists(reports_dir)


def get_assets_dir() -> Path:
    """Get the assets directory, creating it if it doesn't exist.
    
    Returns:
        Path: Path to the assets directory
    """
    assets_dir = get_project_root() / 'assets'
    return ensure_dir_exists(assets_dir)


def get_models_dir() -> Path:
    """Get the models directory, creating it if it doesn't exist.
    
    Returns:
        Path: Path to the models directory
    """
    models_dir = get_project_root() / 'models'
    return ensure_dir_exists(models_dir)


def get_data_dir() -> Path:
    """Get the data directory, creating it if it doesn't exist.
    
    Returns:
        Path: Path to the data directory
    """
    data_dir = get_project_root() / 'data'
    return ensure_dir_exists(data_dir)
