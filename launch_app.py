#!/usr/bin/env python3
"""
Launcher script for Trunkline ML application.
This ensures the GUI launches properly when double-clicking the app bundle.
"""
import os
import sys
import traceback
import logging
from datetime import datetime

def setup_logging():
    """Set up logging to a file for debugging."""
    log_dir = os.path.expanduser("~/Library/Logs/Trunkline")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"launch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('TrunklineLauncher')

def main():
    logger = setup_logging()
    logger.info("Starting Trunkline Launcher")
    
    try:
        # Get the directory containing this script
        app_root = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Application root: {app_root}")
        
        # Set Python path to include the project directory
        os.environ['PYTHONPATH'] = f"{app_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
        logger.info(f"PYTHONPATH: {os.environ['PYTHONPATH']}")
        
        # Check Python version and paths
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        
        # Activate virtual environment if it exists
        venv_path = os.path.join(app_root, 'venv')
        if os.path.exists(venv_path):
            logger.info(f"Found virtual environment at: {venv_path}")
            if sys.platform == 'darwin':  # macOS
                activate_script = os.path.join(venv_path, 'bin', 'activate_this.py')
                if os.path.exists(activate_script):
                    logger.info("Activating virtual environment")
                    with open(activate_script) as f:
                        exec(f.read(), {'__file__': activate_script})
        else:
            logger.warning("No virtual environment found")
        
        # Log installed packages
        try:
            import pkg_resources
            installed_packages = [f"{p.key}=={p.version}" for p in pkg_resources.working_set]
            logger.info(f"Installed packages: {', '.join(installed_packages)}")
        except Exception as e:
            logger.warning(f"Could not list installed packages: {e}")
        
        # Import and run the GUI
        logger.info("Attempting to import TrunklineGUI")
        from src.gui.gui import TrunklineGUI
        
        logger.info("Creating TrunklineGUI instance")
        app = TrunklineGUI()
        
        logger.info("Starting main loop")
        app.run()
        
    except Exception as e:
        error_msg = f"Error launching application: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Show error dialog if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            messagebox.showerror("Trunkline Error", f"Failed to start Trunkline:\n\n{str(e)}")
            root.destroy()
        except Exception as dialog_error:
            logger.error(f"Could not show error dialog: {dialog_error}")
        
        sys.exit(1)

if __name__ == "__main__":
    main()
