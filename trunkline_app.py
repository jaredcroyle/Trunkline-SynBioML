#!/usr/bin/env python3
"""
Trunkline ML Application
"""
import os
import sys
import logging
import traceback
from pathlib import Path

# Set up logging
log_dir = Path.home() / "Library" / "Logs" / "Trunkline"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "trunkline.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("Trunkline")

def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Try to show an error dialog if possible
    try:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        
        error_msg = f"An unhandled exception occurred.\n\n"
        error_msg += f"Type: {exc_type.__name__}\n"
        error_msg += f"Message: {str(exc_value)}\n\n"
        error_msg += f"Check the log file for details:\n{log_file}"
        
        messagebox.showerror("Error", error_msg)
        root.destroy()
    except Exception as e:
        logger.error("Failed to show error dialog: %s", str(e))

# Set the exception hook
sys.excepthook = log_uncaught_exceptions

def main():
    """Main application entry point."""
    try:
        logger.info("Starting Trunkline ML application")
        logger.info("Python executable: %s", sys.executable)
        logger.info("Working directory: %s", os.getcwd())
        logger.info("Environment variables: %s", {k: v for k, v in os.environ.items() 
                                             if k in ['PATH', 'PYTHONPATH', 'PYTHONHOME', 'TK_LIBRARY']})
        
        # Add the project root to the Python path
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Import the GUI
        try:
            from src.gui.gui import TrunklineGUI
            logger.info("Successfully imported TrunklineGUI")
        except ImportError as e:
            logger.error("Failed to import TrunklineGUI: %s", str(e))
            logger.error("Python path: %s", sys.path)
            raise
        
        # Initialize and run the GUI
        logger.info("Initializing GUI...")
        app = TrunklineGUI()
        logger.info("Starting main GUI loop")
        app.run()
        logger.info("GUI loop exited")
        
    except Exception as e:
        logger.critical("Fatal error in main: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
