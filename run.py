import os
import sys

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import and run the GUI
from src.gui.main import TrunklineGUI

if __name__ == "__main__":
    app = TrunklineGUI()
    app.run()
