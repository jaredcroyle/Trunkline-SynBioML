import tkinter as tk
from demo_gui import DemoTrunklineGUI, configure_styles

def main():
    """Main entry point for the application."""
    # Create the main window
    root = tk.Tk()
    
    # Configure styles
    configure_styles()
    
    # Create and run the application
    app = DemoTrunklineGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
