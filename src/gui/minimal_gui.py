import tkinter as tk
from tkinter import ttk
import os

class MinimalGUI:
    def __init__(self):
        print("Initializing MinimalGUI...")
        self.root = tk.Tk()
        self.root.title("Minimal GUI Test")
        
        # Set window size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 800
        window_height = 600
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Add a simple label
        label = ttk.Label(self.root, text="Hello, Trunkline ML!", font=('Arial', 16))
        label.pack(pady=20)
        
        # Add a button
        button = ttk.Button(self.root, text="Click Me", command=self.on_button_click)
        button.pack(pady=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        print("Minimal GUI initialized")
    
    def on_button_click(self):
        print("Button clicked!")
        self.status_var.set("Button was clicked!")
    
    def run(self):
        print("Starting main loop...")
        
        # Force window to stay on top and be visible
        self.root.attributes('-topmost', True)
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        # Add a delay before starting main loop
        print("Waiting 1 second before starting main loop...")
        self.root.after(1000, self.root.attributes, '-topmost', False)
        
        # Print debug info
        print(f"Window state: {self.root.state()}")
        print(f"Window size: {self.root.winfo_width()}x{self.root.winfo_height()}")
        print(f"Window position: +{self.root.winfo_x()}+{self.root.winfo_y()}")
        
        # Add a quit button to help with debugging
        quit_btn = ttk.Button(self.root, text="Quit", command=self.root.quit)
        quit_btn.pack(pady=20)
        
        print("Starting mainloop...")
        
        try:
            self.root.mainloop()
            print("Main loop ended")
        except Exception as e:
            print(f"Error in main loop: {e}")
            raise

if __name__ == "__main__":
    print("Starting application...")
    app = MinimalGUI()
    app.run()
    print("Application closed")
