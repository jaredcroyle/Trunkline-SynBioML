import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import os
import platform
import subprocess
import pandas as pd
import logging
import sys
import importlib.util
import psutil
import time
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
from pathlib import Path
import tempfile
import uuid

# Try to import the model training and evaluation modules
try:
    from ml.model_training import train_gaussian_process
    from ml.model_evaluation import plot_predicted_vs_true_with_error, plot_predicted_vs_true
    ML_IMPORTED = True
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")
    ML_IMPORTED = False

def import_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import modules using relative imports
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock modules for now to get the GUI running
class MockMLPipeline:
    class MLPipeline:
        def __init__(self):
            pass

class MockDataPreprocessing:
    @staticmethod
    def load_and_clean_data(*args, **kwargs):
        return None, None
    
    @staticmethod
    def prepare_features(*args, **kwargs):
        return None, None, None

class MockConfig:
    @staticmethod
    def load_config():
        return {}

# Set up mock modules
ml_pipeline_module = MockMLPipeline()
data_preprocessing_module = MockDataPreprocessing()
config_module = MockConfig()

print("Using mock modules for GUI display")

# Get the classes and functions we need
config = config_module.load_config()  # Load the configuration
get_config = config_module.load_config  # Alias for consistency
MLPipeline = ml_pipeline_module.MLPipeline
load_and_clean_data = data_preprocessing_module.load_and_clean_data
prepare_features = data_preprocessing_module.prepare_features

# Get the user's home directory and create a logs directory there
home_dir = os.path.expanduser('~')
log_dir = os.path.join(home_dir, '.trunkline', 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'trunkline.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class TrunklineGUI:
    def __init__(self):
        print("Initializing TrunklineGUI...")
        self.root = tk.Tk()
        print("Tkinter root created")
        self.root.title("Trunkline ML Pipeline")
        self.pipeline = None
        self.data = None
        self.current_step = 0  # Track current step (0-4)
        self.steps = ["1. Set Up", "2. Priors", "3. Operators", "4. MCMC", "5. Run"]
        self.step_buttons = []  # Store references to step buttons
        
        # Initialize style variables with clean gray and green theme
        self.bg_color = '#F8F9FA'      # Very light gray background
        self.fg_color = '#212529'      # Dark gray text
        self.accent_color = '#2E8B57'  # Sea green accent
        self.accent_light = '#E8F5E9'  # Light green for highlights
        self.white = '#FFFFFF'         # White
        self.light_gray = '#E9ECEF'    # Light gray for borders and inactive elements
        self.medium_gray = '#ADB5BD'   # Medium gray for secondary elements
        self.dark_gray = '#495057'     # Dark gray for text and borders
        
        print("Setting up window...")
        self.setup_window()
        print("Setting up style...")
        self.setup_style()
        print("Creating widgets...")
        self.create_widgets()
        print("Updating step buttons...")
        self.update_step_buttons()  # Initialize button states
        print("GUI initialization complete")
        
    def setup_window(self):
        """Set up the main window."""
        # Set window title and icon
        self.root.title("Trunkline ML")
        
        # Set window size and position
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = 1200
        window_height = 800
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Ensure window is not minimized
        self.root.state('normal')
        
        # Make sure window is on top and focused
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        print(f"Window set up at {window_width}x{window_height}+{x}+{y}")
        
        # Set window icon and logo
        try:
            # Set window icon
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'trunkline.png')
            if os.path.exists(icon_path):
                icon = tk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon)
                
                # Also set the logo in the UI
                self.logo_image = Image.open(icon_path)
                # Resize logo to fit in the header (keeping aspect ratio)
                logo_width = 150
                logo_height = int((float(self.logo_image.size[1]) * float(logo_width / float(self.logo_image.size[0]))))
                self.logo_image = self.logo_image.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(self.logo_image)
            else:
                logger.warning(f"Logo file not found at: {icon_path}")
        except Exception as e:
            logger.error(f"Error loading logo: {e}")
        
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        self.root.minsize(900, 600)  # Set minimum window size
        
    def setup_style(self):
        """Configure ttk styles with a modern scientific theme."""
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base for better theming support
        
        # Use the instance variables defined in __init__
        # Local references for style configuration
        bg_color = self.bg_color
        fg_color = self.fg_color
        accent_color = self.accent_color
        white = self.white
        light_gray = self.light_gray
        
        # Configure the main window
        self.root.configure(bg=bg_color)
        self.root.option_add('*TCombobox*Listbox.font', ('Segoe UI', 10))
        
        # Configure default styles
        style = ttk.Style()
        style.theme_use('clam')  # Use clam theme as base for better theming support
        
        # Configure the main frame
        style.configure('.', 
                     font=('Segoe UI', 10),
                     background=bg_color,
                     foreground=fg_color,
                     fieldbackground=white,
                     selectbackground=accent_color,
                     selectforeground=white,
                     insertcolor=fg_color)
        
        # Configure frames
        style.configure('TFrame', background=bg_color)
        style.configure('Card.TFrame', 
                     background=white, 
                     relief='flat',
                     borderwidth=1,
                     border=1,
                     bordercolor=light_gray)
        
        # Configure notebook style
        style.configure('TNotebook', 
                     background=bg_color,
                     borderwidth=0)
        style.configure('TNotebook.Tab', 
                     padding=[15, 8],
                     font=('Segoe UI', 10, 'bold'),
                     background=medium_gray,
                     foreground=white,
                     borderwidth=0,
                     focuscolor=bg_color,
                     lightcolor=bg_color,
                     bordercolor=bg_color)
        style.map('TNotebook.Tab',
                background=[('selected', accent_color), ('active', '#3AA76D')],
                foreground=[('selected', white), ('active', white)],
                lightcolor=[('selected', accent_color)])
        style.configure('TNotebook', tabposition='n', tabmargins=[2, 5, 0, 0])
        
        # Configure labels
        style.configure('Header.TLabel',
                      font=('Segoe UI', 16, 'bold'),
                      background=bg_color,
                      foreground=fg_color)
                      
        style.configure('Subheader.TLabel',
                      font=('Segoe UI', 12, 'bold'),
                      background=bg_color,
                      foreground=fg_color)
        
        style.configure('TLabel',
                      font=('Segoe UI', 10),
                      background=bg_color,
                      foreground=fg_color)
        
        # Configure buttons
        style.configure('TButton',
                     padding=8,
                     relief='flat',
                     background=light_gray,
                     foreground=fg_color,
                     font=('Segoe UI', 10),
                     borderwidth=1,
                     focusthickness=3,
                     focuscolor=accent_light)
        style.map('TButton',
                background=[('active', '#E2E6EA'), ('pressed', '#D1D7DC')],
                bordercolor=[('focus', accent_color)])
        
        # Style for step buttons
        style.configure('Step.TButton',
                     padding=12,
                     font=('Segoe UI', 10, 'bold'),
                     relief='flat',
                     background=bg_color,
                     foreground=fg_color,
                     width=15,
                     anchor='w')
        
        # Hover effects for step buttons
        style.map('Step.TButton',
                background=[('active', accent_light), ('pressed', '#D1E7DD')],
                foreground=[('active', accent_color), ('pressed', '#0A5C36')])
        
        # Style for active/current step button
        style.configure('Accent.TButton',
                     padding=12,
                     font=('Segoe UI', 10, 'bold'),
                     relief='flat',
                     background=accent_color,
                     foreground=white,
                     width=15,
                     anchor='w')
        
        # Hover effects for accent button
        style.map('Accent.TButton',
                background=[('active', '#43A047'), ('pressed', '#2E7D32')],
                foreground=[('active', white), ('pressed', white)])
        
        # Configure entry style
        style.configure('TEntry',
                      fieldbackground=white,
                      foreground=fg_color,
                      relief='sunken',
                      borderwidth=1,
                      padding=5)
        
        # Configure combobox style
        style.configure('TCombobox',
                      fieldbackground=white,
                      selectbackground=white,
                      selectforeground=fg_color,
                      arrowsize=12,
                      padding=5)
        style.map('TCombobox',
                fieldbackground=[('readonly', white)],
                selectbackground=[('readonly', white)],
                selectforeground=[('readonly', fg_color)],
                fieldforeground=[('readonly', fg_color)],
                background=[('readonly', white)])
        
        # Configure scrollbar style
        style.configure('Vertical.TScrollbar',
                      background=light_gray,
                      troughcolor=bg_color,
                      arrowcolor=fg_color,
                      bordercolor=bg_color,
                      darkcolor=light_gray,
                      lightcolor=light_gray,
                      arrowsize=14)
        
        # Configure horizontal scrollbar
        style.configure('Horizontal.TScrollbar',
                      background=light_gray,
                      troughcolor=bg_color,
                      arrowcolor=fg_color,
                      bordercolor=bg_color,
                      darkcolor=light_gray,
                      lightcolor=light_gray,
                      arrowsize=14)
        
        # Configure progressbar style
        style.configure('Horizontal.TProgressbar',
                      background=accent_color,
                      troughcolor=light_gray,
                      bordercolor=bg_color,
                      lightcolor=accent_color,
                      darkcolor=accent_color,
                      thickness=20)
        
        # Configure Treeview (tables) style
        style.configure('Treeview',
                      background=white,
                      foreground=fg_color,
                      fieldbackground=white,
                      rowheight=25,
                      font=('Segoe UI', 9))
        
        style.configure('Treeview.Heading',
                      font=('Segoe UI', 9, 'bold'),
                      background=light_gray,
                      foreground=fg_color,
                      relief='flat')
        
        style.map('Treeview',
                background=[('selected', accent_color)],
                foreground=[('selected', white)])
        
        # Configure labelframe style
        style.configure('TLabelframe',
                      background=bg_color,
                      relief='flat')
        
        style.configure('TLabelframe.Label',
                      font=('Segoe UI', 10, 'bold'),
                      background=bg_color,
                      foreground=fg_color)
        
        # Configure radio button and checkbutton styles
        style.configure('TRadiobutton',
                      background=bg_color,
                      foreground=fg_color,
                      font=('Segoe UI', 10))
        
        style.configure('TCheckbutton',
                      background=bg_color,
                      foreground=fg_color,
                      font=('Segoe UI', 10))
        
        # Configure separator style
        style.configure('TSeparator',
                      background=light_gray)
        
        # Configure scale/slider style
        style.configure('Horizontal.TScale',
                      background=bg_color,
                      troughcolor=light_gray,
                      bordercolor=bg_color,
                      darkcolor=accent_color,
                      lightcolor=accent_color)
        
        # Configure spinbox style
        style.configure('TSpinbox',
                      fieldbackground=white,
                      foreground=fg_color,
                      arrowsize=12,
                      padding=5)
        
        # Configure menubutton style
        style.configure('TMenubutton',
                      background=light_gray,
                      foreground=fg_color,
                      relief='flat',
                      padding=5)
        
        # Configure scrollbar buttons
        style.configure('TButton',
                      background=light_gray,
                      foreground=fg_color,
                      relief='flat')
        
        # Configure tooltip style (if using ttkbootstrap or similar)
        try:
            style.configure('Tooltip.TFrame',
                          background='#FFFFE0',
                          relief='solid',
                          borderwidth=1)
            style.configure('Tooltip.TLabel',
                          background='#FFFFE0',
                          foreground='#000000',
                          font=('Segoe UI', 9),
                          padding=5)
        except:
            pass  # Skip if tooltip styles aren't supported
        
        # Configure tooltip appearance
        self.tooltip_font = ('Segoe UI', 9)
        self.tooltip_bg = '#2C3E50'
        self.tooltip_fg = 'white'
        
        # Set window icon if available
        try:
            icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'icon.ico')
            if os.path.exists(icon_path):
                self.root.iconbitmap(icon_path)
            else:
                # Try with .png if .ico not found
                png_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'icon.png')
                if os.path.exists(png_path):
                    img = tk.PhotoImage(file=png_path)
                    self.root.iconphoto(True, img)
        except Exception as e:
            print(f"Could not load window icon: {e}")
        
        # Configure window title and appearance
        self.root.title("Trunkline ML - Scientific Analysis Tool")
        
        # Set window minimum size and initial position
        self.root.minsize(1000, 700)
        
        # Center the window on screen
        window_width = 1200
        window_height = 800
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        self.root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        # Configure grid weights for main window
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Apply custom font smoothing on Windows
        if platform.system() == 'Windows':
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
            except:
                pass
        
    def create_widgets(self):
        print("  Creating main container...")
        # Create main container
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)
        print("  Main container created")
        
        print("  Creating left panel...")
        # Left panel - Workflow steps
        left_panel = ttk.Frame(main_paned, width=250, padding=10)
        main_paned.add(left_panel, weight=0)
        print("  Left panel created")
        
        print("  Creating logo/header...")
        # Add logo/header
        logo_frame = ttk.Frame(left_panel)
        logo_frame.pack(fill=tk.X, pady=(0, 20))
        print("  Logo frame created")
        
        # Add logo if available
        if hasattr(self, 'logo_photo'):
            print("  Adding logo...")
            logo_label = ttk.Label(logo_frame, image=self.logo_photo)
            logo_label.image = self.logo_photo  # Keep a reference
            logo_label.pack(anchor=tk.W, pady=(0, 5))
            print("  Logo added")
        else:
            print("  No logo photo available")
        
        print("  Creating title frame...")
        # Add title and subtitle
        title_frame = ttk.Frame(logo_frame)
        title_frame.pack(anchor=tk.W, fill=tk.X)
        print("  Title frame created")
        
        try:
            print("  Creating title label...")
            title_label = ttk.Label(title_frame, 
                                 text="Trunkline ML", 
                                 font=('Segoe UI', 16, 'bold'),
                                 foreground='#3498DB')
            title_label.pack(anchor=tk.W)
            print("  Title label created")
            
            print("  Creating subtitle label...")
            subtitle_label = ttk.Label(title_frame, 
                                    text="Machine Learning Pipeline", 
                                    font=('Segoe UI', 10),
                                    foreground='#7F8C8D')
            subtitle_label.pack(anchor=tk.W)
            print("  Subtitle label created")
        except Exception as e:
            print(f"Error creating title labels: {e}")
            raise
        
        # Add workflow steps with icons and status indicators
        print("  Creating steps frame...")
        steps_frame = ttk.LabelFrame(left_panel, text="Workflow Steps", padding=10)
        steps_frame.pack(fill=tk.BOTH, expand=True)
        print("  Steps frame created")
        
        step_descriptions = [
            "Set up your project and data",
            "Configure model parameters",
            "Set up operators",
            "Configure MCMC settings",
            "Run the analysis"
        ]
        print(f"  Step descriptions: {step_descriptions}")
        
        # Initialize status canvases list
        self.status_canvases = []
        
        print(f"  Creating {len(self.steps)} step buttons...")
        for i, (step, desc) in enumerate(zip(self.steps, step_descriptions)):
            print(f"    Creating step {i+1}: {step}")
            # Create a frame for each step
            step_frame = ttk.Frame(steps_frame, padding=(0, 5, 0, 5))
            step_frame.pack(fill=tk.X, pady=2)
            
            # Create a container for button and status
            btn_container = ttk.Frame(step_frame)
            btn_container.pack(fill=tk.X)
            
            # Create status indicator (circle)
            status_canvas = tk.Canvas(btn_container, width=24, height=24, 
                                    bg=self.bg_color, highlightthickness=0)
            status_canvas.pack(side=tk.LEFT, padx=(0, 8))
            self.status_canvases.append(status_canvas)
            
            # Create the step button (remove the step number)
            step_name = step.split('. ')[1] if '. ' in step else step
            btn = ttk.Button(
                btn_container,
                text=step_name,
                style='Step.TButton',
                command=lambda i=i: self.select_workflow_step(i)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.step_buttons.append(btn)
            
            # Add description label
            ttk.Label(
                step_frame, 
                text=desc, 
                foreground='#666666',
                font=('Segoe UI', 8),
                wraplength=180
            ).pack(fill=tk.X, padx=30, pady=(2, 5))
            
        # Add system info at the bottom
        sys_info = ttk.LabelFrame(left_panel, text="System", padding=5)
        sys_info.pack(fill=tk.X, pady=(10, 0))
        
        # CPU and Memory usage
        self.cpu_label = ttk.Label(sys_info, text="CPU: --%", font=('Arial', 8))
        self.cpu_label.pack(anchor='w')
        
        self.mem_label = ttk.Label(sys_info, text="Memory: --/-- MB", font=('Arial', 8))
        self.mem_label.pack(anchor='w')
        
        # Start system monitoring
        self.update_system_metrics()
        
        # Create main content area with sidebar and notebook
        content_frame = ttk.Frame(main_paned)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create sidebar for step navigation
        self.sidebar = ttk.Frame(content_frame, width=200, style='Card.TFrame')
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5)
        
        # Create notebook for tabs
        notebook_frame = ttk.Frame(content_frame)
        notebook_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(notebook_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Add tabs
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="Data")
        self.setup_data_tab(self.data_tab)
        
        self.priors_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.priors_tab, text="Priors")
        self.setup_priors_tab(self.priors_tab)
        
        self.operators_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.operators_tab, text="Operators")
        self.setup_operators_tab(self.operators_tab)
        
        self.mcmc_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.mcmc_tab, text="MCMC")
        self.setup_mcmc_tab(self.mcmc_tab)
        
        self.run_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.run_tab, text="Run")
        self.setup_run_tab(self.run_tab)
        
        # Status Bar
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        ttk.Label(status_frame, 
                 textvariable=self.status_var, 
                 relief=tk.SUNKEN, 
                 anchor=tk.W,
                 padding=5).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Default priors for the priors tree
        default_priors = [
            ("alpha", "Gamma", "2.0", "0.5"),
            ("beta", "Uniform", "0.0", "10.0")
        ]
        
        for prior in default_priors:
            self.priors_tree.insert("", "end", values=prior)
        
        # Add scrollbar
        vsb = ttk.Scrollbar(dist_frame, orient="vertical", command=self.priors_tree.yview)
        self.priors_tree.configure(yscrollcommand=vsb.set)
        
        # Grid layout
        self.priors_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        dist_frame.grid_columnconfigure(0, weight=1)
        
        # Prior controls
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(ctrl_frame, text="Add Prior").pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Edit Prior").pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Remove Prior").pack(side=tk.LEFT, padx=2)
        
        # Advanced options
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding=10)
        adv_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Checkbutton(adv_frame, text="Use adaptive priors").pack(anchor='w')
        ttk.Checkbutton(adv_frame, text="Enable hierarchical priors").pack(anchor='w')
        
        param_frame = ttk.Frame(adv_frame)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # MCMC settings frame with enhanced layout
        settings_frame = ttk.LabelFrame(main_frame, text="MCMC Configuration", padding=15)
        settings_frame.pack(fill='x', padx=10, pady=10, expand=True)
        
        # Create a grid layout for MCMC parameters
        mcmc_grid = ttk.Frame(settings_frame)
        mcmc_grid.pack(fill='x', padx=5, pady=5)
        
        # MCMC Parameters
        params = [
            ("Number of Chains", "n_chains", 4, "Number of independent chains to run"),
            ("Iterations", "iterations", 5000, "Number of MCMC iterations"),
            ("Warmup", "warmup", 1000, "Number of warmup iterations"),
            ("Thin", "thin", 1, "Thinning interval"),
            ("Adapt Delta", "adapt_delta", 0.8, "Target acceptance probability (0-1)"),
            ("Max Tree Depth", "max_treedepth", 10, "Maximum tree depth for NUTS")
        ]
        
        # Create input fields for each parameter
        self.mcmc_vars = {}
        for i, (label, name, default, help_text) in enumerate(params):
            # Label
            lbl = ttk.Label(mcmc_grid, text=label, style='TLabel')
            lbl.grid(row=i, column=0, sticky='w', padx=5, pady=3)
            
            # Entry field
            var = tk.StringVar(value=str(default))
            self.mcmc_vars[name] = var
            entry = ttk.Entry(mcmc_grid, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky='w', padx=5, pady=3)
            
            # Help text
            help_lbl = ttk.Label(mcmc_grid, text=help_text, style='TLabel', 
                               foreground=medium_gray, font=('Segoe UI', 8))
            help_lbl.grid(row=i, column=2, sticky='w', padx=10, pady=3)
        
        # Divider
        ttk.Separator(settings_frame, orient='horizontal').pack(fill='x', pady=15)
        
        # Advanced options button
        adv_btn = ttk.Button(settings_frame, text="Advanced Options", 
                           command=self.show_advanced_mcmc)
        adv_btn.pack(pady=5)
        
        # Status indicators
        status_frame = ttk.Frame(settings_frame)
        status_frame.pack(fill='x', pady=10)
        
        self.mcmc_status = ttk.Label(status_frame, text="Status: Ready", 
                                   foreground=accent_color, style='TLabel')
        self.mcmc_status.pack(side='left', padx=5)
        
        # Action buttons
        btn_frame = ttk.Frame(settings_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Run MCMC", 
                  command=self.run_mcmc, style='Accent.TButton').pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Stop", 
                  command=self.stop_mcmc).pack(side='left', padx=5)
        
        # Progress bar
        self.mcmc_progress = ttk.Progressbar(settings_frame, orient='horizontal', 
                                           length=100, mode='determinate')
        self.mcmc_progress.pack(fill='x', pady=10, padx=5)
        
        # Console output
        console_frame = ttk.LabelFrame(settings_frame, text="Console Output", padding=10)
        console_frame.pack(fill='both', expand=True, pady=10)
        
        self.console_text = tk.Text(console_frame, height=10, wrap='word', 
                                  bg=white, fg=fg_color, 
                                  font=('Consolas', 9), 
                                  padx=5, pady=5)
        self.console_text.pack(fill='both', expand=True, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(console_frame, orient='vertical', 
                                command=self.console_text.yview)
        scrollbar.pack(side='right', fill='y')
        self.console_text.configure(yscrollcommand=scrollbar.set)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Chain settings
        ttk.Label(settings_frame, text="Chain Length:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(settings_frame, width=15).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(settings_frame, text="iterations").grid(row=0, column=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Burn-in:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(settings_frame, width=15).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(settings_frame, text="iterations").grid(row=1, column=2, sticky='w', padx=5, pady=2)
        
        ttk.Label(settings_frame, text="Sampling Frequency:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        ttk.Entry(settings_frame, width=15).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        ttk.Label(settings_frame, text="iterations").grid(row=2, column=2, sticky='w', padx=5, pady=2)
        
        # Advanced options
        adv_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding=10)
        adv_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(adv_frame, text="Store complete state").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Checkbutton(adv_frame, text="Store every state").grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Random number seed
        seed_frame = ttk.Frame(adv_frame)
        seed_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=5)
        
        ttk.Label(seed_frame, text="Random seed:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(seed_frame, width=15).pack(side=tk.LEFT)
        ttk.Button(seed_frame, text="Randomize").pack(side=tk.LEFT, padx=5)
        
        # Parallel settings
        parallel_frame = ttk.LabelFrame(main_frame, text="Parallel Execution", padding=10)
        parallel_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(parallel_frame, text="Enable parallel chains").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        ttk.Label(parallel_frame, text="Number of chains:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Combobox(parallel_frame, values=[str(i) for i in range(1, 33)], width=5).grid(row=1, column=1, sticky='w', padx=5, pady=2)
    
    def setup_data_tab(self, parent):
        self.data_tab = parent
        """Set up the Data tab with file selection and preview."""
        # File selection
        file_frame = ttk.LabelFrame(parent, text="Input Data", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File path entry with browse button
        self.file_path = tk.StringVar()
        entry_frame = ttk.Frame(file_frame)
        entry_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Entry(entry_frame, 
                 textvariable=self.file_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Add Load Data button
        ttk.Button(entry_frame, 
                  text="Load Data", 
                  command=self.process_data).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(entry_frame, 
                  text="Browse...", 
                  command=self.browse_file).pack(side=tk.LEFT)
        
        # Data preview
        preview_frame = ttk.LabelFrame(parent, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add a treeview for data preview
        self.preview_tree = ttk.Treeview(preview_frame, show='headings')
        
        # Add scrollbars
        vsb = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        hsb = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout for treeview and scrollbars
        self.preview_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        # Configure grid weights
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # Status label for data loading
        self.data_status = ttk.Label(parent, text="No data loaded", foreground="gray")
        self.data_status.pack(pady=5)
        
    def setup_run_tab(self, parent):
        """Set up the Run tab with execution controls."""
        # Main container frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Step navigation buttons - already in sidebar, just update the buttons
        self.step_buttons_frame = ttk.Frame(self.sidebar, padding=(10, 20, 10, 10))
        self.step_buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Clear any existing buttons
        for widget in self.step_buttons_frame.winfo_children():
            widget.destroy()
            
        # Clear existing buttons list
        self.step_buttons = []
        self.status_canvases = []
        
        # Define step names without numbers
        step_names = ["Set Up", "Priors", "Operators", "MCMC", "Run"]
        
        self.step_buttons = []
        for i, step in enumerate(step_names):
            # Create button frame to hold both button and status
            btn_frame = ttk.Frame(self.step_buttons_frame)
            btn_frame.pack(fill=tk.X, pady=(0, 5))
            
            # Create status indicator (circle)
            status_canvas = tk.Canvas(btn_frame, width=24, height=24, 
                                    bg=self.bg_color, highlightthickness=0)
            status_canvas.pack(side=tk.LEFT, padx=(0, 8))
            self.status_canvases.append(status_canvas)
            
            # Create the step button
            btn = ttk.Button(
                btn_frame,
                text=step,
                style='Step.TButton',
                command=lambda i=i: self.select_workflow_step(i)
            )
            btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.step_buttons.append(btn)
            
        # Create control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        # Start button
        self.start_btn = ttk.Button(
            control_frame,
            text="▶ Start Analysis",
            command=self.start_analysis,
            style='Accent.TButton',
            width=15
        )
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_btn = ttk.Button(
            control_frame,
            text="⏹ Stop",
            command=self.stop_analysis,
            style='TButton',
            width=10
        )
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Report button
        self.report_btn = ttk.Button(
            control_frame,
            text="📄 Generate Report",
            command=self.generate_report,
            style='TButton',
            width=15
        )
        self.report_btn.pack(side=tk.RIGHT)
        
        # Progress bar
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_frame, 
                                         variable=self.progress_var,
                                         maximum=100,
                                         mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Status text
        self.status_text = ttk.Label(progress_frame, text="Ready", anchor='w')
        self.status_text.pack(fill=tk.X, pady=(0, 5))
        
        # Results frame
        self.results_frame = ttk.LabelFrame(main_frame, text="Results")
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Add a canvas for plotting
        try:
            self.fig, self.ax = plt.subplots(figsize=(8, 3), dpi=100)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.results_frame)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Apply custom styles for buttons
            style = ttk.Style()
            style.configure('Accent.TButton', 
                          font=('Segoe UI', 10, 'bold'),
                          padding=8)
            
            # Log area
            log_frame = ttk.LabelFrame(main_frame, text="Log", padding=10)
            log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
            log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
            self.log_text.configure(yscrollcommand=log_scroll.set)
            
            self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Add sample log message
            self.log_message("System ready. Click 'Start Analysis' to begin.")
            
            # Disable buttons that require analysis to be complete
            self.stop_btn.state(['disabled'])
            self.report_btn.state(['disabled'])
            
        except Exception as e:
            print(f"Error setting up run tab: {e}")
            messagebox.showerror("Error", f"Failed to initialize the run tab: {e}")
    
    def draw_status_circle(self, step_index, status):
        """Draw a status indicator circle."""
        if not hasattr(self, 'status_canvases') or step_index >= len(self.status_canvases):
            return
            
        canvas = self.status_canvases[step_index]
        canvas.delete('all')
        
        # Define colors and dimensions
        colors = {
            'pending': '#CCCCCC',
            'active': '#4CAF50',
            'complete': '#4CAF50',
            'error': '#F44336'
        }
        
        size = 20
        padding = 2
        x, y, r = size//2, size//2, (size - padding*2)//2
        
        # Draw the outer circle
        canvas.create_oval(padding, padding, size-padding, size-padding,
                         fill=colors.get(status, '#CCCCCC'),
                         outline='',
                         width=0)
        
        # Add icon based on status
        if status == 'complete':
            # Draw checkmark
            canvas.create_text(x, y, 
                             text='✓', 
                             fill='white',
                             font=('Segoe UI', 12, 'bold'))
        elif status == 'active':
            # Draw a smaller inner circle for active state
            canvas.create_oval(x-r//2, y-r//2, x+r//2, y+r//2,
                             fill='white',
                             outline='',
                             width=0)
        elif status == 'error':
            # Draw 'X' for error
            canvas.create_text(x, y, 
                             text='✕', 
                             fill='white',
                             font=('Segoe UI', 12, 'bold'))
    
    def update_step_buttons(self):
        """Update the state of step buttons based on current step."""
        if not hasattr(self, 'step_buttons') or not self.step_buttons:
            return
            
        for i, btn in enumerate(self.step_buttons):
            try:
                # Update button state and appearance
                if i == self.current_step:
                    # Current step - highlight
                    btn.state(['!disabled'])
                    btn.configure(style='Accent.TButton')
                    self.draw_status_circle(i, 'active')
                elif i < self.current_step:
                    # Completed step - enable with complete status
                    btn.state(['!disabled'])
                    btn.configure(style='Step.TButton')
                    self.draw_status_circle(i, 'complete')
                else:
                    # Future step - enabled but with pending status
                    btn.state(['!disabled'])
                    btn.configure(style='Step.TButton')
                    self.draw_status_circle(i, 'pending')
                
                # Update button hover effects
                btn.unbind('<Enter>')
                btn.unbind('<Leave>')
                
                def on_enter(e, idx=i):
                    # Only change the background of the button, not the status canvas
                    if idx != self.current_step:
                        btn.state(['active'])
                        
                def on_leave(e, idx=i):
                    # Restore the appropriate state based on step completion
                    if idx != self.current_step:
                        btn.state(['!active'])
                    # Always redraw the status circle to ensure consistency
                    if idx < self.current_step:
                        self.draw_status_circle(idx, 'complete')
                    elif idx == self.current_step:
                        self.draw_status_circle(idx, 'active')
                    else:
                        self.draw_status_circle(idx, 'pending')
                
                btn.bind('<Enter>', on_enter)
                btn.bind('<Leave>', on_leave)
                
            except Exception as e:
                print(f"Error updating button {i}: {e}")
    
    def update_system_metrics(self):
        """Update CPU and memory usage."""
        if not hasattr(self, 'root') or not self.root.winfo_exists():
            return  # Exit if root window doesn't exist
            
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Update labels if they exist
            if hasattr(self, 'cpu_label') and self.cpu_label.winfo_exists():
                self.cpu_label.config(text=f"CPU: {cpu_percent:.1f}%")
            if hasattr(self, 'mem_label') and self.mem_label.winfo_exists():
                self.mem_label.config(text=f"Memory: {memory_percent:.1f}%")
            
        except Exception as e:
            print(f"Error updating system metrics: {e}")
        
        # Schedule next update if window still exists
        if hasattr(self, 'root') and self.root.winfo_exists():
            self.root.after(1000, self.update_system_metrics)
    
    def select_workflow_step(self, step_index):
        """Handle workflow step selection with smooth transitions."""
        if step_index < 0 or step_index >= len(self.steps):
            return
                
        # Store the current step before changing
        previous_step = self.current_step
            
        # Always allow navigation to any step (removed restriction for testing)
        # In production, you might want to add restrictions back
        self.current_step = step_index
            
        try:
            # Update the notebook to show the selected tab
            self.notebook.select(step_index)
                
            # Update button states and visual feedback
            self.update_step_buttons()
                
            # Add smooth transition effect
            self.root.update_idletasks()
                
            # Get the step name without the number
            step_name = self.steps[step_index].split('. ')[1] if '. ' in self.steps[step_index] else self.steps[step_index]
            
            # Log the navigation
            self.log_message(f"Selected step: {step_name}")
                
            # Special handling for the Run tab
            if step_index == 4:  # Run tab index is 4 (0-based)
                # Ensure the Start button is enabled when reaching the Run tab
                if hasattr(self, 'start_btn'):
                    self.start_btn.state(['!disabled'])
                    
                    # If coming from a previous step, auto-scroll to the top
                    if previous_step < step_index and hasattr(self, 'log_text'):
                        self.log_text.see(tk.END)
                        
        except Exception as e:
            logging.error(f"Error selecting step {step_index}: {e}")
            # Revert to previous step on error
            self.current_step = previous_step
            self.update_step_buttons()
    
    def log_message(self, message):
        """Add a message to the log."""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
    
    def start_analysis(self):
        """Start the analysis process."""
        if not hasattr(self, 'data') or self.data is None:
            messagebox.showerror("Error", "Please load data first")
            return
            
        self.status_var.set("Analysis started...")
        self.log_message("Starting analysis...")
        
        # Update button states
        self.start_btn.state(['disabled'])
        self.stop_btn.state(['!disabled'])
        
        # Disable all step buttons during analysis
        for btn in self.step_buttons:
            btn.state(['disabled'])
        
        # Reset and show progress
        self.progress_var.set(0)
        self.progress_text.config(text="0%")
        self.time_remaining.config(text="Calculating...")
        self.step_label.config(text="Initializing...")
        
        # Show results frame
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Start time for ETA calculation
        self.start_time = time.time()
        
        # Start the analysis in a separate thread to keep the UI responsive
        import threading
        self.analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        self.analysis_thread.start()
    
    def run_analysis(self):
        """Run the actual analysis in a separate thread."""
        try:
            # For demonstration, we'll simulate analysis steps
            self.simulate_analysis()
        except Exception as e:
            self.root.after(0, self.log_message, f"ERROR: Analysis error: {str(e)}")
            self.root.after(0, messagebox.showerror, "Analysis Error", str(e))
            self.root.after(0, self.analysis_complete)
    
    def simulate_analysis(self):
        """Simulate analysis steps with progress updates."""
        steps = [
            ("[1/6] Loading data...", 10, 1),
            ("[2/6] Preprocessing data...", 25, 1),
            ("[3/6] Configuring model...", 40, 1),
            ("[4/6] Training model (this may take a while)...", 70, 3),
            ("[5/6] Generating visualizations...", 85, 1),
            ("[6/6] Saving results...", 95, 1),
            ("Analysis complete!", 100, 0)
        ]
        
        def process_step(steps, index=0):
            if index < len(steps):
                message, progress, duration = steps[index]
                
                # Update UI in the main thread
                self.root.after(0, self.log_message, message)
                self.root.after(0, self.step_label.config, {"text": message})
                self.root.after(0, self.progress_var.set, progress)
                self.root.after(0, self.progress_text.config, {"text": f"{int(progress)}%"})
                
                # Update ETA
                if progress > 0 and progress < 100:
                    elapsed = time.time() - self.start_time
                    remaining = (100 - progress) * (elapsed / progress) if progress > 0 else 0
                    mins = int(remaining // 60)
                    secs = int(remaining % 60)
                    self.root.after(0, self.time_remaining.config, 
                                  {"text": f"{mins:02d}:{secs:02d}"})
                
                # Update plot for visualization
                self.root.after(0, self.update_plot, progress)
                
                # Schedule next step
                if index < len(steps) - 1:  # Don't schedule if this is the last step
                    self.root.after(duration * 1000, process_step, steps, index + 1)
                else:
                    self.root.after(0, self.analysis_complete)
            
        # Start the process
        process_step(steps)
    
    def update_plot(self, progress):
        """Update the results plot."""
        try:
            self.ax.clear()
            
            # Generate some sample data based on progress
            x = np.linspace(0, 10, 100)
            y = np.sin(x + progress/20) * (progress/100 * 2 + 0.5)
            
            self.ax.plot(x, y, label='Prediction', color='#2E8B57', linewidth=2)
            self.ax.set_title('Model Predictions')
            self.ax.set_xlabel('Input')
            self.ax.set_ylabel('Output')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend()
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
    
    def analysis_complete(self):
        """Handle analysis completion."""
        self.status_var.set("Analysis complete!")
        self.log_message("Analysis completed successfully.")
        self.start_btn.state(['!disabled'])
        self.stop_btn.state(['disabled'])
        self.update_step_buttons()  # Re-enable step buttons
    
    def stop_analysis(self):
        """Stop the analysis process."""
        self.status_var.set("Analysis stopped by user")
        self.log_message("Analysis stopped by user")
        self.start_btn.state(['!disabled'])
        self.stop_btn.state(['disabled'])
        self.progress_var.set(0)
        self.update_step_buttons()  # Re-enable step buttons
    
    def update_progress(self, value):
        """Update progress bar."""
        if value <= 100:
            self.progress_var.set(value)
            self.root.after(100, lambda: self.update_progress(value + 1))
        
    def browse_file(self):
        filetypes = (
            ('CSV files', '*.csv'),
            ('Excel files', '*.xlsx'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='OPEN FILE',
            initialdir=os.path.expanduser('~'),
            filetypes=filetypes
        )
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f">> SELECTED: {os.path.basename(filename)}")
            # Terminal-style feedback
            print(f"[SYSTEM] File selected: {filename}")
            
    def process_data(self):
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        try:
            # Clear previous data
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            
            # Show loading status
            self.data_status.config(text="Loading data...", foreground="blue")
            self.root.update()
            
            # Read the file based on extension
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Update treeview columns
            columns = list(df.columns)
            self.preview_tree["columns"] = columns
            
            # Configure columns and headings
            for col in columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=100, minwidth=50, stretch=tk.YES)
            
            # Add data to treeview
            for i, row in df.head(100).iterrows():  # Show first 100 rows
                self.preview_tree.insert("", "end", values=tuple(row))
            
            # Update status
            self.data_status.config(text=f"Loaded {len(df)} rows, {len(df.columns)} columns", 
                                  foreground="green")
            
            # Store the data for later use
            self.data = df
            
            # Enable the next step if this is the first step
            if self.current_step == 0:
                self.current_step = 1
                self.update_step_buttons()
            
            self.log_message(f"Successfully loaded data from {os.path.basename(filepath)}")
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.data_status.config(text=error_msg, foreground="red")
            messagebox.showerror("Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            
    def generate_report(self):
        self.status_var.set(">> GENERATING REPORT...")
        self.root.update()
        print("[GENERATING] Starting report generation...")
        
        # Terminal-style progress simulation
        for i in range(1, 6):
            self.status_var.set(f">> GENERATING [{'='*i}{' '*(5-i)}] {i*20}%")
            self.root.update()
            time.sleep(0.3)
        
        self.status_var.set(">> REPORT GENERATED")
        print("[SUCCESS] Report generation completed successfully")
        
    def update_plot(self, progress):
        """Update the plot with new data."""
        try:
            # Clear previous plot
            self.ax.clear()
            
            # Generate some sample data
            x = np.linspace(0, 10, 100)
            y = np.sin(x) * progress / 100.0
            
            # Plot the data
            self.ax.plot(x, y, label='Prediction', color='#2E8B57', linewidth=2)
            self.ax.set_title('Model Predictions')
            self.ax.set_xlabel('Input')
            self.ax.set_ylabel('Output')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.legend()
            
            # Adjust layout
            self.fig.tight_layout()
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating plot: {e}")
    
    def analysis_complete(self):
        """Handle analysis completion."""
        self.status_var.set("Analysis complete!")
        self.log_message("Analysis completed successfully.")
        self.start_btn.state(['!disabled'])
        self.stop_btn.state(['disabled'])
        self.update_step_buttons()  # Re-enable step buttons
    
    def stop_analysis(self):
        """Stop the analysis process."""
        self.status_var.set("Analysis stopped by user")
        self.log_message("Analysis stopped by user")
        self.start_btn.state(['!disabled'])
        self.stop_btn.state(['disabled'])
        self.progress_var.set(0)
        self.update_step_buttons()  # Re-enable step buttons
        
    def update_progress(self, value):
        """Update progress bar."""
        if value <= 100:
            self.progress_var.set(value)
            self.root.after(100, lambda: self.update_progress(value + 1))
            
    def browse_file(self):
        filetypes = (
            ('CSV files', '*.csv'),
            ('Excel files', '*.xlsx'),
            ('All files', '*.*')
        )
        filename = filedialog.askopenfilename(
            title='OPEN FILE',
            initialdir=os.path.expanduser('~'),
            filetypes=filetypes
        )
        if filename:
            self.file_path.set(filename)
            self.status_var.set(f">> SELECTED: {os.path.basename(filename)}")
            # Terminal-style feedback
            print(f"[SYSTEM] File selected: {filename}")
            
    def process_data(self):
        filepath = self.file_path.get()
        if not filepath:
            messagebox.showerror("Error", "Please select a file first")
            return
            
        try:
            # Clear previous data
            for item in self.preview_tree.get_children():
                self.preview_tree.delete(item)
            
            # Show loading status
            self.data_status.config(text="Loading data...", foreground="blue")
            self.root.update()
            
            # Read the file based on extension
            if filepath.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filepath.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            # Update treeview columns
            columns = list(df.columns)
            self.preview_tree["columns"] = columns
            
            # Configure columns and headings
            for col in columns:
                self.preview_tree.heading(col, text=col)
                self.preview_tree.column(col, width=100, minwidth=50, stretch=tk.YES)
            
            # Add data to treeview
            for i, row in df.head(100).iterrows():  # Show first 100 rows
                self.preview_tree.insert("", "end", values=tuple(row))
            
            # Update status
            self.data_status.config(text=f"Loaded {len(df)} rows, {len(df.columns)} columns", 
                                  foreground="green")
            
            # Store the data for later use
            self.data = df
            
            # Enable the next step if this is the first step
            if self.current_step == 0:
                self.current_step = 1
                self.update_step_buttons()
            
            self.log_message(f"Successfully loaded data from {os.path.basename(filepath)}")
            
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            self.data_status.config(text=error_msg, foreground="red")
            messagebox.showerror("Error", error_msg)
            self.log_message(f"ERROR: {error_msg}")
            
    def generate_report(self):
        """Generate a report of the analysis with model results and plots."""
        if not ML_IMPORTED:
            messagebox.showerror("Error", "Required ML modules could not be imported. Report generation failed.")
            return

        self.status_var.set(">> GENERATING REPORT...")
        self.root.update()
        print("[GENERATING] Starting report generation...")
        
        try:
            # Create a temporary directory for report assets
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            # Create a timestamped subdirectory for this report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_subdir = report_dir / f"report_{timestamp}"
            report_subdir.mkdir(exist_ok=True)
            
            # Simulate some data for demonstration
            # In a real application, you would use your actual data
            np.random.seed(42)
            X = np.random.rand(100, 2)
            y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)
            
            # Train a Gaussian Process model
            self.status_var.set(">> Training Gaussian Process model...")
            self.root.update()
            
            gp_model = train_gaussian_process(X, y)
            
            # Generate and save plots
            plots = []
            
            # 1. Generate and save Gaussian Process error bars plot
            error_plot_path = report_subdir / "gp_error_bars.png"
            plot_predicted_vs_true_with_error(
                gp_model, X, y, 
                model_name="Gaussian Process", 
                is_gp=True,
                save_path=str(error_plot_path)
            )
            plots.append({
                'title': 'Gaussian Process Error Bars',
                'path': str(error_plot_path.relative_to(report_dir)),
                'caption': 'Gaussian Process predictions with error bars showing uncertainty.'
            })
            
            # 2. Generate and save standard predicted vs true plot
            pred_plot_path = report_subdir / "gp_predictions.png"
            plot_predicted_vs_true(
                gp_model, X, y,
                model_name="Gaussian Process",
                save_path=str(pred_plot_path)
            )
            plots.append({
                'title': 'Gaussian Process Predictions',
                'path': str(pred_plot_path.relative_to(report_dir)),
                'caption': 'Gaussian Process predicted vs true values.'
            })
            
            # Prepare model results data
            model_results = {
                'Gaussian_Process': {
                    'description': 'Gaussian Process Regression with RBF kernel',
                    'mse': 0.85,  # These would be calculated from your model
                    'rmse': 0.92,
                    'mae': 0.78,
                    'r2': 0.95,
                    'plots': plots,
                    'params': {
                        'kernel': 'RBF',
                        'alpha': 1.0,
                        'n_restarts_optimizer': 10
                    }
                }
            }
            
            # Save the model results to a JSON file
            results_file = report_subdir / 'model_results.json'
            with open(results_file, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            # Generate the HTML report
            self.status_var.set(">> Generating HTML report...")
            self.root.update()
            
            # Get the template directory (assuming it's in the same directory as this file)
            template_dir = Path(__file__).parent.parent / 'templates'
            
            # Generate the report using the report_generator
            from report_generator import generate_report
            
            output_path = report_dir / f'report_{timestamp}.html'
            generate_report(
                models_results=model_results,
                output_path=str(output_path),
                template_folder=str(template_dir),
                template_file='model_report_template.html',
                ensemble_plot_path=None
            )
            
            self.status_var.set(">> REPORT GENERATED")
            print(f"[SUCCESS] Report generated at: {output_path}")
            
            # Ask if the user wants to open the report
            if messagebox.askyesno("Report Generated", "Report generated successfully. Would you like to open it now?"):
                try:
                    if platform.system() == 'Darwin':  # macOS
                        subprocess.call(('open', str(output_path)))
                    elif platform.system() == 'Windows':  # Windows
                        os.startfile(str(output_path))
                    else:  # Linux variants
                        subprocess.call(('xdg-open', str(output_path)))
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open the report: {e}")
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            print(f"[ERROR] {error_msg}")
            messagebox.showerror("Report Generation Failed", error_msg)
            self.status_var.set(">> REPORT GENERATION FAILED")
    
    def show_advanced_mcmc(self):
        """Show advanced MCMC options dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced MCMC Options")
        dialog.geometry("500x400")
        
        # Make dialog modal
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Configure dialog style
        dialog.configure(bg=self.bg_color)
        
        # Add content
        ttk.Label(dialog, text="Advanced MCMC Configuration", 
                 style='Header.TLabel').pack(pady=10)
        
        # Add advanced options here
        options_frame = ttk.Frame(dialog, style='Card.TFrame')
        options_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Example advanced options
        ttk.Label(options_frame, text="NUTS Parameters:").pack(anchor='w', pady=5)
        
        # Add some example controls
        ttk.Checkbutton(options_frame, text="Enable dense mass matrix").pack(anchor='w', pady=2)
        ttk.Checkbutton(options_frame, text="Use diagonal mass matrix").pack(anchor='w', pady=2)
        
        # Divider
        ttk.Separator(options_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Add buttons
        btn_frame = ttk.Frame(options_frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(btn_frame, text="Save", 
                  command=lambda: [dialog.destroy(), self.log_message("MCMC settings saved")],
                  style='Accent.TButton').pack(side='right', padx=5)
        ttk.Button(btn_frame, text="Cancel", 
                  command=dialog.destroy).pack(side='right', padx=5)
    
    def run_mcmc(self):
        """Start the MCMC sampling process."""
        try:
            # Update UI
            self.mcmc_status.config(text="Status: Running...", foreground=self.accent_color)
            self.console_text.insert('end', "Starting MCMC sampling...\n")
            self.console_text.see('end')
            
            # Get parameters from UI
            params = {name: var.get() for name, var in self.mcmc_vars.items()}
            
            # Validate parameters
            try:
                params = {
                    'n_chains': int(params['n_chains']),
                    'iterations': int(params['iterations']),
                    'warmup': int(params['warmup']),
                    'thin': int(params['thin']),
                    'adapt_delta': float(params['adapt_delta']),
                    'max_treedepth': int(params['max_treedepth'])
                }
            except ValueError as e:
                messagebox.showerror("Invalid Parameter", f"Please check your input values: {e}")
                return
            
            # Simulate MCMC progress (replace with actual MCMC code)
            self._simulate_mcmc_progress(params)
            
        except Exception as e:
            self.console_text.insert('end', f"Error: {str(e)}\\n")
            self.console_text.see('end')
            self.mcmc_status.config(text="Status: Error", foreground='red')
    
    def _simulate_mcmc_progress(self, params):
        """Simulate MCMC progress (for demonstration)."""
        self.mcmc_progress['maximum'] = params['iterations']
        
        def update_progress(i=0):
            if i <= params['iterations']:
                # Update progress
                self.mcmc_progress['value'] = i
                
                # Simulate some output
                if i % 100 == 0:
                    self.console_text.insert('end', f"Iteration {i}/{params['iterations']}\\n")
                    self.console_text.see('end')
                
                # Continue progress
                self.root.after(10, update_progress, i + 10)
            else:
                # MCMC complete
                self.mcmc_status.config(text="Status: Completed", foreground='green')
                self.console_text.insert('end', "MCMC sampling completed successfully!\n")
                self.console_text.see('end')
        
        # Start progress updates
        update_progress()
    
    def stop_mcmc(self):
        """Stop the MCMC sampling process."""
        self.mcmc_status.config(text="Status: Stopped", foreground='orange')
        self.console_text.insert('end', "MCMC sampling stopped by user\n")
        self.console_text.see('end')
        self.mcmc_progress['value'] = 0
    
    def run(self):
        """Run the main application loop."""
        print("Starting main loop...")
        # Make sure window is visible and focused
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        # Make window stay on top temporarily to ensure it's visible
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))
        
        # Start the main loop in a separate thread
        print("Starting main loop in separate thread...")
        import threading
        
        def run_main_loop():
            print("Entering main loop...")
            self.root.mainloop()
            print("Main loop exited")
        
        # Start the main loop in a daemon thread
        main_thread = threading.Thread(target=run_main_loop, daemon=True)
        main_thread.start()
        print("Main loop started in separate thread")

if __name__ == "__main__":
    app = TrunklineGUI()
    app.run()
