import tkinter as tk
from tkinter import ttk, messagebox
import os

class DemoTrunklineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trunkline ML Demo")
        
        # Colors
        self.bg_color = '#F0F0F0'
        self.accent_color = '#2E8B57'
        self.text_color = '#333333'
        
        # Configure window
        self.root.geometry("800x600")
        self.root.configure(bg=self.bg_color)
        
        # Create main container
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header = ttk.Label(
            self.main_frame, 
            text="Trunkline ML Pipeline",
            font=('Arial', 16, 'bold'),
            foreground=self.accent_color
        )
        header.pack(pady=10)
        
        # Steps navigation
        self.create_steps_nav()
        
        # Content area
        self.content_frame = ttk.Frame(self.main_frame)
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Show initial content
        self.show_step_content(0)
    
    def create_steps_nav(self):
        """Create the steps navigation bar."""
        steps_frame = ttk.Frame(self.main_frame)
        steps_frame.pack(fill=tk.X, pady=10)
        
        self.steps = ["1. Set Up", "2. Priors", "3. MCMC", "4. Results"]
        self.step_buttons = []
        
        for i, step in enumerate(self.steps):
            btn = ttk.Button(
                steps_frame,
                text=step,
                command=lambda i=i: self.show_step_content(i),
                style=f'Step{1 if i == 0 else ""}.TButton'
            )
            btn.pack(side=tk.LEFT, padx=5)
            self.step_buttons.append(btn)
    
    def show_step_content(self, step_index):
        """Show content for the selected step."""
        # Clear current content
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        # Update button styles
        for i, btn in enumerate(self.step_buttons):
            btn.state(['!pressed', '!disabled'])
            if i == step_index:
                btn.configure(style='Accent.TButton')
            else:
                btn.configure(style='TButton')
        
        # Show step-specific content
        if step_index == 0:  # Set Up
            self.show_setup_content()
        elif step_index == 1:  # Priors
            self.show_priors_content()
        elif step_index == 2:  # MCMC
            self.show_mcmc_content()
        else:  # Results
            self.show_results_content()
    
    def show_setup_content(self):
        """Show setup step content."""
        frame = ttk.LabelFrame(self.content_frame, text="Data Input", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # File selection
        ttk.Label(frame, text="Input File:").pack(anchor='w', pady=5)
        file_frame = ttk.Frame(frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        
        # Parameters
        ttk.Label(frame, text="Parameters:").pack(anchor='w', pady=(15, 5))
        
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Iterations:").grid(row=0, column=0, sticky='e', padx=5, pady=2)
        self.iter_var = tk.StringVar(value="1000")
        ttk.Entry(param_frame, textvariable=self.iter_var, width=10).grid(row=0, column=1, sticky='w', pady=2)
        
        ttk.Label(param_frame, text="Chains:").grid(row=1, column=0, sticky='e', padx=5, pady=2)
        self.chains_var = tk.StringVar(value="4")
        ttk.Entry(param_frame, textvariable=self.chains_var, width=10).grid(row=1, column=1, sticky='w', pady=2)
    
    def show_priors_content(self):
        """Show priors configuration content."""
        frame = ttk.LabelFrame(self.content_frame, text="Priors Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Simple priors table
        columns = ("Parameter", "Prior Type", "Parameters")
        tree = ttk.Treeview(frame, columns=columns, show='headings', height=5)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)
        
        # Sample data
        priors = [
            ("alpha", "Normal", "μ=0, σ=1"),
            ("beta", "Normal", "μ=0, σ=1"),
            ("sigma", "Exponential", "λ=1")
        ]
        
        for prior in priors:
            tree.insert('', tk.END, values=prior)
        
        tree.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add/Edit buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="Add Prior").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Edit Prior").pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Remove Prior").pack(side=tk.LEFT, padx=2)
    
    def show_mcmc_content(self):
        """Show MCMC configuration and controls."""
        frame = ttk.LabelFrame(self.content_frame, text="MCMC Configuration", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # MCMC parameters
        params = [
            ("Warmup Samples:", "1000"),
            ("Thinning:", "1"),
            ("Adapt Delta:", "0.8"),
            ("Max Tree Depth:", "10")
        ]
        
        for i, (label, default) in enumerate(params):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky='e', padx=5, pady=2)
            ttk.Entry(frame, width=10).grid(row=i, column=1, sticky='w', pady=2)
            ttk.Label(frame, text=default).grid(row=i, column=2, sticky='w', padx=5, pady=2)
        
        # Control buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=len(params), column=0, columnspan=3, pady=15)
        
        ttk.Button(btn_frame, text="Run MCMC", style='Accent.TButton', 
                  command=self.run_mcmc).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_mcmc).pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(
            frame, 
            orient='horizontal', 
            length=300, 
            mode='determinate',
            variable=self.progress_var
        ).grid(row=len(params)+1, column=0, columnspan=3, pady=10, sticky='ew')
        
        # Console output
        console = tk.Text(frame, height=8, wrap=tk.WORD, bg='white', fg='black')
        console.grid(row=len(params)+2, column=0, columnspan=3, sticky='nsew', pady=5)
        
        # Configure grid weights
        frame.grid_rowconfigure(len(params)+2, weight=1)
        frame.grid_columnconfigure(2, weight=1)
    
    def show_results_content(self):
        """Show results and visualizations."""
        frame = ttk.LabelFrame(self.content_frame, text="Results", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placeholder for results visualization
        ttk.Label(
            frame, 
            text="Results will be displayed here",
            foreground='gray',
            font=('Arial', 12)
        ).pack(expand=True)
        
        # Results summary
        results_frame = ttk.Frame(frame)
        results_frame.pack(fill=tk.X, pady=10)
        
        metrics = [
            ("R-hat", "1.002"),
            ("ESS", "1250"),
            ("Divergences", "0")
        ]
        
        for i, (label, value) in enumerate(metrics):
            ttk.Label(results_frame, text=f"{label}:", font=('Arial', 9, 'bold')).grid(row=0, column=i*2, padx=5)
            ttk.Label(results_frame, text=value).grid(row=0, column=i*2+1, padx=5, sticky='w')
    
    def browse_file(self):
        """Open file dialog to select input file."""
        filetypes = (
            ('CSV files', '*.csv'),
            ('All files', '*.*')
        )
        filename = tk.filedialog.askopenfilename(
            title='Open file',
            filetypes=filetypes
        )
        if filename:
            self.file_var.set(filename)
    
    def run_mcmc(self):
        """Start MCMC sampling."""
        self.status_var.set("Running MCMC...")
        self.progress_var.set(0)
        self.update_progress()
    
    def stop_mcmc(self):
        """Stop MCMC sampling."""
        self.status_var.set("MCMC stopped")
        self.progress_var.set(0)
    
    def update_progress(self, value=0):
        """Update progress bar."""
        if value <= 100:
            self.progress_var.set(value)
            self.root.after(50, self.update_progress, value + 1)
        else:
            self.status_var.set("MCMC completed")

def configure_styles():
    """Configure ttk styles."""
    style = ttk.Style()
    
    # Configure the main window
    style.configure('.', background='#F0F0F0')
    
    # Configure buttons
    style.configure('TButton', padding=6)
    
    # Configure accent button
    style.configure('Accent.TButton', 
                   background='#2E8B57', 
                   foreground='white',
                   font=('Arial', 10, 'bold'))
    
    # Configure labels
    style.configure('TLabel', background='#F0F0F0')
    style.configure('TFrame', background='#F0F0F0')
    style.configure('TLabelframe', background='#F0F0F0')
    style.configure('TLabelframe.Label', background='#F0F0F0')

if __name__ == "__main__":
    root = tk.Tk()
    configure_styles()
    app = DemoTrunklineGUI(root)
    root.mainloop()
