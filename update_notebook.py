import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
import os

def update_notebook():
    # Define paths
    notebook_path = os.path.join("notebooks", "data.ipynb")
    data_path = os.path.join("..", "Trunkline", "data", "Limonene_data.csv")
    
    # Create the updated code
    code = f"""import pandas as pd
import os

# Define the path to the data file
data_path = os.path.join("..", "Trunkline", "data", "Limonene_data.csv")

# Load the data
try:
    df = pd.read_csv(data_path)
    print("Data loaded successfully!")
    print("\nColumns in the dataset:")
    print(df.columns)
    print("\nFirst few rows:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Could not find the data file at {{data_path}}")
    print("Current working directory:", os.getcwd())
    print("\nPlease make sure the file exists at the specified location.")
"""
    
    # Create a new notebook
    nb = new_notebook()
    
    # Add cells to the notebook
    nb.cells.append(new_markdown_cell("# Data Loading and Exploration"))
    nb.cells.append(new_markdown_cell("This notebook demonstrates loading and exploring the Limonene dataset."))
    nb.cells.append(new_code_cell(code))
    
    # Write the notebook to disk
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print(f"Notebook updated successfully at {notebook_path}")

if __name__ == "__main__":
    update_notebook()
