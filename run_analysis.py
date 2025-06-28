#!/usr/bin/env python3
"""
NonaTalks25ML - Biological Strain Analysis Tool

A simple launcher for the biological strain analysis pipeline.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the analysis pipeline."""
    print("Launching NonaTalks25ML Analysis Pipeline...")
    
    # Check if we're in the right directory
    if not (Path("biolab").exists() and Path("biolab/main.py").exists()):
        print("ERROR: Could not find the biolab directory. Please run this from the project root.")
        return 1
    
    # Run the main analysis
    try:
        print("Starting analysis...")
        print("This may take a few minutes...")
        print("Check biolab/results/ for output files when complete.")
        print("-" * 50)
        
        # Run the main script
        result = subprocess.run(
            [sys.executable, "biolab/main.py"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            check=True
        )
        
        print("\nAnalysis complete!")
        print(f"Results saved to: {os.path.abspath('biolab/results')}")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"\nError running analysis: {e}")
        return 1
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
