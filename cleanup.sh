#!/bin/bash

# Remove Python cache and compiled files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.py[co]" -delete
find . -type d -name "*.pyc" -delete

# Remove build and distribution directories
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Remove Jupyter notebook checkpoints
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Remove test and coverage files
rm -f .coverage
rm -rf htmlcov/
rm -rf .pytest_cache/

echo "Cleanup complete!"
