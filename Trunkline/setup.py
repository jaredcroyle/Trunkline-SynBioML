from setuptools import setup, find_packages

setup(
    name="trunkline",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'scipy>=1.7.0',
        'joblib>=1.0.1',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.2',
        'shap>=0.40.0',
        'tqdm>=4.62.0',
    ],
    python_requires='>=3.8',
)
