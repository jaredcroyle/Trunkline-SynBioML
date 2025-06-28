from setuptools import setup
import py2app

APP = ['src/gui/gui.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    'iconfile': 'src/gui/assets/logo.png',
    'plist': {
        'LSUIElement': False,
        'CFBundleName': 'Trunkline',
        'CFBundleDisplayName': 'Trunkline',
        'CFBundleIdentifier': 'com.nonatalks.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSRequiresAquaSystemAppearance': False,
    },
}

setup(
    app=APP,
    name='Trunkline',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'scipy>=1.7.0',
        'joblib>=1.0.1',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.2',
        'pyyaml>=5.4.1',
        'Pillow>=8.4.0'
    ],
)
