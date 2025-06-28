# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Get the project root directory
project_root = os.getcwd()

# Collect all Python files in the src directory
src_files = []
for root, dirs, files in os.walk(os.path.join(project_root, 'src')):
    for file in files:
        if file.endswith('.py'):
            src_files.append(os.path.join(root, file))

# Collect data files for matplotlib and other packages
matplotlib_data = collect_data_files('matplotlib')
shap_data = collect_data_files('shap')

# Add the source files to datas
datas = [
    ('src', 'src'),
    *[(src, os.path.dirname(os.path.relpath(src, project_root))) for src in src_files],
    *matplotlib_data,
    *shap_data
]

# Hidden imports
hiddenimports = [
    'shap',
    'shap._explanation',
    'shap.utils',
    'shap.plots',
    'shap.plots._beeswarm',
    'shap.plots._bar',
    'shap.plots._decision',
    'shap.plots._force',
    'shap.plots._image',
    'shap.plots._monitoring',
    'shap.plots._partial_dependence',
    'shap.plots._scatter',
    'shap.plots._text',
    'shap.plots._violin',
    'shap.plots._waterfall',
    'shap.plots.colors',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.neighbors._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils',
    'sklearn.neural_network._multilayer_perceptron',
    'sklearn.neural_network._base',
    'sklearn.utils._cython_blas',
    'scipy._lib.messagestream',
    'scipy._cyutility',
    'scipy.sparse._csparsetools',
    'pandas',
    'numpy',
    'matplotlib',
    'PIL',
    'PIL._tkinter_finder'
]

# Add any additional hidden imports from your project
hiddenimports.extend(collect_submodules('src'))

# Analysis
block_cipher = None

a = Analysis(
    ['trunkline_app.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Trunkline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['trunkline.icns'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Trunkline',
)
app = BUNDLE(
    coll,
    name='Trunkline.app',
    icon='trunkline.icns',
    bundle_identifier=None,
)
