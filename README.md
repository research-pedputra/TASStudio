# TAS Studio

TAS Studio is a Python desktop app for transient absorption spectroscopy (TAS). It loads TA CSV matrices, applies preprocessing, renders heatmaps, spectral profiles and GSB lifetime plots, and saves editable projects and publication-ready figures. The interface is built with PySide6 and Matplotlib.

## Start the app

Install Python 3.10 or newer. In PowerShell, open this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

The application is launched from the Python terminal with `python main.py`. To open an existing project from the terminal, pass its path:

```powershell
python main.py ".\data\S40.tas"
```

If an import reports a missing library, install the listed dependencies with `python -m pip install -r requirements.txt` using the same Python environment that runs `main.py`.

## Load data and make plots

1. Choose **File → Add data** and select one or more CSV files. The first row contains delay times, the first column contains wavelengths, and the remaining cells contain signal values.
2. Select a plot tab: **TA heatmap**, **Spectral profiles**, or **GSB lifetimes**.
3. Adjust the settings in the Figure settings panel and select **Render figure**. Each plot tab keeps its own settings.
4. Use **File → Export figure** to save the active plot as PNG, PDF, SVG, TIFF, or EPS.

The `data/` folder includes example CSV data and `.tas` projects that you can open and explore.

## Features

- Configure heatmap color maps and color limits, spectral snapshot times and colors, and lifetime fitting ranges and components.
- Apply per-file preprocessing, including background correction, chirp correction, smoothing, scaling, and row limits.
- Edit embedded matrix data, add annotations, change plot text and typography, and inspect advanced Matplotlib artist properties.
- Arrange files in a matrix layout and set figure size, spacing, margins, and export resolution.
- Save `.tas` projects with data, plot settings, edits, annotations, and imported fonts embedded, so the original CSV files are not needed to reopen a saved project.

For detailed controls, open the **Help** menu in the app. The main source files are `main.py` (entry point), `tas_app.py` (interface), `tas_core.py` (project/plot integration), and `mytas.py` (data processing and plotting).
