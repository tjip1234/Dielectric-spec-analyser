# S1P GUI Application

A modular, interactive GUI application for analyzing S1P (reflection) dielectric measurements.

## Features

- **Multi-file support**: Load and compare multiple S1P files simultaneously
- **Live plotting**: Interactive plots with real-time updates
- **Flexible metrics**: Choose from S11 magnitude/phase, permittivity (real/imaginary/magnitude)
- **Frequency ranges**: Predefined ranges (low/mid/full) or custom frequency selection
- **Parameter calculation**: Automatic calculation of slope, integral, means, and other statistics
- **Professional visualization**: Scientific-quality plots with customizable scales

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements_gui.txt
```

2. Run the application:
```bash
python run_s1p_gui.py
```

## Project Structure

```
s1p_gui/
├── __init__.py          # Package initialization
├── constants.py         # Physical constants and configuration
├── formulas.py          # Dielectric calculations and analysis
├── data_loader.py       # File loading and data management
└── gui_main.py          # Main GUI window and controls
```

## Usage

1. **Add Files**: Click "Add File" to load S1P files
2. **Select Metric**: Choose which property to plot (ε', ε'', |S11|, etc.)
3. **Set Frequency Range**: Select predefined or custom frequency range
4. **View Statistics**: See calculated parameters (slope, integral, means, etc.)
5. **Toggle Files**: Check/uncheck files to show/hide from plot
6. **Interactive Plot**: Use matplotlib toolbar to zoom, pan, and save plots

## Calculated Parameters

- **Basic Statistics**: Mean, median, standard deviation, min/max, range
- **Slope**: Linear fit slope (per GHz)
- **Integral**: Area under curve (trapezoidal integration)
- **Sub-range Means**: For 0.5-1 GHz, 1-2 GHz, 2-3 GHz ranges
- **Ratio**: Low/High frequency ratio

## Keyboard Shortcuts

- Use standard matplotlib navigation (pan, zoom, home, save)
- File list supports multi-select (Ctrl+Click, Shift+Click)

## Future Enhancements

- Export statistics to CSV
- Baseline subtraction mode
- Cole-Cole plot view
- Auto-detect stable regions
- Batch processing
