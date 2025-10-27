# S1P GUI Application - Quick Start Guide

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements_gui.txt
```

2. **Launch the application:**
```bash
python run_s1p_gui.py
```

## Basic Workflow

### 1. Loading Files
- Click **"Add File"** button
- Navigate to your S1P file(s)
- Multiple files will be plotted with different colors
- Check/uncheck files to show/hide them

### 2. Selecting What to Plot
**Available Metrics:**
- |S₁₁| (dB) - Reflection coefficient magnitude in decibels
- ∠S₁₁ (degrees) - Reflection phase
- Real Permittivity (ε') - Real part of complex permittivity
- Imaginary Permittivity (ε'') - Imaginary part (loss)
- Complex Permittivity Magnitude (|ε*|) - Total permittivity

### 3. Frequency Ranges
**Predefined:**
- **Low:** 10 MHz - 500 MHz
- **Mid:** 500 MHz - 3 GHz (includes sub-range analysis)
- **Full Sweep:** All available frequencies

**Custom:**
- Select "Custom Range" from dropdown
- Adjust Min/Max frequency sliders

### 4. Plot Options
- **Log X-axis:** Logarithmic frequency scale (useful for full sweep)
- **Log Y-axis:** Logarithmic value scale (useful for magnitude plots)
- **Matplotlib Toolbar:** Pan, zoom, save plots

### 5. Statistics Panel
Shows real-time calculated parameters:
- **Basic stats:** Mean, median, std dev, min/max, range
- **Slope:** Linear fit over frequency range (per GHz)
- **Integral:** Total area under curve
- **Sub-ranges:** Mean values for 0.5-1, 1-2, 2-3 GHz (when in mid-range)
- **Ratio:** Low/High frequency ratio

Click **"Refresh Statistics"** to update if needed.

## Tips & Tricks

### Comparing Samples
1. Load baseline/solvent file first (will appear in gray)
2. Load sample file(s) (will appear in color)
3. Use checkboxes to toggle visibility
4. Compare statistics in the panel below

### Finding Stable Regions
1. Plot |S₁₁| (dB) with full sweep
2. Enable Log X-axis
3. Look for flat regions in the curve
4. Switch to custom range to zoom into specific regions

### Analyzing Dielectric Loss
1. Select "Imaginary Permittivity (ε'')" metric
2. Set frequency range to "Mid" (500 MHz - 3 GHz)
3. Check statistics for:
   - **Slope**: Rate of change with frequency
   - **Sub-range means**: Loss at different frequencies
   - **Integral**: Total loss over frequency range

### Export-Ready Plots
1. Use matplotlib toolbar's save button
2. Plots are saved at 300 DPI (publication quality)
3. Supports PNG, PDF, SVG formats

## Keyboard Shortcuts
- **Ctrl+O**: Open file (when implemented)
- **Delete**: Remove selected files (when implemented)
- **Matplotlib navigation**:
  - Pan: Click and drag with mouse
  - Zoom: Right-click and drag box
  - Home: Reset view
  - Back/Forward: Navigate zoom history

## Common Use Cases

### Use Case 1: Quick Quality Check
```
1. Load S1P file
2. Select "Full Sweep" range
3. Plot |S₁₁| (dB)
4. Check for anomalies or noise
```

### Use Case 2: Material Characterization
```
1. Load sample and solvent files
2. Select "Mid" range (500 MHz - 3 GHz)
3. Plot ε' and ε'' separately
4. Compare statistics between files
5. Note slope and integral values
```

### Use Case 3: Frequency Sweep Analysis
```
1. Load file
2. Select "Full Sweep"
3. Enable Log X-axis
4. Plot different metrics to find:
   - Resonances (peaks in |S₁₁|)
   - Dispersions (changes in ε')
   - Relaxations (peaks in ε'')
```

## Project Structure
```
s1p_gui/
├── __init__.py       - Package initialization
├── constants.py      - Physical constants, plot settings
├── formulas.py       - All calculations (dielectric, statistics, slope, integral)
├── data_loader.py    - File I/O and data management
├── gui_main.py       - Main GUI window
└── README.md         - Detailed documentation

run_s1p_gui.py        - Application launcher
example_usage.py      - Programmatic usage examples
requirements_gui.txt  - Python dependencies
```

## Troubleshooting

**Problem: Import errors when running**
- Solution: Install requirements: `pip install -r requirements_gui.txt`

**Problem: "No module named PyQt5"**
- Solution: `pip install PyQt5`

**Problem: "No module named skrf"**
- Solution: `pip install scikit-rf`

**Problem: Plot not updating**
- Solution: Click "Refresh Statistics" or toggle file checkboxes

**Problem: Statistics show NaN values**
- Solution: Check that frequency range contains data points

**Problem: Slow performance with many files**
- Solution: Use frequency filtering to reduce data points

## Advanced: Programmatic Usage

See `example_usage.py` for examples of using the modules without the GUI:
```python
from s1p_gui.data_loader import DataManager
from s1p_gui.formulas import calculate_statistics

manager = DataManager()
data_file = manager.add_file(Path("your_file.s1p"))
stats = calculate_statistics(data_file.get_data(), 'epsilon_prime')
print(f"Mean ε': {stats['mean']:.4e}")
```

## Next Steps: S2P GUI
Once comfortable with S1P GUI, the S2P version will have similar interface but for transmission (S21) measurements.

## Support
For issues or questions, check:
1. This guide
2. README.md in s1p_gui/ folder
3. Comments in source code
