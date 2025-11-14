# Reaction GIF Animator

Creates animated GIFs from NanoVNA CSV time-series recordings showing reaction progression.

## Features

- **Pre-smoothing**: Savitzky-Golay filter removes noise spikes before trendline fitting
- **Polynomial Trendlines**: Smooth curves (no noisy raw data points)
- **Phase Unwrapping**: Continuous phase without ±180° jumps  
- **Log Time Scale**: More frames for early reaction (when changes are fastest)
- **Multiple Metrics**: S11 magnitude, phase, and permittivity
- **Reference Line**: Initial state shown as dotted line

## Quick Start

```bash
# Create all GIFs from a CSV recording
python3 create_reaction_gifs.py reactions/recording_20251112_234332.csv

# Custom settings (poly_order, fps, duration, smooth_window)
python3 create_reaction_gifs.py recording.csv 7 30 10 11
#                                              ^  ^  ^  ^
#                                              |  |  |  smoothing window (removes noise)
#                                              |  |  duration (seconds)
#                                              |  fps
#                                              polynomial order
```

## Output Files

The script creates 4 GIF files:

1. `recording_s11_mag_db.gif` - S11 magnitude in dB
2. `recording_s11_phase_unwrapped.gif` - S11 phase (unwrapped)
3. `recording_epsilon_prime.gif` - Real permittivity (ε')
4. `recording_epsilon_double_prime.gif` - Imaginary permittivity (ε")

## CSV Format

Expected format from NanoVNA recordings:

```csv
# NanoVNA Recording
# Start Time:,2025-11-12 22:46:39
# End Time:,2025-11-12 23:43:30
# Total Sweeps:,899
# Interval:,1.0 seconds

# Sweep 1 - Time: 22:46:39
Frequency (Hz),S11 Real,S11 Imag,S11 Mag (dB),S11 Phase (deg),...
200000000.0,-0.204094,-0.011478,-13.78968,-176.78114,...
...
```

## Python API

```python
from s1p_gui.gif_animator import ReactionAnimator

# Create animator with custom smoothing
animator = ReactionAnimator('recording.csv', poly_order=7, smooth_window=11)
animator.load_csv_data()

# Create single metric GIF
animator.create_gif(
    metric='s11_phase_unwrapped',
    fps=30,
    duration=10,
    freq_range_ghz=(0.5, 3.0)  # Optional frequency filter
)

# Create all GIFs
animator.create_all_gifs(fps=30, duration=10)
```

## Parameters

### Smoothing Window
- **Default**: 11
- **Range**: 5-51 (must be odd)
- **Smaller** (5-9): Less smoothing, preserves more detail but may show noise
- **Larger** (15-51): More aggressive smoothing, removes more noise spikes
- Uses Savitzky-Golay filter to remove noise before polynomial fitting

### Polynomial Order
- **Default**: 7
- **Range**: 3-15
- **Lower** (3-5): Less smooth, follows data more closely
- **Higher** (9-15): Very smooth, may miss small features

### FPS (Frames Per Second)
- **Default**: 30
- **Range**: 10-60
- **Lower**: Smaller file, choppier animation
- **Higher**: Smoother animation, larger file

### Duration
- **Default**: 10 seconds
- **Range**: 5-30 seconds
- **Longer**: More detail, larger file

## Log Time Scale

The animator uses logarithmic time scaling, meaning:
- **Early reaction** (first minutes): Many frames, high detail
- **Later reaction** (last hours): Fewer frames, less detail

This is ideal because reactions typically change fastest at the beginning.

Time distribution example for 10-second GIF:
- First 1 minute: ~3 seconds of animation
- Next 10 minutes: ~3 seconds of animation  
- Next 100 minutes: ~2 seconds of animation
- Remaining time: ~2 seconds of animation

## Troubleshooting

### No matplotlib/pillow/scipy
```bash
pip install matplotlib pillow scipy
```

### Out of memory
Reduce FPS or duration:
```bash
python3 create_reaction_gifs.py recording.csv 7 20 8
```

### Too much smoothing / losing detail
Reduce smooth_window:
```bash
python3 create_reaction_gifs.py recording.csv 7 30 10 7
```

### Still seeing noise spikes
Increase smooth_window:
```bash
python3 create_reaction_gifs.py recording.csv 7 30 10 21
```

### Phase looks wrong
Phase is automatically unwrapped. If you want wrapped phase [-180, 180], 
modify the code to use `s11_phase_deg` instead of `s11_phase_unwrapped`.

## Integration with GUI

To add to the S1P GUI, the animator can be imported and called:

```python
from s1p_gui.gif_animator import ReactionAnimator

# In your GUI code
def on_create_gif_clicked():
    animator = ReactionAnimator(self.csv_path)
    animator.load_csv_data()
    animator.create_all_gifs()
```

A button and dialog could be added to `gui_main.py` for interactive GIF creation.
