#!/usr/bin/env python3
"""Quick script to plot slope evolution over time from NanoVNA CSV recording."""

import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path

# Load CSV data
csv_path = Path(__file__).parent / 'recording_20251112_234332.csv'

# Parse the CSV file
sweeps = []
current_sweep = None
sweep_times = []

with open(csv_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('# Sweep'):
            match = re.search(r'Sweep (\d+) - Time: (\d+:\d+:\d+)', line)
            if match:
                sweep_times.append(match.group(2))
                if current_sweep is not None:
                    sweeps.append(current_sweep)
                current_sweep = {'freq': [], 'db': [], 'phase': []}
        elif line.startswith('Frequency') or line.startswith('#') or not line:
            continue
        elif current_sweep is not None:
            parts = line.split(',')
            if len(parts) >= 5:
                try:
                    current_sweep['freq'].append(float(parts[0]))
                    current_sweep['db'].append(float(parts[3]))
                    current_sweep['phase'].append(float(parts[4]))
                except:
                    pass

if current_sweep is not None:
    sweeps.append(current_sweep)

print(f"Loaded {len(sweeps)} sweeps")

# Convert time to seconds from start
def time_to_seconds(t, start):
    h, m, s = map(int, t.split(':'))
    sh, sm, ss = map(int, start.split(':'))
    return (h - sh) * 3600 + (m - sm) * 60 + (s - ss)

times_seconds = [time_to_seconds(t, sweep_times[0]) for t in sweep_times]

# Calculate poly5 slopes for each sweep
poly_degree = 5
db_slopes = []
phase_slopes = []

for sweep in sweeps:
    freq = np.array(sweep['freq'])
    db = np.array(sweep['db'])
    phase = np.array(sweep['phase'])
    
    # Filter to 0.5 GHz and above
    mask = freq >= 0.5e9
    freq = freq[mask]
    db = db[mask]
    phase = phase[mask]
    
    # Normalize frequency for numerical stability
    freq_norm = (freq - freq.min()) / (freq.max() - freq.min())
    
    # Fit degree 5 polynomials
    db_coeffs = np.polyfit(freq_norm, db, poly_degree)
    phase_coeffs = np.polyfit(freq_norm, np.unwrap(np.deg2rad(phase)), poly_degree)
    
    # Linear slope is coefficient at index -2 (degree 1 term)
    db_slopes.append(db_coeffs[-2])
    phase_slopes.append(phase_coeffs[-2])

# Convert to arrays and apply moving average
db_slopes = np.array(db_slopes)
phase_slopes = np.array(phase_slopes)
times_minutes = np.array(times_seconds) / 60

# Moving average window
window = 10
db_slopes_avg = np.convolve(db_slopes, np.ones(window)/window, mode='valid')
phase_slopes_avg = np.convolve(phase_slopes, np.ones(window)/window, mode='valid')
times_avg = times_minutes[window//2 : len(times_minutes) - window//2 + 1]

# Create plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# DB slope
axes[0].plot(times_avg, db_slopes_avg, 'b-', linewidth=1.0)
axes[0].set_ylabel('S11 dB Linear Slope', fontsize=11)
axes[0].set_title('Slope Evolution Over Time (Polyfit Degree 5, freq > 0.5 GHz , 10-pt avg)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')
axes[0].axhline(y=np.mean(db_slopes_avg), color='b', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(db_slopes_avg):.2f}')
axes[0].legend()

# Phase slope
axes[1].plot(times_avg, phase_slopes_avg, 'r-', linewidth=1.0)
axes[1].set_ylabel('Phase Linear Slope (unwrapped)', fontsize=11)
axes[1].set_xlabel('Time (minutes)', fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')
axes[1].axhline(y=np.mean(phase_slopes_avg), color='r', linestyle='--', alpha=0.5, label=f'Mean: {np.mean(phase_slopes_avg):.2f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(csv_path.parent / 'slope_evolution.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to reactions/slope_evolution.png")
