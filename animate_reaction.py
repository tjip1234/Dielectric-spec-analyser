#!/usr/bin/env python3
"""
Animate reaction progression through S1P files with polynomial trendlines
Creates two separate 10-second GIFs: one for |S11| (dB) and one for phase
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import skrf as rf

# Configuration
DATA_FOLDER = Path("reaction/times")
POLY_ORDER = 7  # Polynomial order for smooth fitting
FPS = 30  # Frames per second
DURATION = 10  # seconds
TOTAL_FRAMES = FPS * DURATION
INTERPOLATION_STEPS = TOTAL_FRAMES // 6  # Steps between each file
FREQ_MIN_MHZ = 60  # Start from 60 MHz
FREQ_MAX_GHZ = 2.95  # Upper limit in GHz

def parse_timestamp(filename_stem):
    """Parse timestamp from filename like '21-45' to minutes"""
    try:
        parts = filename_stem.split('-')
        hours = int(parts[0])
        minutes = int(parts[1])
        return hours * 60 + minutes  # Convert to total minutes
    except:
        return 0

def load_s1p_files():
    """Load all S1P files in chronological order and filter by frequency"""
    files = sorted(DATA_FOLDER.glob("*.s1p"))
    
    data = []
    for f in files:
        network = rf.Network(str(f))
        freq = network.f / 1e9  # Convert to GHz
        s11 = network.s[:, 0, 0]
        
        # Filter by frequency (keep >= 60 MHz and <= 2.95 GHz)
        freq_mask = (freq >= (FREQ_MIN_MHZ / 1000)) & (freq <= FREQ_MAX_GHZ)
        freq = freq[freq_mask]
        s11 = s11[freq_mask]
        
        # Calculate magnitude (dB) and phase
        s11_db = 20 * np.log10(np.maximum(np.abs(s11), 1e-12))
        s11_phase = np.angle(s11, deg=True)
        
        # Parse timestamp
        timestamp_minutes = parse_timestamp(f.stem)
        
        data.append({
            'name': f.stem,
            'timestamp': timestamp_minutes,
            'freq': freq,
            's11_db': s11_db,
            's11_phase': s11_phase
        })
    
    print(f"Loaded {len(data)} files ({FREQ_MIN_MHZ} MHz - {FREQ_MAX_GHZ} GHz):")
    for d in data:
        hours = d['timestamp'] // 60
        mins = d['timestamp'] % 60
        print(f"  - {d['name']} ({hours:02d}:{mins:02d})")
    
    return data

def fit_polynomial(freq, values, order):
    """Fit polynomial and return fitted values"""
    valid_mask = np.isfinite(values)
    if not np.any(valid_mask):
        return values
    
    freq_valid = freq[valid_mask]
    values_valid = values[valid_mask]
    
    coeffs = np.polyfit(freq_valid, values_valid, order)
    fitted = np.polyval(coeffs, freq)
    
    return fitted

def interpolate_data(data1, data2, alpha):
    """
    Interpolate between two datasets
    alpha: 0 = fully data1, 1 = fully data2
    """
    # Use the frequency grid from data1 (they should be the same)
    freq = data1['freq']
    
    # Fit polynomials to both datasets
    db1_fit = fit_polynomial(data1['freq'], data1['s11_db'], POLY_ORDER)
    db2_fit = fit_polynomial(data2['freq'], data2['s11_db'], POLY_ORDER)
    
    phase1_fit = fit_polynomial(data1['freq'], data1['s11_phase'], POLY_ORDER)
    phase2_fit = fit_polynomial(data2['freq'], data2['s11_phase'], POLY_ORDER)
    
    # Interpolate between fitted curves
    db_interp = (1 - alpha) * db1_fit + alpha * db2_fit
    phase_interp = (1 - alpha) * phase1_fit + alpha * phase2_fit
    
    return freq, db_interp, phase_interp

def create_animation_db(data_files):
    """Create animated GIF for S11 magnitude (dB)"""
    print("\n" + "="*60)
    print("Creating S11 Magnitude (dB) Animation")
    print("="*60)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get initial state for dotted reference line
    initial_freq = data_files[0]['freq']
    initial_db_fit = fit_polynomial(initial_freq, data_files[0]['s11_db'], POLY_ORDER)
    
    # Plot initial state as dotted line
    ax.plot(initial_freq, initial_db_fit, 'k:', linewidth=2, 
            label=f'Initial ({data_files[0]["name"]})', alpha=0.5)
    
    # Initialize empty line for current state
    line, = ax.plot([], [], 'b-', linewidth=2.5, label='Current State')
    
    # Setup axes
    freq_min = data_files[0]['freq'][0]
    freq_max = data_files[0]['freq'][-1]
    
    # Get data ranges for y-limits
    all_db = np.concatenate([d['s11_db'] for d in data_files])
    db_min, db_max = np.nanmin(all_db) - 5, np.nanmax(all_db) + 5
    
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(db_min, db_max)
    ax.set_xlabel('Frequency (GHz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('|S₁₁| (dB)', fontweight='bold', fontsize=12)
    ax.set_title('Reaction Progression - S11 Magnitude (dB)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Time label
    time_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=14, 
                       fontweight='bold', color='darkblue',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        # Determine which two files we're interpolating between
        files_per_frame = TOTAL_FRAMES / (len(data_files) - 1)
        file_idx = int(frame / files_per_frame)
        
        if file_idx >= len(data_files) - 1:
            file_idx = len(data_files) - 2
        
        # Calculate interpolation alpha
        local_frame = frame - (file_idx * files_per_frame)
        alpha = local_frame / files_per_frame
        alpha = np.clip(alpha, 0, 1)
        
        # Get interpolated data
        data1 = data_files[file_idx]
        data2 = data_files[file_idx + 1]
        freq, db_interp, _ = interpolate_data(data1, data2, alpha)
        
        # Update line
        line.set_data(freq, db_interp)
        
        # Interpolate timestamp
        time1 = data1['timestamp']
        time2 = data2['timestamp']
        current_time = time1 + alpha * (time2 - time1)
        hours = int(current_time // 60)
        mins = int(current_time % 60)
        secs = int((current_time % 1) * 60)
        
        time_label = f"Time: {hours:02d}:{mins:02d}:{secs:02d}"
        time_text.set_text(time_label)
        
        return line, time_text
    
    print(f"Animating {TOTAL_FRAMES} frames...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=TOTAL_FRAMES, interval=1000/FPS, blit=True
    )
    
    # Save as GIF
    output_file = 'reaction_db.gif'
    print(f"Saving to {output_file}...")
    
    writer = animation.PillowWriter(fps=FPS)
    anim.save(output_file, writer=writer, dpi=100)
    
    print(f"✓ Saved: {output_file} ({Path(output_file).stat().st_size / 1024 / 1024:.1f} MB)")
    
    plt.close()
    return output_file


def create_animation_phase(data_files):
    """Create animated GIF for S11 phase"""
    print("\n" + "="*60)
    print("Creating S11 Phase Animation")
    print("="*60)
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get initial state for dotted reference line
    initial_freq = data_files[0]['freq']
    initial_phase_fit = fit_polynomial(initial_freq, data_files[0]['s11_phase'], POLY_ORDER)
    
    # Plot initial state as dotted line
    ax.plot(initial_freq, initial_phase_fit, 'k:', linewidth=2, 
            label=f'Initial ({data_files[0]["name"]})', alpha=0.5)
    
    # Initialize empty line for current state
    line, = ax.plot([], [], 'r-', linewidth=2.5, label='Current State')
    
    # Setup axes
    freq_min = data_files[0]['freq'][0]
    freq_max = data_files[0]['freq'][-1]
    
    # Get data ranges for y-limits
    all_phase = np.concatenate([d['s11_phase'] for d in data_files])
    phase_min, phase_max = np.nanmin(all_phase) - 10, np.nanmax(all_phase) + 10
    
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(phase_min, phase_max)
    ax.set_xlabel('Frequency (GHz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('∠S₁₁ (degrees)', fontweight='bold', fontsize=12)
    ax.set_title('Reaction Progression - S11 Phase', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Time label
    time_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, 
                       ha='right', va='top', fontsize=14, 
                       fontweight='bold', color='darkred',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    def animate(frame):
        # Determine which two files we're interpolating between
        files_per_frame = TOTAL_FRAMES / (len(data_files) - 1)
        file_idx = int(frame / files_per_frame)
        
        if file_idx >= len(data_files) - 1:
            file_idx = len(data_files) - 2
        
        # Calculate interpolation alpha
        local_frame = frame - (file_idx * files_per_frame)
        alpha = local_frame / files_per_frame
        alpha = np.clip(alpha, 0, 1)
        
        # Get interpolated data
        data1 = data_files[file_idx]
        data2 = data_files[file_idx + 1]
        freq, _, phase_interp = interpolate_data(data1, data2, alpha)
        
        # Update line
        line.set_data(freq, phase_interp)
        
        # Interpolate timestamp
        time1 = data1['timestamp']
        time2 = data2['timestamp']
        current_time = time1 + alpha * (time2 - time1)
        hours = int(current_time // 60)
        mins = int(current_time % 60)
        secs = int((current_time % 1) * 60)
        
        time_label = f"Time: {hours:02d}:{mins:02d}:{secs:02d}"
        time_text.set_text(time_label)
        
        return line, time_text
    
    print(f"Animating {TOTAL_FRAMES} frames...")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=TOTAL_FRAMES, interval=1000/FPS, blit=True
    )
    
    # Save as GIF
    output_file = 'reaction_phase.gif'
    print(f"Saving to {output_file}...")
    
    writer = animation.PillowWriter(fps=FPS)
    anim.save(output_file, writer=writer, dpi=100)
    
    print(f"✓ Saved: {output_file} ({Path(output_file).stat().st_size / 1024 / 1024:.1f} MB)")
    
    plt.close()
    return output_file

if __name__ == '__main__':
    print("="*60)
    print("Reaction Progression Animator")
    print("="*60)
    
    print("\nLoading data...")
    data_files = load_s1p_files()
    
    if len(data_files) < 2:
        print("Error: Need at least 2 files to animate!")
        exit(1)
    
    # Create both animations
    file1 = create_animation_db(data_files)
    file2 = create_animation_phase(data_files)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"\nCreated 2 animations:")
    print(f"  1. {file1} - S11 Magnitude (dB)")
    print(f"  2. {file2} - S11 Phase")
    print(f"\nBoth show:")
    print(f"  - Initial state ({data_files[0]['name']}) as dotted line")
    print(f"  - Smooth interpolation through all {len(data_files)} measurements")
    print(f"  - Polynomial order: {POLY_ORDER}")
    print(f"  - Frequency range: {FREQ_MIN_MHZ} MHz - {data_files[0]['freq'][-1]*1000:.0f} MHz")
