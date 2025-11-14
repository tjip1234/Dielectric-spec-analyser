"""
GIF Animation Generator for NanoVNA Reaction Recordings

Creates animated GIFs from CSV recordings with:
- Pre-smoothing using Savitzky-Golay filter to remove noise spikes
- Polynomial trendlines (no raw data points)
- Log time scale
- Phase unwrapping
- Multiple metrics visualization

The two-stage smoothing approach:
1. Savitzky-Golay filter removes noise spikes from raw data
2. Polynomial fit creates smooth trendline from cleaned data

This handles noisy measurements effectively while preserving signal features.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime, timedelta
from scipy.signal import savgol_filter


class ReactionAnimator:
    """Generate animated GIFs from NanoVNA CSV recordings"""
    
    def __init__(self, csv_path: str, poly_order: int = 7, smooth_window: int = 11, smooth_polyorder: int = 3):
        """
        Initialize animator
        
        Args:
            csv_path: Path to CSV recording file
            poly_order: Polynomial order for trendline fitting
            smooth_window: Window length for Savitzky-Golay filter (must be odd)
            smooth_polyorder: Polynomial order for Savitzky-Golay filter
        """
        self.csv_path = Path(csv_path)
        self.poly_order = poly_order
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.smooth_polyorder = min(smooth_polyorder, smooth_window - 1)
        self.sweeps_data = []
        self.metadata = {}
        
    def load_csv_data(self) -> None:
        """Load and parse CSV recording file"""
        print(f"Loading {self.csv_path.name}...")
        
        with open(self.csv_path, 'r') as f:
            lines = f.readlines()
        
        # Parse metadata
        for line in lines[:10]:
            if line.startswith('# Start Time:'):
                self.metadata['start_time'] = line.split(',', 1)[1].strip()
            elif line.startswith('# End Time:'):
                self.metadata['end_time'] = line.split(',', 1)[1].strip()
            elif line.startswith('# Total Sweeps:'):
                self.metadata['total_sweeps'] = int(line.split(',', 1)[1].strip())
            elif line.startswith('# Interval:'):
                self.metadata['interval'] = float(line.split(',', 1)[1].split()[0])
        
        # Parse sweeps
        current_sweep = None
        current_time = None
        
        for line in lines:
            # Check for sweep header
            sweep_match = re.match(r'# Sweep (\d+) - Time: (.+)', line)
            if sweep_match:
                if current_sweep is not None:
                    self.sweeps_data.append(current_sweep)
                
                sweep_num = int(sweep_match.group(1))
                time_str = sweep_match.group(2).strip()
                current_time = time_str
                current_sweep = {
                    'sweep_num': sweep_num,
                    'time_str': time_str,
                    'frequencies': [],
                    's11_real': [],
                    's11_imag': [],
                    's11_mag_db': [],
                    's11_phase_deg': []
                }
                continue
            
            # Skip comments and headers
            if line.startswith('#') or line.startswith('Frequency'):
                continue
            
            # Parse data line
            if current_sweep is not None and ',' in line:
                try:
                    parts = line.strip().split(',')
                    if len(parts) >= 5:
                        freq = float(parts[0])
                        s11_real = float(parts[1])
                        s11_imag = float(parts[2])
                        s11_mag_db = float(parts[3])
                        s11_phase_deg = float(parts[4])
                        
                        current_sweep['frequencies'].append(freq)
                        current_sweep['s11_real'].append(s11_real)
                        current_sweep['s11_imag'].append(s11_imag)
                        current_sweep['s11_mag_db'].append(s11_mag_db)
                        current_sweep['s11_phase_deg'].append(s11_phase_deg)
                except (ValueError, IndexError):
                    continue
        
        # Add last sweep
        if current_sweep is not None:
            self.sweeps_data.append(current_sweep)
        
        # Convert lists to numpy arrays and calculate unwrapped phase
        for sweep in self.sweeps_data:
            sweep['frequencies'] = np.array(sweep['frequencies'])
            sweep['s11_real'] = np.array(sweep['s11_real'])
            sweep['s11_imag'] = np.array(sweep['s11_imag'])
            sweep['s11_mag_db'] = np.array(sweep['s11_mag_db'])
            sweep['s11_phase_deg'] = np.array(sweep['s11_phase_deg'])
            
            # Unwrap phase
            phase_rad = np.deg2rad(sweep['s11_phase_deg'])
            phase_unwrapped_rad = np.unwrap(phase_rad)
            sweep['s11_phase_unwrapped'] = np.rad2deg(phase_unwrapped_rad)
            
            # Calculate complex permittivity (simplified model)
            s11_complex = sweep['s11_real'] + 1j * sweep['s11_imag']
            epsilon_complex = ((1 + s11_complex) / (1 - s11_complex)) ** 2
            sweep['epsilon_prime'] = np.real(epsilon_complex)
            sweep['epsilon_double_prime'] = np.imag(epsilon_complex)
        
        print(f"Loaded {len(self.sweeps_data)} sweeps")
        print(f"Frequency range: {self.sweeps_data[0]['frequencies'][0]/1e9:.3f} - "
              f"{self.sweeps_data[0]['frequencies'][-1]/1e9:.3f} GHz")
    
    def calculate_time_elapsed(self, sweep_num: int) -> float:
        """
        Calculate elapsed time in seconds from sweep number
        
        Args:
            sweep_num: Sweep number (1-indexed)
            
        Returns:
            Elapsed time in seconds
        """
        return (sweep_num - 1) * self.metadata.get('interval', 1.0)
    
    def fit_polynomial(self, freq: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Fit polynomial trendline to data with pre-smoothing
        
        Args:
            freq: Frequency array
            values: Values to fit
            
        Returns:
            Fitted values
        """
        valid_mask = np.isfinite(values) & np.isfinite(freq)
        if not np.any(valid_mask):
            return values
        
        freq_valid = freq[valid_mask]
        values_valid = values[valid_mask]
        
        # Apply Savitzky-Golay filter to remove noise spikes before fitting
        if len(values_valid) > self.smooth_window:
            try:
                values_smoothed = savgol_filter(values_valid, self.smooth_window, self.smooth_polyorder)
            except:
                # Fallback to simple moving average if savgol fails
                window = self.smooth_window
                values_smoothed = np.convolve(values_valid, np.ones(window)/window, mode='same')
        else:
            values_smoothed = values_valid
        
        # Fit polynomial to smoothed data
        coeffs = np.polyfit(freq_valid / 1e9, values_smoothed, self.poly_order)
        fitted = np.polyval(coeffs, freq / 1e9)
        
        return fitted
    
    def interpolate_sweeps(self, sweep1: Dict, sweep2: Dict, alpha: float) -> Tuple[np.ndarray, Dict]:
        """
        Interpolate between two sweeps
        
        Args:
            sweep1: First sweep data
            sweep2: Second sweep data
            alpha: Interpolation factor (0 = sweep1, 1 = sweep2)
            
        Returns:
            (frequencies, interpolated_data_dict)
        """
        freq = sweep1['frequencies']
        
        interp_data = {}
        
        for metric in ['s11_mag_db', 's11_phase_unwrapped', 'epsilon_prime', 'epsilon_double_prime']:
            # Fit polynomials
            fit1 = self.fit_polynomial(sweep1['frequencies'], sweep1[metric])
            fit2 = self.fit_polynomial(sweep2['frequencies'], sweep2[metric])
            
            # Interpolate
            interp_data[metric] = (1 - alpha) * fit1 + alpha * fit2
        
        return freq, interp_data
    
    def create_gif(self, 
                   metric: str = 's11_mag_db',
                   output_path: Optional[str] = None,
                   fps: int = 30,
                   duration: int = 10,
                   freq_range_ghz: Optional[Tuple[float, float]] = None,
                   show_initial: bool = True) -> str:
        """
        Create animated GIF for a specific metric
        
        Args:
            metric: Metric to animate ('s11_mag_db', 's11_phase_unwrapped', 'epsilon_prime', 'epsilon_double_prime')
            output_path: Output file path (auto-generated if None)
            fps: Frames per second
            duration: Duration in seconds
            freq_range_ghz: (min, max) frequency range in GHz, or None for full range
            show_initial: Show initial sweep as dotted reference line
            
        Returns:
            Path to created GIF file
        """
        if not self.sweeps_data:
            raise ValueError("No data loaded. Call load_csv_data() first.")
        
        # Setup output path
        if output_path is None:
            metric_name = metric.replace('_', '-')
            output_path = self.csv_path.parent / f"{self.csv_path.stem}_{metric_name}.gif"
        else:
            output_path = Path(output_path)
        
        total_frames = fps * duration
        
        # Setup figure
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Get frequency range
        freq = self.sweeps_data[0]['frequencies'] / 1e9  # Convert to GHz
        if freq_range_ghz:
            freq_mask = (freq >= freq_range_ghz[0]) & (freq <= freq_range_ghz[1])
        else:
            freq_mask = np.ones_like(freq, dtype=bool)
        
        freq_plot = freq[freq_mask]
        
        # Plot initial state if requested
        if show_initial:
            initial_fit = self.fit_polynomial(
                self.sweeps_data[0]['frequencies'],
                self.sweeps_data[0][metric]
            )[freq_mask]
            
            initial_time = self.sweeps_data[0]['time_str']
            ax.plot(freq_plot, initial_fit, 'k:', linewidth=2,
                   label=f'Initial ({initial_time})', alpha=0.5)
        
        # Initialize line
        line, = ax.plot([], [], linewidth=2.5, label='Current State')
        
        # Get data range for y-limits (include initial state if shown)
        all_values = []
        for sweep in self.sweeps_data:
            fitted = self.fit_polynomial(sweep['frequencies'], sweep[metric])[freq_mask]
            all_values.extend(fitted[np.isfinite(fitted)])
        
        # Also include initial fit in y-limit calculation if shown
        if show_initial:
            all_values.extend(initial_fit[np.isfinite(initial_fit)])
        
        y_min = np.percentile(all_values, 0.5)  # Use 0.5 percentile to ensure initial line visible
        y_max = np.percentile(all_values, 99.5)
        y_margin = (y_max - y_min) * 0.15  # Increased margin to 15%
        
        # Setup axes
        ax.set_xlim(freq_plot[0], freq_plot[-1])
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        ax.set_xlabel('Frequency (GHz)', fontweight='bold', fontsize=12)
        
        # Y-axis label based on metric
        ylabel_map = {
            's11_mag_db': '|S₁₁| (dB)',
            's11_phase_unwrapped': '∠S₁₁ (degrees, unwrapped)',
            'epsilon_prime': "ε' (Real Permittivity)",
            'epsilon_double_prime': 'ε" (Imaginary Permittivity)'
        }
        ax.set_ylabel(ylabel_map.get(metric, metric), fontweight='bold', fontsize=12)
        
        title_map = {
            's11_mag_db': 'S11 Magnitude (dB)',
            's11_phase_unwrapped': 'S11 Phase (Unwrapped)',
            'epsilon_prime': 'Real Permittivity',
            'epsilon_double_prime': 'Imaginary Permittivity'
        }
        ax.set_title(f'Reaction Progression - {title_map.get(metric, metric)}',
                    fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Time label with log scale info
        time_text = ax.text(0.98, 0.98, '', transform=ax.transAxes,
                          ha='right', va='top', fontsize=12,
                          fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def init():
            line.set_data([], [])
            time_text.set_text('')
            return line, time_text
        
        def animate(frame):
            # Use log scale for time progression
            # Map frame to log-scaled sweep index
            n_sweeps = len(self.sweeps_data)
            
            # Log scale: early sweeps get more frames
            max_time = self.calculate_time_elapsed(n_sweeps)
            log_max = np.log10(max_time + 1)
            
            # Map frame to log time
            progress = frame / total_frames
            log_time = progress * log_max
            current_time_sec = 10**log_time - 1
            
            # Find corresponding sweep indices
            sweep_times = [self.calculate_time_elapsed(s['sweep_num']) 
                          for s in self.sweeps_data]
            
            # Find bracketing sweeps
            idx2 = np.searchsorted(sweep_times, current_time_sec)
            idx2 = min(idx2, n_sweeps - 1)
            idx1 = max(0, idx2 - 1)
            
            if idx1 == idx2:
                alpha = 0
            else:
                time1 = sweep_times[idx1]
                time2 = sweep_times[idx2]
                alpha = (current_time_sec - time1) / (time2 - time1) if time2 > time1 else 0
            
            alpha = np.clip(alpha, 0, 1)
            
            # Get interpolated data
            freq_full, interp_data = self.interpolate_sweeps(
                self.sweeps_data[idx1],
                self.sweeps_data[idx2],
                alpha
            )
            
            values = interp_data[metric][freq_mask]
            
            # Update line
            line.set_data(freq_plot, values)
            
            # Update time label
            mins = int(current_time_sec // 60)
            secs = int(current_time_sec % 60)
            sweep_num = int(idx1 + alpha * (idx2 - idx1) + 1)
            
            time_label = f"Time: {mins:02d}:{secs:02d}\nSweep: {sweep_num}/{n_sweeps}"
            time_text.set_text(time_label)
            
            return line, time_text
        
        print(f"Creating {metric} animation ({total_frames} frames)...")
        anim = animation.FuncAnimation(
            fig, animate, init_func=init,
            frames=total_frames, interval=1000/fps, blit=True
        )
        
        # Save GIF
        print(f"Saving to {output_path.name}...")
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(output_path), writer=writer, dpi=100)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"✓ Saved: {output_path.name} ({file_size_mb:.1f} MB)")
        
        plt.close()
        return str(output_path)
    
    def create_all_gifs(self, 
                       output_dir: Optional[str] = None,
                       fps: int = 30,
                       duration: int = 10) -> List[str]:
        """
        Create GIFs for all common metrics
        
        Args:
            output_dir: Output directory (uses CSV directory if None)
            fps: Frames per second
            duration: Duration in seconds
            
        Returns:
            List of created file paths
        """
        if output_dir is None:
            output_dir = self.csv_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        metrics = [
            's11_mag_db',
            's11_phase_unwrapped',
            'epsilon_prime',
            'epsilon_double_prime'
        ]
        
        created_files = []
        
        for metric in metrics:
            try:
                output_path = output_dir / f"{self.csv_path.stem}_{metric}.gif"
                gif_path = self.create_gif(
                    metric=metric,
                    output_path=str(output_path),
                    fps=fps,
                    duration=duration
                )
                created_files.append(gif_path)
            except Exception as e:
                print(f"Error creating {metric} GIF: {e}")
        
        return created_files


def create_reaction_gifs(csv_path: str, 
                        poly_order: int = 7,
                        fps: int = 30,
                        duration: int = 10,
                        metrics: Optional[List[str]] = None,
                        smooth_window: int = 11,
                        smooth_polyorder: int = 3) -> List[str]:
    """
    Convenience function to create reaction GIFs from CSV file
    
    Args:
        csv_path: Path to CSV recording
        poly_order: Polynomial order for trendlines
        fps: Frames per second
        duration: Duration in seconds
        metrics: List of metrics to animate (None = all)
        smooth_window: Window length for pre-smoothing filter
        smooth_polyorder: Polynomial order for pre-smoothing filter
        
    Returns:
        List of created GIF file paths
    """
    animator = ReactionAnimator(csv_path, poly_order=poly_order, 
                                smooth_window=smooth_window, 
                                smooth_polyorder=smooth_polyorder)
    animator.load_csv_data()
    
    if metrics is None:
        return animator.create_all_gifs(fps=fps, duration=duration)
    else:
        created_files = []
        for metric in metrics:
            gif_path = animator.create_gif(metric=metric, fps=fps, duration=duration)
            created_files.append(gif_path)
        return created_files


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python gif_animator.py <csv_file> [poly_order] [fps] [duration] [smooth_window]")
        print("\nExample: python gif_animator.py recording.csv 7 30 10 11")
        print("\nParameters:")
        print("  poly_order: Polynomial order for trendline (default: 7)")
        print("  fps: Frames per second (default: 30)")
        print("  duration: Duration in seconds (default: 10)")
        print("  smooth_window: Smoothing window size, must be odd (default: 11)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    poly_order = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    smooth_window = int(sys.argv[5]) if len(sys.argv) > 5 else 11
    
    print("="*60)
    print("Reaction GIF Animator")
    print("="*60)
    print(f"CSV File: {csv_file}")
    print(f"Polynomial Order: {poly_order}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration}s")
    print(f"Smoothing Window: {smooth_window}")
    print("="*60)
    
    created_files = create_reaction_gifs(csv_file, poly_order, fps, duration, 
                                        smooth_window=smooth_window)
    
    print("\n" + "="*60)
    print(f"Created {len(created_files)} GIF(s):")
    for f in created_files:
        print(f"  - {Path(f).name}")
    print("="*60)
