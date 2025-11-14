#!/usr/bin/env python3
"""
Standalone script to create reaction GIFs from NanoVNA CSV recordings

Usage:
    python3 create_reaction_gifs.py <csv_file>
    python3 create_reaction_gifs.py reactions/recording_20251112_234332.csv
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from s1p_gui.gif_animator import create_reaction_gifs

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 create_reaction_gifs.py <csv_file> [poly_order] [fps] [duration] [smooth_window]")
        print("\nExample: python3 create_reaction_gifs.py reactions/recording.csv 7 30 10 11")
        print("\nDefaults:")
        print("  poly_order: 7 (smoothness of trendlines)")
        print("  fps: 30 (frames per second)")
        print("  duration: 10 (seconds)")
        print("  smooth_window: 11 (pre-smoothing to remove noise spikes)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    poly_order = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    fps = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    duration = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    smooth_window = int(sys.argv[5]) if len(sys.argv) > 5 else 11
    
    print("="*70)
    print("NanoVNA Reaction GIF Generator")
    print("="*70)
    print(f"CSV File: {csv_file}")
    print(f"Polynomial Order: {poly_order}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration}s")
    print(f"Smoothing Window: {smooth_window}")
    print(f"\nFeatures:")
    print(f"  ✓ Pre-smoothing to remove noise spikes")
    print(f"  ✓ Polynomial trendlines (no raw data)")
    print(f"  ✓ Log time scale (more detail at start)")
    print(f"  ✓ Phase unwrapping")
    print(f"  ✓ Initial state reference line")
    print("="*70)
    print()
    
    try:
        created_files = create_reaction_gifs(csv_file, poly_order, fps, duration, 
                                            smooth_window=smooth_window)
        
        print("\n" + "="*70)
        print(f"✓ Successfully created {len(created_files)} GIF(s):")
        for f in created_files:
            size_mb = Path(f).stat().st_size / 1024 / 1024
            print(f"  - {Path(f).name} ({size_mb:.1f} MB)")
        print("="*70)
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
