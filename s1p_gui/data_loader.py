"""
Data loading and file management
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import skrf as rf
from .formulas import calculate_dielectric_properties, filter_frequency_range


class S1PDataFile:
    """Represents a single S1P data file with computed properties"""

    def __init__(self, filepath: Path, name: Optional[str] = None, calibration=None):
        """
        Initialize S1P data file

        Args:
            filepath: Path to .s1p file
            name: Display name (defaults to filename stem)
            calibration: ProbeCalibration object (optional)
        """
        self.filepath = Path(filepath)
        self.name = name or self.filepath.stem
        self.network = None
        self.full_data = None
        self.filtered_data = None
        self.is_loaded = False
        self.color = None
        self.calibration = calibration

    def load(self) -> bool:
        """
        Load S1P file and calculate properties

        Uses calibration if available, otherwise falls back to simplified model.

        Returns:
            True if successful, False otherwise
        """
        try:
            self.network = rf.Network(str(self.filepath))

            # Calculate full spectrum properties
            freq = self.network.f
            s11 = self.network.s[:, 0, 0]

            self.full_data = calculate_dielectric_properties(s11, freq, self.calibration)
            self.filtered_data = self.full_data.copy()
            self.is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading {self.filepath}: {e}")
            self.is_loaded = False
            return False

    def set_calibration(self, calibration):
        """
        Set calibration for this file and reload data

        Args:
            calibration: ProbeCalibration object
        """
        self.calibration = calibration
        if self.is_loaded and self.network is not None:
            # Recalculate with new calibration
            self.load()
    
    def apply_frequency_filter(self, freq_min: float, freq_max: float):
        """
        Filter data to specific frequency range
        
        Args:
            freq_min: Minimum frequency in Hz
            freq_max: Maximum frequency in Hz
        """
        if not self.is_loaded:
            return
        
        self.filtered_data = filter_frequency_range(self.full_data, freq_min, freq_max)
    
    def get_data(self, use_filtered: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """
        Get data dictionary
        
        Args:
            use_filtered: Return filtered data if True, full data if False
            
        Returns:
            Data dictionary or None if not loaded
        """
        if not self.is_loaded:
            return None
        
        return self.filtered_data if use_filtered else self.full_data
    
    def get_frequency_range(self) -> tuple:
        """Get current frequency range (min, max) in Hz"""
        if not self.is_loaded or self.filtered_data is None:
            return (0, 0)
        
        freq = self.filtered_data['frequency']
        return (freq[0], freq[-1])
    
    def __repr__(self):
        status = "loaded" if self.is_loaded else "not loaded"
        return f"S1PDataFile('{self.name}', {status})"


class DataManager:
    """Manages multiple S1P data files with optional calibration"""

    def __init__(self, calibration=None):
        self.files: List[S1PDataFile] = []
        self.active_files: List[bool] = []
        self.calibration = calibration

    def set_calibration(self, calibration):
        """
        Set calibration and apply to all files

        Args:
            calibration: ProbeCalibration object
        """
        self.calibration = calibration
        # Apply to all loaded files
        for file in self.files:
            file.set_calibration(calibration)

    def add_file(self, filepath: Path, name: Optional[str] = None) -> Optional[S1PDataFile]:
        """
        Add and load a new S1P file

        Args:
            filepath: Path to .s1p file
            name: Optional display name

        Returns:
            S1PDataFile object if successful, None otherwise
        """
        data_file = S1PDataFile(filepath, name, self.calibration)

        if data_file.load():
            self.files.append(data_file)
            self.active_files.append(True)
            return data_file
        
        return None
    
    def remove_file(self, index: int) -> bool:
        """
        Remove a file by index
        
        Args:
            index: Index of file to remove
            
        Returns:
            True if successful
        """
        if 0 <= index < len(self.files):
            self.files.pop(index)
            self.active_files.pop(index)
            return True
        return False
    
    def clear_all(self):
        """Remove all files"""
        self.files.clear()
        self.active_files.clear()
    
    def set_active(self, index: int, active: bool):
        """Set whether a file is active for plotting"""
        if 0 <= index < len(self.active_files):
            self.active_files[index] = active
    
    def get_active_files(self) -> List[S1PDataFile]:
        """Get list of active files"""
        return [f for f, active in zip(self.files, self.active_files) if active]
    
    def apply_frequency_filter_all(self, freq_min: float, freq_max: float):
        """Apply frequency filter to all files"""
        for file in self.files:
            file.apply_frequency_filter(freq_min, freq_max)
    
    def get_global_frequency_range(self) -> tuple:
        """Get the global frequency range across all files"""
        if not self.files:
            return (0, 1e9)
        
        all_ranges = [f.get_frequency_range() for f in self.files if f.is_loaded]
        if not all_ranges:
            return (0, 1e9)
        
        min_freq = min(r[0] for r in all_ranges)
        max_freq = max(r[1] for r in all_ranges)
        
        return (min_freq, max_freq)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return self.files[index]


def find_s1p_files(directory: Path) -> List[Path]:
    """
    Find all .s1p files in a directory
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Path objects for .s1p files
    """
    return sorted(directory.glob("*.s1p"))


def find_result_folders(directory: Path) -> List[Path]:
    """
    Find all folders starting with 'result' in a directory
    
    Args:
        directory: Directory to search
        
    Returns:
        List of Path objects for result folders
    """
    return sorted([d for d in directory.iterdir() 
                   if d.is_dir() and d.name.lower().startswith('result')])
