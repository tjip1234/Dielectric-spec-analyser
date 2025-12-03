"""
Data loading and file management
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import skrf as rf
from .formulas import calculate_dielectric_properties, filter_frequency_range


def _format_concentration(conc: Optional[Dict]) -> Optional[str]:
    """Format a concentration dict into a concise string for plot labels.

    Mirrors the logic used in database_loader to handle different key names
    and key orders.
    """
    if not conc:
        return None

    # Try volume_percent first (most specific)
    if 'volume_percent' in conc and conc['volume_percent'] is not None:
        try:
            vp = float(conc['volume_percent'])
            return f"{vp:g}% v/v"
        except Exception:
            return f"{conc['volume_percent']}% v/v"
    
    # Try generic percent
    if 'percent' in conc and conc['percent'] is not None:
        try:
            p = float(conc['percent'])
            return f"{p:g}%"
        except Exception:
            return f"{conc['percent']}%"

    # Try mass/volume combination
    mass_keys = ('mass_g', 'mass', 'g')
    vol_keys = ('volume_ml', 'volume', 'vol_ml', 'ml')

    mass = None
    vol = None
    for k in mass_keys:
        if k in conc and conc[k] is not None:
            mass = conc[k]
            break
    for k in vol_keys:
        if k in conc and conc[k] is not None:
            vol = conc[k]
            break

    if mass is not None and vol is not None:
        try:
            m = float(mass)
            v = float(vol)
            return f"{m:g}g/{v:g}ml"
        except Exception:
            return f"{mass}g/{vol}ml"

    # If the concentration is already a string
    if isinstance(conc, str):
        return conc

    return None


class S1PDataFile:
    """Represents a single S1P data file with computed properties"""

    def __init__(self, filepath: Path, name: Optional[str] = None, calibration=None, 
                 chemical_name: Optional[str] = None, concentration: Optional[Dict] = None,
                 unwrap_phase: bool = True, temperature_c: Optional[float] = None):
        """
        Initialize S1P data file

        Args:
            filepath: Path to .s1p file
            name: Display name (defaults to filename stem)
            calibration: ProbeCalibration object (optional)
            chemical_name: Chemical name of the compound (optional)
            concentration: Concentration dict with mass_g and volume_ml (optional)
            unwrap_phase: Whether to unwrap phase discontinuities (default True)
            temperature_c: Temperature in Celsius (optional, for temperature series)
        """
        self.filepath = Path(filepath)
        self.name = name or self.filepath.stem
        self.network = None
        self.full_data = None
        self.filtered_data = None
        self.is_loaded = False
        self.color = None
        self.calibration = calibration
        self.chemical_name = chemical_name
        self.concentration = concentration
        self.unwrap_phase = unwrap_phase
        self.temperature_c = temperature_c

    def load(self, unwrap_phase: Optional[bool] = None) -> bool:
        """
        Load S1P file and calculate properties

        Uses calibration if available, otherwise falls back to simplified model.

        Args:
            unwrap_phase: Override unwrap_phase setting for this load (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            self.network = rf.Network(str(self.filepath))

            # Calculate full spectrum properties
            freq = self.network.f
            s11 = self.network.s[:, 0, 0]

            # Use override if provided, otherwise use instance setting
            use_unwrap = unwrap_phase if unwrap_phase is not None else self.unwrap_phase
            
            self.full_data = calculate_dielectric_properties(s11, freq, self.calibration, use_unwrap)
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
    
    def get_plot_label(self) -> str:
        """Get label for plot legend with chemical name, concentration, and temperature if available"""
        if self.chemical_name:
            label = self.chemical_name
            parts = []
            conc_str = _format_concentration(self.concentration)
            if conc_str:
                parts.append(conc_str)
            if self.temperature_c is not None:
                parts.append(f"{self.temperature_c}°C")
            if parts:
                label += f" ({', '.join(parts)})"
            return label
        # If no chemical name but we have temperature, include it with the file name
        if self.temperature_c is not None:
            return f"{self.name} ({self.temperature_c}°C)"
        return self.name
    
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

    def add_file(self, filepath: Path, name: Optional[str] = None, 
                 chemical_name: Optional[str] = None, 
                 concentration: Optional[Dict] = None,
                 unwrap_phase: bool = True,
                 temperature_c: Optional[float] = None) -> Optional[S1PDataFile]:
        """
        Add and load a new S1P file

        Args:
            filepath: Path to .s1p file
            name: Optional display name
            chemical_name: Optional chemical name for plot labels
            concentration: Optional concentration dict
            unwrap_phase: Whether to unwrap phase discontinuities (default True)
            temperature_c: Optional temperature in Celsius (for temperature series)

        Returns:
            S1PDataFile object if successful, None otherwise
        """
        data_file = S1PDataFile(filepath, name, self.calibration, chemical_name, concentration, unwrap_phase, temperature_c)

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
    
    def reload_all_with_unwrap(self, unwrap_phase: bool):
        """
        Reload all files with new phase unwrapping setting
        
        Args:
            unwrap_phase: Whether to unwrap phase discontinuities
        """
        for file in self.files:
            file.unwrap_phase = unwrap_phase
            file.load()
    
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
