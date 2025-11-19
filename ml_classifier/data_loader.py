"""
Data loading and parsing for ML classification pipeline.
Loads YAML metadata and S1P files from the database directory.
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re


def parse_s1p_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse S1P file and return frequency, magnitude (dB), and phase (degrees).
    
    S1P format: frequency real imag (RI format, 50 ohm)
    We convert to magnitude (dB) and phase (degrees).
    
    Args:
        filepath: Path to .s1p file
        
    Returns:
        freq_hz: Frequency in Hz
        s11_db: S11 magnitude in dB
        phase_deg: S11 phase in degrees
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip comment lines starting with '!'
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('!') and not line.startswith('#')]
    
    freq = []
    s11_db = []
    phase_deg = []
    
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            f_hz = float(parts[0])
            real = float(parts[1])
            imag = float(parts[2])
            
            # Convert RI to magnitude and phase
            magnitude = np.sqrt(real**2 + imag**2)
            s11_db_val = 20 * np.log10(magnitude) if magnitude > 0 else -100
            phase_rad = np.arctan2(imag, real)
            phase_deg_val = np.degrees(phase_rad)
            
            freq.append(f_hz)
            s11_db.append(s11_db_val)
            phase_deg.append(phase_deg_val)
    
    return np.array(freq), np.array(s11_db), np.array(phase_deg)


def load_compound_yaml(yaml_path: Path) -> Dict:
    """
    Load and parse a compound.yaml file.
    
    Args:
        yaml_path: Path to compound.yaml file
        
    Returns:
        Dictionary containing compound info and measurements
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def find_all_compounds(database_dir: Path) -> List[Dict]:
    """
    Recursively find all compound.yaml files in the database directory.
    
    Args:
        database_dir: Root database directory
        
    Returns:
        List of dictionaries, each containing compound data and its directory path
    """
    compounds = []
    
    for yaml_file in database_dir.rglob('compound.yaml'):
        try:
            compound_data = load_compound_yaml(yaml_file)
            compound_data['_dir'] = yaml_file.parent
            compounds.append(compound_data)
        except Exception as e:
            print(f"Warning: Failed to load {yaml_file}: {e}")
    
    return compounds


def find_pure_solvent_file(compound_dir: Path, solvent_name: str, database_dir: Path) -> Optional[Path]:
    """
    Find the S1P file for pure solvent (0% concentration).
    
    Strategy:
    1. Look for 0%.s1p in the current compound's directory
    2. Look for the solvent in the database (e.g., "DI water" -> Water/DI-water.s1p)
    
    Args:
        compound_dir: Directory of the compound measurement
        solvent_name: Name of the solvent (e.g., "DI water")
        database_dir: Root database directory
        
    Returns:
        Path to pure solvent S1P file, or None if not found
    """
    # Strategy 1: Look for 0%.s1p in the same directory
    zero_percent = compound_dir / '0%.s1p'
    if zero_percent.exists():
        return zero_percent
    
    # Strategy 2: Find solvent in database
    if solvent_name and 'water' in solvent_name.lower():
        # Look for Water directory
        water_dir = database_dir / 'Water'
        if water_dir.exists():
            # Try common water file names
            for water_file in ['DI-water.s1p', 'DI-water2.s1p']:
                water_path = water_dir / water_file
                if water_path.exists():
                    return water_path
    
    return None


def extract_concentration_percent(measurement: Dict) -> Optional[float]:
    """
    Extract concentration as a percentage from measurement metadata.
    
    Args:
        measurement: Measurement dictionary from YAML
        
    Returns:
        Concentration as percentage (0-100), or None if not available
    """
    if 'concentration' not in measurement:
        return None
    
    conc = measurement['concentration']
    
    # Try volume_percent first
    if 'volume_percent' in conc:
        return float(conc['volume_percent'])
    
    # Try generic percent
    if 'percent' in conc:
        return float(conc['percent'])
    
    # Try to calculate from mass/volume
    if 'mass_g' in conc and 'volume_ml' in conc:
        mass_g = float(conc['mass_g'])
        volume_ml = float(conc['volume_ml'])
        # Mass percent = (mass / volume) * 100
        # This is an approximation (weight/volume percent)
        return (mass_g / volume_ml) * 100
    
    return None


def normalize_solvent_name(solvent: Optional[str]) -> str:
    """
    Normalize solvent name for one-hot encoding.
    
    Args:
        solvent: Solvent name from YAML (e.g., "DI water", "ethanol")
        
    Returns:
        Normalized solvent name (e.g., "water", "ethanol", "none")
    """
    if solvent is None:
        return 'none'
    
    solvent_lower = solvent.lower().strip()
    
    if 'water' in solvent_lower:
        return 'water'
    elif 'ethanol' in solvent_lower:
        return 'ethanol'
    elif 'isopropanol' in solvent_lower or 'propanol' in solvent_lower:
        return 'isopropanol'
    else:
        return 'other'
