"""
Database loader for compound YAML files
"""

import yaml
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import date


def _format_concentration(conc: Optional[Dict]) -> Optional[str]:
    """Format a concentration dict into a short human readable string.

    Handles keys in any order and several common key names. Returns a string
    like "1g/10ml", "10% v/v", or similar depending on available keys.
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


@dataclass
class Measurement:
    """Represents a single measurement entry"""
    date: date
    temperature_c: float
    solvent: Optional[str]
    is_pure: bool
    data_file: str
    concentration: Optional[Dict] = None
    purity_percent: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class Compound:
    """Represents a compound with its metadata and measurements"""
    folder_path: Path
    common_name: str
    chemical_name: str
    cas_number: str
    molecular_weight: float
    functional_groups: List[str]
    chemical_formula: str
    measurements: List[Measurement]
    temperature_series: bool = False
    
    def get_display_name(self) -> str:
        """Get a nice display name for the compound"""
        if self.common_name != self.chemical_name:
            return f"{self.common_name} ({self.chemical_name})"
        return self.common_name
    
    def get_measurement_file_path(self, measurement: Measurement) -> Path:
        """Get the full path to a measurement data file"""
        return self.folder_path / measurement.data_file
    
    def get_measurement_description(self, measurement: Measurement) -> str:
        """Get a description string for a measurement"""
        desc_parts = [str(measurement.date)]
        
        if measurement.solvent:
            desc_parts.append(f"in {measurement.solvent}")
        
        if not measurement.is_pure:
            # Format concentration robustly (keys may be in different orders or use alternate names)
            conc_str = _format_concentration(measurement.concentration)
            if conc_str:
                desc_parts.append(conc_str)
            elif measurement.purity_percent:
                desc_parts.append(f"{measurement.purity_percent}% pure")
        
        if measurement.temperature_c:
            desc_parts.append(f"{measurement.temperature_c}°C")
        
        if measurement.notes:
            desc_parts.append(f"({measurement.notes})")
        
        return " - ".join(desc_parts)


def load_compound_yaml(yaml_path: Path) -> Optional[Compound]:
    """
    Load a compound from its YAML file
    
    Args:
        yaml_path: Path to compound.yaml file
        
    Returns:
        Compound object or None if loading fails
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        compound_data = data.get('compound', {})
        measurements_data = data.get('measurements', [])
        
        # Parse measurements
        measurements = []
        for m in measurements_data:
            # Parse date string
            date_obj = m['date'] if isinstance(m['date'], date) else date.fromisoformat(str(m['date']))
            
            measurement = Measurement(
                date=date_obj,
                temperature_c=m.get('temperature_c'),
                solvent=m.get('solvent'),
                is_pure=m.get('is_pure', True),
                data_file=m.get('data_file'),
                concentration=m.get('concentration'),
                purity_percent=m.get('purity_percent'),
                notes=m.get('notes')
            )
            measurements.append(measurement)
        
        compound = Compound(
            folder_path=yaml_path.parent,
            common_name=compound_data.get('common_name', ''),
            chemical_name=compound_data.get('chemical_name', ''),
            cas_number=compound_data.get('cas_number', ''),
            molecular_weight=compound_data.get('molecular_weight', 0.0),
            functional_groups=compound_data.get('functional_groups', []),
            chemical_formula=compound_data.get('chemical_formula', ''),
            measurements=measurements,
            temperature_series=compound_data.get('temperature_series', False)
        )
        
        return compound
        
    except Exception as e:
        print(f"Error loading compound from {yaml_path}: {e}")
        return None


def find_all_compounds(database_dir: Path) -> List[Compound]:
    """
    Find and load all compounds from the database directory
    
    Args:
        database_dir: Path to Database directory
        
    Returns:
        List of Compound objects sorted by common name
    """
    compounds = []
    
    if not database_dir.exists():
        return compounds
    
    # Find all compound.yaml files
    for folder in database_dir.iterdir():
        if not folder.is_dir():
            continue
        
        yaml_file = folder / 'compound.yaml'
        if yaml_file.exists():
            compound = load_compound_yaml(yaml_file)
            if compound:
                compounds.append(compound)
    
    # Sort by common name
    compounds.sort(key=lambda c: c.common_name.lower())
    
    return compounds


def get_compound_measurements_as_tuples(compound: Compound) -> List[Tuple[str, Path, str]]:
    """
    Get all measurements for a compound as (name, filepath, description) tuples
    
    Args:
        compound: Compound object
        
    Returns:
        List of (display_name, file_path, description) tuples
    """
    results = []
    
    for i, measurement in enumerate(compound.measurements):
        file_path = compound.get_measurement_file_path(measurement)
        
        if not file_path.exists():
            continue
        
        # Create display name
        if len(compound.measurements) == 1:
            display_name = compound.get_display_name()
        else:
            display_name = f"{compound.get_display_name()} #{i+1}"
        
        description = compound.get_measurement_description(measurement)
        
        results.append((display_name, file_path, description))
    
    return results


def search_compounds(compounds: List[Compound], query: str) -> List[Compound]:
    """
    Search compounds by name, formula, functional groups, or CAS number
    
    Args:
        compounds: List of compounds to search
        query: Search query
        
    Returns:
        Filtered list of matching compounds
    """
    if not query:
        return compounds
    
    query_lower = query.lower()
    results = []
    
    for compound in compounds:
        # Search in various fields
        if (query_lower in compound.common_name.lower() or
            query_lower in compound.chemical_name.lower() or
            query_lower in compound.chemical_formula.lower() or
            query_lower in compound.cas_number.lower() or
            any(query_lower in fg.lower() for fg in compound.functional_groups)):
            results.append(compound)
    
    return results
