"""
Physical constants and configuration parameters
"""

# Physical constants
EPSILON_0 = 8.854e-12  # Permittivity of free space (F/m)
SPEED_OF_LIGHT = 299792458  # Speed of light (m/s)

# Frequency ranges for analysis (Hz)
FREQ_RANGES = {
    'low': (10e6, 500e6, '10 MHz - 500 MHz'),
    'mid': (500e6, 3e9, '500 MHz - 3 GHz'),
    'full': (0, float('inf'), 'Full Sweep'),
    'custom': (0, 0, 'Custom Range')
}

# Plot colors (professional scientific palette)
PLOT_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC"
]


# Available metrics for plotting
AVAILABLE_METRICS = [
    ('s11_db', '|S₁₁| (dB)'),
    ('s11_phase', '∠S₁₁ (degrees)'),
    ('epsilon_prime', "Real Permittivity (ε')"),
    ('epsilon_double_prime', "Imaginary Permittivity (ε'')"),
    ('epsilon_magnitude', 'Complex Permittivity Magnitude (|ε*|)'),
]

# Metric units for labels
METRIC_UNITS = {
    's11_db': 'dB',
    's11_phase': 'degrees',
    'epsilon_prime': '',
    'epsilon_double_prime': '',
    'epsilon_magnitude': '',
}

# Default plot settings
DEFAULT_PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 1.2,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
}
