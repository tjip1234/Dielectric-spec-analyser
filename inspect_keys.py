import numpy as np
from s1p_gui.formulas import calculate_dielectric_properties
s11 = np.array([0.5+0j, 0.5+0j])
freq = np.array([10e6, 20e6])
data = calculate_dielectric_properties(s11, freq)
print(data.keys())
