import numpy as np
from s1p_gui.formulas import calculate_dielectric_properties

s11 = np.random.rand(100) + 1j * np.random.rand(100)
freq = np.linspace(10e6, 3e9, 100)
data = calculate_dielectric_properties(s11, freq)

td_values = np.abs(np.fft.ifft(data['s11']))
df = freq[1] - freq[0]
time_array = np.fft.fftfreq(len(freq), d=df)
# actually, time typically starts at 0 to T. np.fft.ifft output corresponds to 0 <= t < T
time_array = np.arange(len(freq)) / (len(freq) * df)

print(time_array[:5])
print(td_values[:5])
