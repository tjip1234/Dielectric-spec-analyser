import re
with open('s1p_gui/formulas.py', 'r') as f:
    content = f.read()

new_stat_start = """    if metric == 's11_ifft':
        if data is None or 's11' not in data:
            return {}
        s11_comp = data['s11']
        freq_vals = data['frequency']
        if len(freq_vals) < 2:
            return {}
        df = freq_vals[1] - freq_vals[0]
        values = np.abs(np.fft.ifft(s11_comp))
        freq = np.arange(len(freq_vals)) / (len(freq_vals) * df) * 1e9  # ns time acts as freq
    else:
        if data is None or metric not in data:
            return {}
        
        freq = data['frequency']
        values = data[metric]"""

old_stat_start = """    if data is None or metric not in data:
        return {}
    
    freq = data['frequency']
    values = data[metric]"""

content = content.replace(old_stat_start, new_stat_start)

with open('s1p_gui/formulas.py', 'w') as f:
    f.write(content)
