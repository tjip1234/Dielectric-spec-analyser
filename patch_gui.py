import re
with open('s1p_gui/gui_main.py', 'r') as f:
    content = f.read()

# Replace the inner loop logic
new_inner_loop = """
        # Prepare storage for hover
        self.plotted_lines = []
        
        is_ifft = (metric_key == 's11_ifft')
        x_label_base = 'Time (ns)' if is_ifft else 'Frequency (GHz)'
        x_unit = 'ns' if is_ifft else 'GHz'
        d_label_base = f'd/d{x_unit}'

        # Plot each active file
        for file in active_files:
            data = file.get_data(use_filtered=True)
            if data is None:
                continue

            if is_ifft:
                if 's11' not in data:
                    continue
                freq_vals = data['frequency']
                if len(freq_vals) < 2:
                    continue
                df = freq_vals[1] - freq_vals[0]
                # compute IFFT
                s11_comp = data['s11']
                values_arr = np.abs(np.fft.ifft(s11_comp))
                # Generate time array in nanoseconds
                time_array = np.arange(len(freq_vals)) / (len(freq_vals) * df)
                x_vals_arr = time_array * 1e9
            else:
                if metric_key not in data:
                    continue
                freq_vals = data['frequency']
                values_arr = data[metric_key]
                x_vals_arr = freq_vals / 1e9

            # Filter out invalid values
            valid_mask = np.isfinite(values_arr)
            if not np.any(valid_mask):
                continue

            freq_valid = x_vals_arr[valid_mask]  # we keep the name freq_valid for minimal change but it's really x_valid
            values_valid = values_arr[valid_mask]

            # Convert frequency to GHz for display
            freq_ghz = freq_valid # Now really x_vals
"""

old_part_1 = """        # Prepare storage for hover
        self.plotted_lines = []

        # Plot each active file
        for file in active_files:
            data = file.get_data(use_filtered=True)
            if data is None or metric_key not in data:
                continue

            freq = data['frequency']
            values = data[metric_key]

            # Filter out invalid values
            valid_mask = np.isfinite(values)
            if not np.any(valid_mask):
                continue

            freq_valid = freq[valid_mask]
            values_valid = values[valid_mask]

            # Convert frequency to GHz for display
            freq_ghz = freq_valid / 1e9"""

content = content.replace(old_part_1, new_inner_loop.strip('\n'))

content = content.replace("'freq_ghz': freq_ghz,", "'x_vals': freq_ghz,\n                    'x_unit': x_unit,")
content = content.replace("d/dGHz", "{d_label_base}")
content = content.replace("Frequency (GHz)", "{x_label_base}")

# Fix label
content = content.replace("self.canvas.axes.set_xlabel('{x_label_base}', fontweight='bold')", "self.canvas.axes.set_xlabel(x_label_base, fontweight='bold')")
content = content.replace("self.canvas.axes.set_title(f'{metric_label} vs Frequency', fontweight='bold')", "self.canvas.axes.set_title(f'{metric_label} vs {x_label_base.split()[0]}', fontweight='bold')")

with open('s1p_gui/gui_main.py', 'w') as f:
    f.write(content)
