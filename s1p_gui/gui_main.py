"""
Main GUI window for S1P analysis
"""

import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QSplitter, QTextEdit,
    QListWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import DataManager, find_s1p_files
from .formulas import calculate_statistics
from .constants import FREQ_RANGES, AVAILABLE_METRICS, PLOT_COLORS, DEFAULT_PLOT_STYLE


class PlotCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for plotting"""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # Apply plot style
        plt.rcParams.update(DEFAULT_PLOT_STYLE)
        
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
    def clear_plot(self):
        """Clear the plot"""
        self.axes.clear()
        self.axes.grid(True, alpha=0.3, linestyle='--')
        self.draw()


class S1PMainWindow(QMainWindow):
    """Main window for S1P analysis GUI"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("S1P Dielectric Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data manager
        self.data_manager = DataManager()
        self.current_directory = Path.home()
        
        # Initialize plot state
        self.annotation = None
        self.plotted_lines = []
        self.deriv_ax = None
        
        # Setup UI
        self.setup_ui()
        
        # Initial state
        self.update_plot()
        
    def setup_ui(self):
        """Setup the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - horizontal split
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel (controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        left_panel.setMaximumWidth(400)
        
        # Right panel (plot)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # Add panels to main layout
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        
        # === LEFT PANEL: CONTROLS ===
        
        # File management group
        file_group = QGroupBox("File Management")
        file_layout = QVBoxLayout()
        file_group.setLayout(file_layout)
        
        # Add file buttons
        btn_layout = QHBoxLayout()
        self.btn_add_file = QPushButton("Add File")
        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_add_file.clicked.connect(self.add_file)
        self.btn_remove_file = QPushButton("Remove Selected")
        self.btn_remove_file.clicked.connect(self.remove_file)
        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all_files)
        btn_layout.addWidget(self.btn_add_file)
        btn_layout.addWidget(self.btn_add_folder)
        btn_layout.addWidget(self.btn_remove_file)
        btn_layout.addWidget(self.btn_clear_all)
        file_layout.addLayout(btn_layout)
        
        # File list with checkboxes
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.itemChanged.connect(self.on_file_toggled)
        file_layout.addWidget(QLabel("Loaded Files:"))
        file_layout.addWidget(self.file_list)
        
        left_layout.addWidget(file_group)
        
        # Plot settings group
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QVBoxLayout()
        plot_group.setLayout(plot_layout)
        
        # Metric selection
        plot_layout.addWidget(QLabel("Metric to Plot:"))
        self.metric_combo = QComboBox()
        for metric_key, metric_label in AVAILABLE_METRICS:
            self.metric_combo.addItem(metric_label, metric_key)
        self.metric_combo.currentIndexChanged.connect(self.update_plot)
        plot_layout.addWidget(self.metric_combo)
        
        # Frequency range selection
        plot_layout.addWidget(QLabel("Frequency Range:"))
        self.range_combo = QComboBox()
        for range_key, (f_min, f_max, label) in FREQ_RANGES.items():
            self.range_combo.addItem(label, range_key)
        self.range_combo.currentIndexChanged.connect(self.on_range_changed)
        plot_layout.addWidget(self.range_combo)
        
        # Custom frequency range
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Min (MHz):"))
        self.freq_min_spin = QDoubleSpinBox()
        self.freq_min_spin.setRange(0.001, 100000)
        self.freq_min_spin.setValue(10)
        self.freq_min_spin.setSuffix(" MHz")
        self.freq_min_spin.setDecimals(3)
        self.freq_min_spin.valueChanged.connect(self.on_custom_range_changed)
        custom_layout.addWidget(self.freq_min_spin)
        
        custom_layout.addWidget(QLabel("Max (MHz):"))
        self.freq_max_spin = QDoubleSpinBox()
        self.freq_max_spin.setRange(0.001, 100000)
        self.freq_max_spin.setValue(3000)
        self.freq_max_spin.setSuffix(" MHz")
        self.freq_max_spin.setDecimals(3)
        self.freq_max_spin.valueChanged.connect(self.on_custom_range_changed)
        custom_layout.addWidget(self.freq_max_spin)
        
        plot_layout.addLayout(custom_layout)
        
        # Log scale options
        scale_layout = QHBoxLayout()
        self.log_x_check = QCheckBox("Log X-axis")
        self.log_x_check.stateChanged.connect(self.update_plot)
        self.log_y_check = QCheckBox("Log Y-axis")
        self.log_y_check.stateChanged.connect(self.update_plot)
        scale_layout.addWidget(self.log_x_check)
        scale_layout.addWidget(self.log_y_check)
        plot_layout.addLayout(scale_layout)

        # Trendline & derivative options
        trend_layout = QHBoxLayout()
        self.trend_check = QCheckBox("Show linear trendline")
        self.trend_check.stateChanged.connect(self.update_plot)
        self.deriv_check = QCheckBox("Show derivative")
        self.deriv_check.stateChanged.connect(self.update_plot)
        trend_layout.addWidget(self.trend_check)
        trend_layout.addWidget(self.deriv_check)
        plot_layout.addLayout(trend_layout)

        left_layout.addWidget(plot_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics & Parameters")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(250)
        stats_layout.addWidget(self.stats_text)
        
        self.btn_refresh_stats = QPushButton("Refresh Statistics")
        self.btn_refresh_stats.clicked.connect(self.update_statistics)
        stats_layout.addWidget(self.btn_refresh_stats)
        
        left_layout.addWidget(stats_group)
        
        # Add stretch to push everything to top
        left_layout.addStretch()
        
        # === RIGHT PANEL: PLOT ===
        
        # Plot canvas
        self.canvas = PlotCanvas(self, width=8, height=6)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # connect hover event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

    def add_folder(self):
        """Add all .s1p files from a selected folder"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", str(self.current_directory))
        if not folder:
            return
        self.current_directory = Path(folder)
        files = find_s1p_files(Path(folder))
        for f in files:
            data_file = self.data_manager.add_file(f)
            if data_file:
                item = QListWidgetItem(data_file.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                color_idx = len(self.data_manager.files) - 1
                data_file.color = PLOT_COLORS[color_idx % len(PLOT_COLORS)]
                item.setForeground(QColor(data_file.color))
                self.file_list.addItem(item)

        # Update UI
        self.update_frequency_range()
        self.update_plot()
        self.update_statistics()

    def on_mouse_move(self, event):
        """Show an annotation with exact values when hovering over plotted lines"""
        if event.inaxes is None:
            if self.annotation:
                self.annotation.set_visible(False)
                self.canvas.draw_idle()
            return

        x = event.xdata  # in GHz
        y = event.ydata
        if x is None:
            return

        # Find closest point among plotted lines
        best = None
        best_dist = float('inf')
        for entry in self.plotted_lines:
            freq = entry['freq_ghz']
            vals = entry['values']
            if len(freq) == 0:
                continue
            # find index of nearest x
            idx = np.argmin(np.abs(freq - x))
            dx = abs(freq[idx] - x)
            dy = abs(vals[idx] - y) if y is not None else 0
            # prioritize x closeness
            dist = dx + 0.1 * dy
            if dist < best_dist:
                best_dist = dist
                best = (entry, idx)

        if best is None:
            return

        entry, idx = best
        freq_val = entry['freq_ghz'][idx]
        val = entry['values'][idx]
        label = entry['file'].name

        text = f"{label}\n{freq_val:.4f} GHz\n{val:.4e}"

        if self.annotation is None:
            self.annotation = self.canvas.axes.annotate(
                text,
                xy=(freq_val, val), xycoords='data',
                xytext=(15, 15), textcoords='offset points',
                bbox=dict(boxstyle='round', fc='w'),
                arrowprops=dict(arrowstyle='->')
            )
        else:
            self.annotation.xy = (freq_val, val)
            self.annotation.set_text(text)
            self.annotation.set_visible(True)

        self.canvas.draw_idle()
        
    def add_file(self):
        """Open file dialog and add S1P file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select S1P File",
            str(self.current_directory),
            "S1P Files (*.s1p);;All Files (*)"
        )
        
        if file_path:
            self.current_directory = Path(file_path).parent
            data_file = self.data_manager.add_file(Path(file_path))
            
            if data_file:
                # Add to list with checkbox
                item = QListWidgetItem(data_file.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                
                # Assign color
                color_idx = len(self.data_manager.files) - 1
                data_file.color = PLOT_COLORS[color_idx % len(PLOT_COLORS)]
                item.setForeground(QColor(data_file.color))
                
                self.file_list.addItem(item)
                
                # Update frequency range to encompass all files
                self.update_frequency_range()
                self.update_plot()
                self.update_statistics()
    
    def remove_file(self):
        """Remove selected files"""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            return
        
        # Remove in reverse order to maintain indices
        for item in reversed(selected_items):
            row = self.file_list.row(item)
            self.data_manager.remove_file(row)
            self.file_list.takeItem(row)
        
        self.update_plot()
        self.update_statistics()
    
    def clear_all_files(self):
        """Clear all loaded files"""
        self.data_manager.clear_all()
        self.file_list.clear()
        self.update_plot()
        self.update_statistics()
    
    def on_file_toggled(self, item):
        """Handle file checkbox toggle"""
        row = self.file_list.row(item)
        is_checked = item.checkState() == Qt.Checked
        self.data_manager.set_active(row, is_checked)
        self.update_plot()
        self.update_statistics()
    
    def on_range_changed(self):
        """Handle frequency range selection change"""
        range_key = self.range_combo.currentData()
        
        if range_key == 'custom':
            self.freq_min_spin.setEnabled(True)
            self.freq_max_spin.setEnabled(True)
            self.on_custom_range_changed()
        else:
            self.freq_min_spin.setEnabled(False)
            self.freq_max_spin.setEnabled(False)
            
            freq_min, freq_max, _ = FREQ_RANGES[range_key]
            
            # Apply filter
            if freq_max == float('inf'):
                # Full range
                min_f, max_f = self.data_manager.get_global_frequency_range()
                self.data_manager.apply_frequency_filter_all(min_f, max_f)
            else:
                self.data_manager.apply_frequency_filter_all(freq_min, freq_max)
            
            self.update_plot()
            self.update_statistics()
    
    def on_custom_range_changed(self):
        """Handle custom frequency range change"""
        if self.range_combo.currentData() != 'custom':
            return
        
        freq_min = self.freq_min_spin.value() * 1e6  # Convert MHz to Hz
        freq_max = self.freq_max_spin.value() * 1e6
        
        self.data_manager.apply_frequency_filter_all(freq_min, freq_max)
        self.update_plot()
        self.update_statistics()
    
    def update_frequency_range(self):
        """Update frequency range spinboxes based on loaded data"""
        min_f, max_f = self.data_manager.get_global_frequency_range()
        
        # Update spinbox ranges
        self.freq_min_spin.setRange(min_f / 1e6, max_f / 1e6)
        self.freq_max_spin.setRange(min_f / 1e6, max_f / 1e6)
        
        # Set default values
        if self.freq_min_spin.value() < min_f / 1e6:
            self.freq_min_spin.setValue(min_f / 1e6)
        if self.freq_max_spin.value() > max_f / 1e6 or self.freq_max_spin.value() < min_f / 1e6:
            self.freq_max_spin.setValue(max_f / 1e6)
    
    def update_plot(self):
        """Update the plot with current settings"""
        self.canvas.clear_plot()
        
        # Clear derivative axis if it exists
        if self.deriv_ax is not None:
            self.deriv_ax.remove()
            self.deriv_ax = None
        
        active_files = self.data_manager.get_active_files()
        if not active_files:
            self.canvas.axes.text(0.5, 0.5, 'No data loaded', 
                                 ha='center', va='center', transform=self.canvas.axes.transAxes,
                                 fontsize=14, color='gray')
            self.canvas.draw()
            return
        
        # Get selected metric
        metric_key = self.metric_combo.currentData()
        metric_label = self.metric_combo.currentText()
        
        # Get log scale settings
        log_x = self.log_x_check.isChecked()
        log_y = self.log_y_check.isChecked()
        
        # Prepare storage for hover
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
            freq_ghz = freq_valid / 1e9
            
            # Plot with appropriate scale and capture line
            if log_x and log_y:
                line, = self.canvas.axes.loglog(freq_ghz, values_valid, label=file.name,
                                                color=file.color, linewidth=2)
            elif log_x:
                line, = self.canvas.axes.semilogx(freq_ghz, values_valid, label=file.name,
                                                 color=file.color, linewidth=2)
            elif log_y:
                line, = self.canvas.axes.semilogy(freq_ghz, values_valid, label=file.name,
                                                 color=file.color, linewidth=2)
            else:
                line, = self.canvas.axes.plot(freq_ghz, values_valid, label=file.name,
                                             color=file.color, linewidth=2)

            # Store for hover and additional plotting
            self.plotted_lines.append({
                'file': file,
                'line': line,
                'freq_ghz': freq_ghz,
                'values': values_valid
            })

            # Draw linear trendline if requested
            if self.trend_check.isChecked():
                # linear fit in GHz
                coeffs = np.polyfit(freq_ghz, values_valid, 1)
                trend = np.polyval(coeffs, freq_ghz)
                self.canvas.axes.plot(freq_ghz, trend, linestyle='--', color=file.color, alpha=0.7)

            # Draw derivative if requested on a twin y-axis
            if self.deriv_check.isChecked():
                if self.deriv_ax is None:
                    self.deriv_ax = self.canvas.axes.twinx()
                    # offset the right spine a bit
                    self.deriv_ax.spines['right'].set_position(('outward', 60))
                # smooth derivative using numpy.gradient
                deriv = np.gradient(values_valid, freq_ghz)
                self.deriv_ax.plot(freq_ghz, deriv, linestyle=':', color=file.color, alpha=0.9)
                self.deriv_ax.set_ylabel('d(metric)/dGHz', fontweight='bold')
        
        # Labels and legend
        self.canvas.axes.set_xlabel('Frequency (GHz)', fontweight='bold')
        self.canvas.axes.set_ylabel(metric_label, fontweight='bold')
        self.canvas.axes.set_title(f'{metric_label} vs Frequency', fontweight='bold')
        self.canvas.axes.legend()
        self.canvas.axes.grid(True, alpha=0.3, linestyle='--', which='both' if (log_x or log_y) else 'major')
        
        self.canvas.draw()
    
    def update_statistics(self):
        """Update statistics display"""
        active_files = self.data_manager.get_active_files()
        metric_key = self.metric_combo.currentData()
        
        if not active_files:
            self.stats_text.setPlainText("No active files")
            return
        
        stats_html = "<html><body style='font-family: monospace;'>"
        
        for file in active_files:
            data = file.get_data(use_filtered=True)
            if data is None:
                continue
            
            stats = calculate_statistics(data, metric_key)
            if not stats:
                continue
            
            stats_html += f"<h3 style='color: {file.color};'>{file.name}</h3>"
            stats_html += "<table cellpadding='3'>"
            
            # Basic statistics
            stats_html += f"<tr><td><b>Mean:</b></td><td>{stats.get('mean', np.nan):.4e}</td></tr>"
            stats_html += f"<tr><td><b>Median:</b></td><td>{stats.get('median', np.nan):.4e}</td></tr>"
            stats_html += f"<tr><td><b>Std Dev:</b></td><td>{stats.get('std', np.nan):.4e}</td></tr>"
            stats_html += f"<tr><td><b>Min:</b></td><td>{stats.get('min', np.nan):.4e}</td></tr>"
            stats_html += f"<tr><td><b>Max:</b></td><td>{stats.get('max', np.nan):.4e}</td></tr>"
            stats_html += f"<tr><td><b>Range:</b></td><td>{stats.get('range', np.nan):.4e}</td></tr>"
            
            # Calculated parameters
            stats_html += "<tr><td colspan='2'><hr></td></tr>"
            stats_html += f"<tr><td><b>Slope:</b></td><td>{stats.get('slope', np.nan):.4e} /GHz</td></tr>"
            stats_html += f"<tr><td><b>Integral:</b></td><td>{stats.get('integral', np.nan):.4e}</td></tr>"
            
            # Sub-range means (if available)
            if 'mean_low_0.5_1GHz' in stats:
                stats_html += "<tr><td colspan='2'><hr></td></tr>"
                stats_html += "<tr><td colspan='2'><b>Sub-range Means:</b></td></tr>"
                stats_html += f"<tr><td>0.5-1 GHz:</td><td>{stats.get('mean_low_0.5_1GHz', np.nan):.4e}</td></tr>"
                stats_html += f"<tr><td>1-2 GHz:</td><td>{stats.get('mean_mid_1_2GHz', np.nan):.4e}</td></tr>"
                stats_html += f"<tr><td>2-3 GHz:</td><td>{stats.get('mean_high_2_3GHz', np.nan):.4e}</td></tr>"
                stats_html += f"<tr><td>Ratio (L/H):</td><td>{stats.get('ratio_low_high', np.nan):.4f}</td></tr>"
            
            stats_html += "</table><br>"
        
        stats_html += "</body></html>"
        self.stats_text.setHtml(stats_html)


def main():
    """Main entry point for the S1P GUI application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = S1PMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
