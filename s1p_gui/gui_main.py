"""
Main GUI window for S1P analysis
"""

import sys
import csv
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QLabel, QComboBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QSplitter, QTextEdit,
    QListWidgetItem, QAbstractItemView, QMessageBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from .data_loader import DataManager, find_s1p_files
from .formulas import calculate_statistics
from .cole_cole import calculate_cole_cole_parameters
from .constants import FREQ_RANGES, AVAILABLE_METRICS, PLOT_COLORS, DEFAULT_PLOT_STYLE
from .calibration import ProbeCalibration


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
        
        # Data manager and calibration
        self.calibration = ProbeCalibration()
        self.data_manager = DataManager(self.calibration)
        self.current_directory = Path.home()

        # Initialize plot state
        self.annotation = None
        self.plotted_lines = []
        self.deriv_ax = None

        # Store statistics data for export
        self.current_stats_data = None

        # Calibration UI state
        self.ref_files = {}  # {'water': path, 'ethanol': path, 'isopropanol': path}
        
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

        # Middle panel (plot)
        plot_panel = QWidget()
        plot_layout_main = QVBoxLayout()
        plot_panel.setLayout(plot_layout_main)

        # Right panel (statistics)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)

        # Add panels to main layout with splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(plot_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        main_layout.addWidget(splitter)

        # Store reference for later use
        self.right_panel = right_panel
        self.right_layout = right_layout
        
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

        # Data lines & trendline options
        data_lines_layout = QHBoxLayout()
        self.data_lines_check = QCheckBox("Show data lines")
        self.data_lines_check.setChecked(True)
        self.data_lines_check.stateChanged.connect(self.update_plot)
        self.trend_check = QCheckBox("Show trendlines")
        self.trend_check.stateChanged.connect(self.update_plot)
        data_lines_layout.addWidget(self.data_lines_check)
        data_lines_layout.addWidget(self.trend_check)
        plot_layout.addLayout(data_lines_layout)

        # Derivative options
        deriv_layout = QHBoxLayout()
        self.deriv_check = QCheckBox("Show derivative")
        self.deriv_check.stateChanged.connect(self.update_plot)
        self.deriv_on_trend_check = QCheckBox("Show deriv on trend")
        self.deriv_on_trend_check.stateChanged.connect(self.update_plot)
        deriv_layout.addWidget(self.deriv_check)
        deriv_layout.addWidget(self.deriv_on_trend_check)
        plot_layout.addLayout(deriv_layout)
        
        # Polynomial order control
        poly_layout = QHBoxLayout()
        poly_layout.addWidget(QLabel("Polynomial order:"))
        self.poly_order_spin = QDoubleSpinBox()
        self.poly_order_spin.setRange(1, 20)
        self.poly_order_spin.setValue(5)
        self.poly_order_spin.setDecimals(0)
        self.poly_order_spin.setSingleStep(1)
        self.poly_order_spin.valueChanged.connect(self.update_plot)
        poly_layout.addWidget(self.poly_order_spin)
        poly_layout.addWidget(QLabel("(for fits/derivs)"))
        plot_layout.addLayout(poly_layout)

        left_layout.addWidget(plot_group)

        # Calibration panel
        cal_group = QGroupBox("Probe Calibration")
        cal_layout = QVBoxLayout()
        cal_group.setLayout(cal_layout)

        # Status label
        self.cal_status_label = QLabel("Status: No calibration")
        self.cal_status_label.setStyleSheet("color: gray; font-weight: bold;")
        cal_layout.addWidget(self.cal_status_label)

        # Load reference files
        ref_files_label = QLabel("Reference Files:")
        cal_layout.addWidget(ref_files_label)

        # Water file button
        water_layout = QHBoxLayout()
        water_layout.addWidget(QLabel("Water:"))
        self.btn_load_water = QPushButton("Load")
        self.btn_load_water.clicked.connect(lambda: self.load_reference_file('water'))
        self.lbl_water = QLabel("(not loaded)")
        self.lbl_water.setStyleSheet("color: gray; font-size: 9px;")
        water_layout.addWidget(self.btn_load_water)
        water_layout.addWidget(self.lbl_water)
        cal_layout.addLayout(water_layout)

        # Ethanol file button
        ethanol_layout = QHBoxLayout()
        ethanol_layout.addWidget(QLabel("Ethanol:"))
        self.btn_load_ethanol = QPushButton("Load")
        self.btn_load_ethanol.clicked.connect(lambda: self.load_reference_file('ethanol'))
        self.lbl_ethanol = QLabel("(not loaded)")
        self.lbl_ethanol.setStyleSheet("color: gray; font-size: 9px;")
        ethanol_layout.addWidget(self.btn_load_ethanol)
        ethanol_layout.addWidget(self.lbl_ethanol)
        cal_layout.addLayout(ethanol_layout)

        # Isopropanol file button
        iso_layout = QHBoxLayout()
        iso_layout.addWidget(QLabel("Isopropanol:"))
        self.btn_load_iso = QPushButton("Load")
        self.btn_load_iso.clicked.connect(lambda: self.load_reference_file('isopropanol'))
        self.lbl_iso = QLabel("(not loaded)")
        self.lbl_iso.setStyleSheet("color: gray; font-size: 9px;")
        iso_layout.addWidget(self.btn_load_iso)
        iso_layout.addWidget(self.lbl_iso)
        cal_layout.addLayout(iso_layout)

        # Calibrate button
        self.btn_calibrate = QPushButton("Perform Calibration")
        self.btn_calibrate.clicked.connect(self.perform_calibration)
        self.btn_calibrate.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        cal_layout.addWidget(self.btn_calibrate)

        # Save/Load calibration
        cal_buttons_layout = QHBoxLayout()
        self.btn_save_cal = QPushButton("Save Cal")
        self.btn_save_cal.clicked.connect(self.save_calibration)
        self.btn_load_cal = QPushButton("Load Cal")
        self.btn_load_cal.clicked.connect(self.load_calibration)
        cal_buttons_layout.addWidget(self.btn_save_cal)
        cal_buttons_layout.addWidget(self.btn_load_cal)
        cal_layout.addLayout(cal_buttons_layout)

        left_layout.addWidget(cal_group)

        # Add stretch to push everything to top
        left_layout.addStretch()

        # === MIDDLE PANEL: PLOT ===

        # Plot canvas
        self.canvas = PlotCanvas(self, width=8, height=6)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # connect hover event
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

        plot_layout_main.addWidget(self.toolbar)
        plot_layout_main.addWidget(self.canvas)

        # === RIGHT PANEL: STATISTICS ===

        # Statistics display
        stats_group = QGroupBox("Compound Comparison")
        stats_layout_group = QVBoxLayout()
        stats_group.setLayout(stats_layout_group)

        # Button row for actions
        button_layout = QHBoxLayout()
        self.btn_refresh_stats = QPushButton("Refresh")
        self.btn_refresh_stats.clicked.connect(self.update_statistics)
        self.btn_copy_stats = QPushButton("Copy")
        self.btn_copy_stats.clicked.connect(self.copy_stats_to_clipboard)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(self.export_stats_csv)
        self.btn_export_json = QPushButton("Export JSON")
        self.btn_export_json.clicked.connect(self.export_stats_json)

        button_layout.addWidget(self.btn_refresh_stats)
        button_layout.addWidget(self.btn_copy_stats)
        button_layout.addWidget(self.btn_export_csv)
        button_layout.addWidget(self.btn_export_json)
        stats_layout_group.addLayout(button_layout)

        # Statistics table
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout_group.addWidget(self.stats_text)

        right_layout.addWidget(stats_group)
        right_layout.addStretch()

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
        show_data_lines = self.data_lines_check.isChecked()
        show_trendlines = self.trend_check.isChecked()
        show_deriv = self.deriv_check.isChecked()
        show_deriv_on_trend = self.deriv_on_trend_check.isChecked()

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

            # Plot data line if requested
            if show_data_lines:
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

            # Draw polynomial trendline if requested
            if show_trendlines:
                # Get polynomial order from spinbox
                poly_order = int(self.poly_order_spin.value())
                
                # Polynomial fit in GHz
                coeffs = np.polyfit(freq_ghz, values_valid, poly_order)
                trend = np.polyval(coeffs, freq_ghz)
                trend_label = f"{file.name} (poly{poly_order})" if show_data_lines else f"{file.name}"
                trend_line, = self.canvas.axes.plot(freq_ghz, trend, linestyle='--',
                                                    color=file.color, alpha=0.7, label=trend_label)

                # Store trendline for hover if data lines are hidden
                if not show_data_lines:
                    self.plotted_lines.append({
                        'file': file,
                        'line': trend_line,
                        'freq_ghz': freq_ghz,
                        'values': trend
                    })

                # Display derivative values on polynomial trendline if requested
                if show_deriv_on_trend:
                    # Calculate derivative of polynomial (analytical derivative)
                    # polyder returns coefficients of derivative polynomial
                    deriv_coeffs = np.polyder(coeffs)
                    deriv_values = np.polyval(deriv_coeffs, freq_ghz)
                    
                    # Display derivative at a few points along the trendline
                    num_samples = min(3, len(freq_ghz) // 5 + 1) if len(freq_ghz) > 5 else 1
                    sample_indices = np.linspace(0, len(freq_ghz) - 1, num_samples, dtype=int)

                    for idx in sample_indices:
                        x_pos = freq_ghz[idx]
                        y_pos = trend[idx]
                        deriv_val = deriv_values[idx]
                        # Format the derivative value with appropriate precision
                        deriv_text = f"d/dGHz:\n{deriv_val:.2e}"
                        self.canvas.axes.annotate(
                            deriv_text,
                            xy=(x_pos, y_pos),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3),
                            color=file.color
                        )

            # Draw derivative if requested on a twin y-axis
            if show_deriv:
                if self.deriv_ax is None:
                    self.deriv_ax = self.canvas.axes.twinx()
                    # offset the right spine a bit
                    self.deriv_ax.spines['right'].set_position(('outward', 60))
                
                # If trendlines are shown, use polynomial derivative, otherwise numerical gradient
                if show_trendlines:
                    # Use analytical derivative of polynomial fit
                    poly_order = int(self.poly_order_spin.value())
                    coeffs = np.polyfit(freq_ghz, values_valid, poly_order)
                    deriv_coeffs = np.polyder(coeffs)
                    deriv = np.polyval(deriv_coeffs, freq_ghz)
                    deriv_label = f"{file.name} d/dGHz (poly{poly_order})"
                else:
                    # Use numerical gradient on raw data
                    deriv = np.gradient(values_valid, freq_ghz)
                    deriv_label = f"{file.name} d/dGHz"
                
                self.deriv_ax.plot(freq_ghz, deriv, linestyle=':', color=file.color, 
                                  alpha=0.9, label=deriv_label)
                self.deriv_ax.set_ylabel('d(metric)/dGHz', fontweight='bold')
                self.deriv_ax.legend(loc='upper right')
        
        # Labels and legend
        self.canvas.axes.set_xlabel('Frequency (GHz)', fontweight='bold')
        self.canvas.axes.set_ylabel(metric_label, fontweight='bold')
        self.canvas.axes.set_title(f'{metric_label} vs Frequency', fontweight='bold')

        # Only show legend if there are items to display
        handles, _ = self.canvas.axes.get_legend_handles_labels()
        if handles:
            self.canvas.axes.legend()

        self.canvas.axes.grid(True, alpha=0.3, linestyle='--', which='both' if (log_x or log_y) else 'major')
        
        self.canvas.draw()
    
    def update_statistics(self):
        """Update statistics display with comparison table format"""
        active_files = self.data_manager.get_active_files()
        metric_key = self.metric_combo.currentData()
        metric_label = self.metric_combo.currentText()

        if not active_files:
            self.stats_text.setPlainText("No active files")
            return

        # Collect all statistics and Cole-Cole data
        all_data = []
        for file in active_files:
            data = file.get_data(use_filtered=True)
            if data is None:
                continue

            stats = calculate_statistics(data, metric_key)
            if not stats:
                continue

            # Get slope for trendline derivative
            slope = stats.get('slope', np.nan)

            # Calculate Cole-Cole parameters if we have epsilon data
            cole_params = {}
            if 'epsilon_prime' in data and 'epsilon_double_prime' in data and 'frequency' in data:
                cole_params = calculate_cole_cole_parameters(
                    data['frequency'],
                    data['epsilon_prime'],
                    data['epsilon_double_prime']
                )

            all_data.append({
                'file': file,
                'name': file.name,
                'color': file.color,
                'stats': stats,
                'slope': slope,
                'cole': cole_params
            })

        # Create comparison table
        stats_html = "<html><body style='font-family: Arial, sans-serif; font-size: 9pt;'>"
        stats_html += f"<h4 style='margin: 5px 0;'>Trendline Derivatives & Cole-Cole Parameters</h4>"
        stats_html += f"<p style='margin: 5px 0; font-size: 8pt;'>Metric: {metric_label}</p>"

        # Create comparison table with compounds as columns
        stats_html += "<table border='1' cellpadding='4' cellspacing='0' style='font-size: 8pt; width: 100%;'>"

        # Header row
        stats_html += "<tr style='background-color: #f0f0f0;'>"
        stats_html += "<th style='min-width: 80px;'><b>Parameter</b></th>"
        for item in all_data:
            color = item['color']
            name = item['name']
            stats_html += f"<th style='background-color: {color}20; text-align: center; color: {color};'><b>{name}</b></th>"
        stats_html += "</tr>"

        # Row: Trendline Slope (Derivative)
        stats_html += "<tr>"
        stats_html += "<td style='font-weight: bold;'>Slope (/GHz)</td>"
        for item in all_data:
            slope = item['slope']
            slope_str = f"{slope:.3e}" if np.isfinite(slope) else "N/A"
            stats_html += f"<td style='text-align: right;'>{slope_str}</td>"
        stats_html += "</tr>"

        # Row: Mean
        stats_html += "<tr style='background-color: #f9f9f9;'>"
        stats_html += "<td style='font-weight: bold;'>Mean</td>"
        for item in all_data:
            mean = item['stats'].get('mean', np.nan)
            mean_str = f"{mean:.3e}" if np.isfinite(mean) else "N/A"
            stats_html += f"<td style='text-align: right;'>{mean_str}</td>"
        stats_html += "</tr>"

        # Row: Std Dev
        stats_html += "<tr>"
        stats_html += "<td style='font-weight: bold;'>Std Dev</td>"
        for item in all_data:
            std = item['stats'].get('std', np.nan)
            std_str = f"{std:.3e}" if np.isfinite(std) else "N/A"
            stats_html += f"<td style='text-align: right;'>{std_str}</td>"
        stats_html += "</tr>"

        # Row: Min
        stats_html += "<tr style='background-color: #f9f9f9;'>"
        stats_html += "<td style='font-weight: bold;'>Min</td>"
        for item in all_data:
            min_val = item['stats'].get('min', np.nan)
            min_str = f"{min_val:.3e}" if np.isfinite(min_val) else "N/A"
            stats_html += f"<td style='text-align: right;'>{min_str}</td>"
        stats_html += "</tr>"

        # Row: Max
        stats_html += "<tr>"
        stats_html += "<td style='font-weight: bold;'>Max</td>"
        for item in all_data:
            max_val = item['stats'].get('max', np.nan)
            max_str = f"{max_val:.3e}" if np.isfinite(max_val) else "N/A"
            stats_html += f"<td style='text-align: right;'>{max_str}</td>"
        stats_html += "</tr>"

        # Cole-Cole parameters (if available)
        if any(item['cole'] for item in all_data):
            stats_html += "<tr style='border-top: 2px solid #ccc;'>"
            stats_html += "<td colspan='{0}' style='font-weight: bold; background-color: #e8f4f8;'>Cole-Cole Parameters</td>".format(len(all_data) + 1)
            stats_html += "</tr>"

            # ε_s (Static Permittivity)
            stats_html += "<tr style='background-color: #f9f9f9;'>"
            stats_html += "<td style='font-weight: bold;'>ε_s (Static)</td>"
            for item in all_data:
                eps_s = item['cole'].get('epsilon_s', np.nan)
                eps_s_str = f"{eps_s:.3f}" if np.isfinite(eps_s) else "N/A"
                stats_html += f"<td style='text-align: right;'>{eps_s_str}</td>"
            stats_html += "</tr>"

            # ε_∞ (Infinite Freq Permittivity)
            stats_html += "<tr>"
            stats_html += "<td style='font-weight: bold;'>ε_∞ (∞ freq)</td>"
            for item in all_data:
                eps_inf = item['cole'].get('epsilon_inf', np.nan)
                eps_inf_str = f"{eps_inf:.3f}" if np.isfinite(eps_inf) else "N/A"
                stats_html += f"<td style='text-align: right;'>{eps_inf_str}</td>"
            stats_html += "</tr>"

            # Δε (Relaxation Strength)
            stats_html += "<tr style='background-color: #f9f9f9;'>"
            stats_html += "<td style='font-weight: bold;'>Δε (Strength)</td>"
            for item in all_data:
                delta_eps = item['cole'].get('relaxation_strength', np.nan)
                delta_eps_str = f"{delta_eps:.3f}" if np.isfinite(delta_eps) else "N/A"
                stats_html += f"<td style='text-align: right;'>{delta_eps_str}</td>"
            stats_html += "</tr>"

            # Max Loss (ε'')
            stats_html += "<tr>"
            stats_html += "<td style='font-weight: bold;'>Max ε''</td>"
            for item in all_data:
                max_loss = item['cole'].get('max_loss', np.nan)
                max_loss_str = f"{max_loss:.3e}" if np.isfinite(max_loss) else "N/A"
                stats_html += f"<td style='text-align: right;'>{max_loss_str}</td>"
            stats_html += "</tr>"

            # Loss Tangent
            stats_html += "<tr style='background-color: #f9f9f9;'>"
            stats_html += "<td style='font-weight: bold;'>tan(δ) (Loss)</td>"
            for item in all_data:
                loss_tan = item['cole'].get('loss_tangent', np.nan)
                loss_tan_str = f"{loss_tan:.3e}" if np.isfinite(loss_tan) else "N/A"
                stats_html += f"<td style='text-align: right;'>{loss_tan_str}</td>"
            stats_html += "</tr>"

        stats_html += "</table>"
        stats_html += "</body></html>"
        self.stats_text.setHtml(stats_html)

        # Store data for export
        self.current_stats_data = {
            'metric_label': metric_label,
            'metric_key': metric_key,
            'all_data': all_data
        }

    def copy_stats_to_clipboard(self):
        """Copy statistics table as tab-separated values to clipboard"""
        if not self.current_stats_data:
            QMessageBox.warning(self, "No Data", "No statistics to copy. Please refresh first.")
            return

        all_data = self.current_stats_data['all_data']
        metric_label = self.current_stats_data['metric_label']

        # Create tab-separated table
        lines = []
        lines.append(f"Trendline Derivatives & Cole-Cole Parameters")
        lines.append(f"Metric: {metric_label}")
        lines.append("")

        # Header row
        header = ["Parameter"]
        for item in all_data:
            header.append(item['name'])
        lines.append("\t".join(header))

        # Data rows
        param_rows = [
            ('Slope (/GHz)', lambda item: f"{item['slope']:.3e}" if np.isfinite(item['slope']) else "N/A"),
            ('Mean', lambda item: f"{item['stats'].get('mean', np.nan):.3e}" if np.isfinite(item['stats'].get('mean', np.nan)) else "N/A"),
            ('Std Dev', lambda item: f"{item['stats'].get('std', np.nan):.3e}" if np.isfinite(item['stats'].get('std', np.nan)) else "N/A"),
            ('Min', lambda item: f"{item['stats'].get('min', np.nan):.3e}" if np.isfinite(item['stats'].get('min', np.nan)) else "N/A"),
            ('Max', lambda item: f"{item['stats'].get('max', np.nan):.3e}" if np.isfinite(item['stats'].get('max', np.nan)) else "N/A"),
        ]

        for param_name, getter in param_rows:
            row = [param_name]
            for item in all_data:
                row.append(getter(item))
            lines.append("\t".join(row))

        # Cole-Cole parameters
        if any(item['cole'] for item in all_data):
            lines.append("")
            lines.append("Cole-Cole Parameters")

            cole_params = [
                ('ε_s (Static)', 'epsilon_s', '.3f'),
                ('ε_∞ (∞ freq)', 'epsilon_inf', '.3f'),
                ('Δε (Strength)', 'relaxation_strength', '.3f'),
                ('Max ε\'\'', 'max_loss', '.3e'),
                ('tan(δ) (Loss)', 'loss_tangent', '.3e'),
            ]

            for param_name, param_key, fmt in cole_params:
                row = [param_name]
                for item in all_data:
                    value = item['cole'].get(param_key, np.nan)
                    if np.isfinite(value):
                        row.append(f"{value:{fmt}}")
                    else:
                        row.append("N/A")
                lines.append("\t".join(row))

        # Copy to clipboard
        text = "\n".join(lines)
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, "Success", "Statistics copied to clipboard!")

    def export_stats_csv(self):
        """Export statistics to CSV file"""
        if not self.current_stats_data:
            QMessageBox.warning(self, "No Data", "No statistics to export. Please refresh first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics as CSV",
            str(self.current_directory / "statistics.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:
            all_data = self.current_stats_data['all_data']
            metric_label = self.current_stats_data['metric_label']

            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Header info
                writer.writerow(["Trendline Derivatives & Cole-Cole Parameters"])
                writer.writerow([f"Metric: {metric_label}"])
                writer.writerow([])

                # Header row
                header = ["Parameter"]
                for item in all_data:
                    header.append(item['name'])
                writer.writerow(header)

                # Data rows
                param_rows = [
                    ('Slope (/GHz)', lambda item: f"{item['slope']:.3e}" if np.isfinite(item['slope']) else "N/A"),
                    ('Mean', lambda item: f"{item['stats'].get('mean', np.nan):.3e}" if np.isfinite(item['stats'].get('mean', np.nan)) else "N/A"),
                    ('Std Dev', lambda item: f"{item['stats'].get('std', np.nan):.3e}" if np.isfinite(item['stats'].get('std', np.nan)) else "N/A"),
                    ('Min', lambda item: f"{item['stats'].get('min', np.nan):.3e}" if np.isfinite(item['stats'].get('min', np.nan)) else "N/A"),
                    ('Max', lambda item: f"{item['stats'].get('max', np.nan):.3e}" if np.isfinite(item['stats'].get('max', np.nan)) else "N/A"),
                ]

                for param_name, getter in param_rows:
                    row = [param_name]
                    for item in all_data:
                        row.append(getter(item))
                    writer.writerow(row)

                # Cole-Cole parameters
                if any(item['cole'] for item in all_data):
                    writer.writerow([])
                    writer.writerow(["Cole-Cole Parameters"])

                    cole_params = [
                        ('ε_s (Static)', 'epsilon_s', '.3f'),
                        ('ε_∞ (∞ freq)', 'epsilon_inf', '.3f'),
                        ('Δε (Strength)', 'relaxation_strength', '.3f'),
                        ('Max ε\'\'', 'max_loss', '.3e'),
                        ('tan(δ) (Loss)', 'loss_tangent', '.3e'),
                    ]

                    for param_name, param_key, fmt in cole_params:
                        row = [param_name]
                        for item in all_data:
                            value = item['cole'].get(param_key, np.nan)
                            if np.isfinite(value):
                                row.append(f"{value:{fmt}}")
                            else:
                                row.append("N/A")
                        writer.writerow(row)

            self.current_directory = Path(file_path).parent
            QMessageBox.information(self, "Success", f"Statistics exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export CSV:\n{str(e)}")

    def export_stats_json(self):
        """Export statistics to JSON file"""
        if not self.current_stats_data:
            QMessageBox.warning(self, "No Data", "No statistics to export. Please refresh first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics as JSON",
            str(self.current_directory / "statistics.json"),
            "JSON Files (*.json);;All Files (*)"
        )

        if not file_path:
            return

        try:
            all_data = self.current_stats_data['all_data']
            metric_label = self.current_stats_data['metric_label']

            # Build export data structure
            export_data = {
                'title': 'Trendline Derivatives & Cole-Cole Parameters',
                'metric': metric_label,
                'compounds': []
            }

            for item in all_data:
                compound_data = {
                    'name': item['name'],
                    'color': item['color'],
                    'slope': float(item['slope']) if np.isfinite(item['slope']) else None,
                    'statistics': {
                        'mean': float(item['stats'].get('mean', np.nan)) if np.isfinite(item['stats'].get('mean', np.nan)) else None,
                        'median': float(item['stats'].get('median', np.nan)) if np.isfinite(item['stats'].get('median', np.nan)) else None,
                        'std': float(item['stats'].get('std', np.nan)) if np.isfinite(item['stats'].get('std', np.nan)) else None,
                        'min': float(item['stats'].get('min', np.nan)) if np.isfinite(item['stats'].get('min', np.nan)) else None,
                        'max': float(item['stats'].get('max', np.nan)) if np.isfinite(item['stats'].get('max', np.nan)) else None,
                        'range': float(item['stats'].get('range', np.nan)) if np.isfinite(item['stats'].get('range', np.nan)) else None,
                        'integral': float(item['stats'].get('integral', np.nan)) if np.isfinite(item['stats'].get('integral', np.nan)) else None,
                    }
                }

                # Add Cole-Cole parameters if available
                if item['cole']:
                    compound_data['cole_cole'] = {
                        'epsilon_s': float(item['cole'].get('epsilon_s', np.nan)) if np.isfinite(item['cole'].get('epsilon_s', np.nan)) else None,
                        'epsilon_inf': float(item['cole'].get('epsilon_inf', np.nan)) if np.isfinite(item['cole'].get('epsilon_inf', np.nan)) else None,
                        'relaxation_strength': float(item['cole'].get('relaxation_strength', np.nan)) if np.isfinite(item['cole'].get('relaxation_strength', np.nan)) else None,
                        'max_loss': float(item['cole'].get('max_loss', np.nan)) if np.isfinite(item['cole'].get('max_loss', np.nan)) else None,
                        'loss_tangent': float(item['cole'].get('loss_tangent', np.nan)) if np.isfinite(item['cole'].get('loss_tangent', np.nan)) else None,
                    }

                export_data['compounds'].append(compound_data)

            # Write JSON
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.current_directory = Path(file_path).parent
            QMessageBox.information(self, "Success", f"Statistics exported to:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export JSON:\n{str(e)}")

    def load_reference_file(self, liquid_type: str):
        """Load a reference liquid S1P file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {liquid_type.capitalize()} S1P File",
            str(self.current_directory),
            "S1P Files (*.s1p);;All Files (*)"
        )

        if file_path:
            self.current_directory = Path(file_path).parent
            self.ref_files[liquid_type] = file_path

            # Update label
            filename = Path(file_path).name
            label_map = {
                'water': self.lbl_water,
                'ethanol': self.lbl_ethanol,
                'isopropanol': self.lbl_iso
            }
            label_map[liquid_type].setText(filename)
            label_map[liquid_type].setStyleSheet("color: green; font-size: 9px;")

    def perform_calibration(self):
        """Perform probe calibration with loaded reference files"""
        # Check if all files are loaded
        required = {'water', 'ethanol', 'isopropanol'}
        if not required.issubset(self.ref_files.keys()):
            missing = required - set(self.ref_files.keys())
            QMessageBox.warning(
                self,
                "Missing Files",
                f"Please load S1P files for: {', '.join(missing)}"
            )
            return

        try:
            self.cal_status_label.setText("Status: Calibrating...")
            self.cal_status_label.setStyleSheet("color: orange; font-weight: bold;")
            QApplication.processEvents()

            # Load reference files
            import skrf as rf

            s11_measurements = {}
            for liquid in ['water', 'ethanol', 'isopropanol']:
                network = rf.Network(str(self.ref_files[liquid]))
                s11_measurements[liquid] = {
                    'frequency': network.f,
                    's11_complex': network.s[:, 0, 0]
                }

            # Load reference data
            reference_data = self.calibration.load_reference_data(temperature=25.0)

            # Perform calibration
            if self.calibration.calibrate(s11_measurements, reference_data):
                # Update data manager with calibration
                self.data_manager.set_calibration(self.calibration)

                self.cal_status_label.setText("Status: ✓ Calibrated (25°C)")
                self.cal_status_label.setStyleSheet("color: green; font-weight: bold;")

                # Reload all files to apply calibration
                for file in self.data_manager.get_active_files():
                    file.load()

                self.update_plot()
                self.update_statistics()

                QMessageBox.information(
                    self,
                    "Success",
                    "Calibration completed successfully!\n\n"
                    "All loaded files will now use the calibration."
                )
            else:
                self.cal_status_label.setText("Status: Calibration failed")
                self.cal_status_label.setStyleSheet("color: red; font-weight: bold;")
                QMessageBox.critical(
                    self,
                    "Error",
                    "Calibration failed. Check that all S1P files are valid and have matching frequency points."
                )

        except Exception as e:
            self.cal_status_label.setText("Status: Error")
            self.cal_status_label.setStyleSheet("color: red; font-weight: bold;")
            QMessageBox.critical(self, "Error", f"Calibration error:\n{str(e)}")

    def save_calibration(self):
        """Save current calibration to file"""
        if not self.calibration.is_calibrated():
            QMessageBox.warning(
                self,
                "No Calibration",
                "Please perform calibration first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration",
            str(self.current_directory / "probe_calibration.pkl"),
            "Pickle Files (*.pkl);;All Files (*)"
        )

        if file_path:
            if self.calibration.save(file_path):
                self.current_directory = Path(file_path).parent
                QMessageBox.information(
                    self,
                    "Success",
                    f"Calibration saved to:\n{file_path}"
                )

    def load_calibration(self):
        """Load a previously saved calibration"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Calibration",
            str(self.current_directory),
            "Pickle Files (*.pkl);;All Files (*)"
        )

        if file_path:
            try:
                loaded_cal = ProbeCalibration.load(file_path)
                if loaded_cal and loaded_cal.is_calibrated():
                    self.calibration = loaded_cal
                    self.data_manager.set_calibration(self.calibration)
                    self.current_directory = Path(file_path).parent

                    self.cal_status_label.setText("Status: ✓ Calibrated (loaded)")
                    self.cal_status_label.setStyleSheet("color: green; font-weight: bold;")

                    # Reload all files to apply calibration
                    for file in self.data_manager.get_active_files():
                        file.load()

                    self.update_plot()
                    self.update_statistics()

                    QMessageBox.information(
                        self,
                        "Success",
                        f"Calibration loaded from:\n{file_path}"
                    )
                else:
                    QMessageBox.critical(self, "Error", "Invalid or empty calibration file.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load calibration:\n{str(e)}")


def main():
    """Main entry point for the S1P GUI application"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    window = S1PMainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
