"""
Database browser widget for compound selection
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QLabel, QLineEdit, QTreeWidget, QTreeWidgetItem, QGroupBox,
    QSplitter, QTextEdit, QAbstractItemView, QListWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from pathlib import Path
from typing import List, Optional

from .database_loader import (
    find_all_compounds, search_compounds, Compound, Measurement,
    get_compound_measurements_as_tuples
)


class CompoundBrowserWidget(QWidget):
    """Widget for browsing and selecting compounds from the database"""
    
    # Signal emitted when measurements are selected: (name, filepath, description) tuples
    measurements_selected = pyqtSignal(list)
    # Signal emitted when clear all is requested
    clear_all_requested_signal = pyqtSignal()
    
    def __init__(self, database_path: Path, parent=None):
        super().__init__(parent)
        self.database_path = database_path
        self.compounds: List[Compound] = []
        self.filtered_compounds: List[Compound] = []
        self.selected_compound: Optional[Compound] = None
        
        self.setup_ui()
        self.load_database()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search by name, formula, CAS, functional groups...")
        self.search_box.textChanged.connect(self.on_search_changed)
        search_layout.addWidget(self.search_box)
        
        self.refresh_btn = QPushButton("🔄")
        self.refresh_btn.setMaximumWidth(40)
        self.refresh_btn.setToolTip("Reload database")
        self.refresh_btn.clicked.connect(self.load_database)
        search_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(search_layout)
        
        # Compound list
        list_label = QLabel("Compounds:")
        list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(list_label)
        
        self.compound_list = QListWidget()
        self.compound_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.compound_list.currentRowChanged.connect(self.on_compound_selected)
        layout.addWidget(self.compound_list)
        
        # Details panel
        details_group = QGroupBox("Compound Details")
        details_layout = QVBoxLayout()
        details_group.setLayout(details_layout)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(200)
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(details_group)
        
        # Measurements list
        meas_label = QLabel("Measurements:")
        meas_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(meas_label)
        
        self.measurements_list = QListWidget()
        self.measurements_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.measurements_list)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        
        self.add_selected_btn = QPushButton("Add Selected")
        self.add_selected_btn.clicked.connect(self.add_selected_measurements)
        self.add_selected_btn.setEnabled(False)
        btn_layout.addWidget(self.add_selected_btn)
        
        self.add_all_btn = QPushButton("Add All")
        self.add_all_btn.clicked.connect(self.add_all_measurements)
        self.add_all_btn.setEnabled(False)
        btn_layout.addWidget(self.add_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Clear button
        self.clear_all_btn = QPushButton("Clear All Loaded Files")
        self.clear_all_btn.clicked.connect(self.clear_all_requested)
        self.clear_all_btn.setStyleSheet("background-color: #ff5555; color: white; font-weight: bold; padding: 8px;")
        layout.addWidget(self.clear_all_btn)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label)
    
    def load_database(self):
        """Load all compounds from the database"""
        self.status_label.setText("Loading database...")
        self.compounds = find_all_compounds(self.database_path)
        self.filtered_compounds = self.compounds.copy()
        self.update_compound_list()
        self.status_label.setText(f"Loaded {len(self.compounds)} compounds")
    
    def update_compound_list(self):
        """Update the compound list display"""
        self.compound_list.clear()
        
        for compound in self.filtered_compounds:
            display_name = compound.get_display_name()
            item_text = f"{display_name} - {compound.chemical_formula}"
            self.compound_list.addItem(item_text)
    
    def on_search_changed(self, query: str):
        """Handle search query changes"""
        if not query:
            self.filtered_compounds = self.compounds.copy()
        else:
            self.filtered_compounds = search_compounds(self.compounds, query)
        
        self.update_compound_list()
        self.status_label.setText(f"Found {len(self.filtered_compounds)} compounds")
    
    def on_compound_selected(self, index: int):
        """Handle compound selection"""
        if index < 0 or index >= len(self.filtered_compounds):
            self.selected_compound = None
            self.details_text.clear()
            self.measurements_list.clear()
            self.add_selected_btn.setEnabled(False)
            self.add_all_btn.setEnabled(False)
            return
        
        self.selected_compound = self.filtered_compounds[index]
        self.update_details_panel()
        self.update_measurements_list()
        
        # Enable buttons if there are measurements
        has_measurements = len(self.selected_compound.measurements) > 0
        self.add_selected_btn.setEnabled(has_measurements)
        self.add_all_btn.setEnabled(has_measurements)
    
    def update_details_panel(self):
        """Update the compound details panel"""
        if not self.selected_compound:
            return
        
        c = self.selected_compound
        
        html = f"""
        <html>
        <body style='font-family: sans-serif;'>
            <h3 style='margin-top: 0;'>{c.common_name}</h3>
            <table style='width: 100%;'>
                <tr><td><b>Chemical Name:</b></td><td>{c.chemical_name}</td></tr>
                <tr><td><b>Formula:</b></td><td>{c.chemical_formula}</td></tr>
                <tr><td><b>CAS Number:</b></td><td>{c.cas_number}</td></tr>
                <tr><td><b>Molecular Weight:</b></td><td>{c.molecular_weight} g/mol</td></tr>
                <tr><td><b>Functional Groups:</b></td><td>{', '.join(c.functional_groups)}</td></tr>
                <tr><td><b>Measurements:</b></td><td>{len(c.measurements)}</td></tr>
            </table>
        </body>
        </html>
        """
        
        self.details_text.setHtml(html)
    
    def update_measurements_list(self):
        """Update the measurements list"""
        self.measurements_list.clear()
        
        if not self.selected_compound:
            return
        
        measurement_tuples = get_compound_measurements_as_tuples(self.selected_compound)
        
        for name, filepath, description in measurement_tuples:
            item_text = f"{description}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, (name, filepath, description))
            
            # Color code based on purity/type
            if filepath.exists():
                item.setForeground(QColor("black"))
            else:
                item.setForeground(QColor("red"))
                item.setText(item_text + " [FILE NOT FOUND]")
            
            self.measurements_list.addItem(item)
    
    def add_selected_measurements(self):
        """Add selected measurements to the analysis"""
        selected_items = self.measurements_list.selectedItems()
        
        if not selected_items or not self.selected_compound:
            return
        
        measurements = []
        for item in selected_items:
            data = item.data(Qt.UserRole)
            if data:
                # data is (name, filepath, description)
                # Add compound object for metadata access
                measurements.append((data, self.selected_compound))
        
        self.measurements_selected.emit(measurements)
        self.status_label.setText(f"Added {len(measurements)} measurement(s)")
    
    def add_all_measurements(self):
        """Add all measurements of the selected compound"""
        if not self.selected_compound:
            return
        
        measurements = []
        for i in range(self.measurements_list.count()):
            item = self.measurements_list.item(i)
            data = item.data(Qt.UserRole)
            if data:
                # data is (name, filepath, description)
                # Add compound object for metadata access
                measurements.append((data, self.selected_compound))
        
        self.measurements_selected.emit(measurements)
        self.status_label.setText(f"Added {len(measurements)} measurement(s)")
    
    def get_database_path(self) -> Path:
        """Get the current database path"""
        return self.database_path
    
    def set_database_path(self, path: Path):
        """Set a new database path and reload"""
        self.database_path = path
        self.load_database()
    
    def clear_all_requested(self):
        """Emit signal to clear all loaded files"""
        self.clear_all_requested_signal.emit()
        self.status_label.setText("Cleared all loaded files")
