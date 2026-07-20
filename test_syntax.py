import sys
from PyQt5.QtWidgets import QApplication
from s1p_gui.gui_main import S1PMainWindow

app = QApplication(sys.argv)
try:
    window = S1PMainWindow()
    print("Loaded without syntax errors!")
except Exception as e:
    print(f"Error: {e}")
