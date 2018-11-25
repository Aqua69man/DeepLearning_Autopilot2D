# ---- QT 
import sys
from PyQt5.QtWidgets import (QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QApplication, QWidget)
from PyQt5.QtCore import Qt

# ---- Matplotlib 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import g, pi


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Lesson 5 - Slider_Active'
        # self.left = 100
        # self.top = 100
        # self.width = 100
        # self.height = 100
        
        self.init_ui()
    
    def init_ui(self):
        self.le = QLineEdit()
        self.b1 = QPushButton('Clear')
        self.b2 = QPushButton('Print')

        self.s1 = QSlider(Qt.Horizontal)
        self.s1.setMinimum(1)
        self.s1.setMaximum(10)
        self.s1.setValue(5)
        self.s1.setTickInterval(1) # discretization frequency
        self.s1.setTickPosition(QSlider.TicksRight) # QSlider.TicksBelow - determins the side where the stick points

        self.c1 = QComboBox()
        self.c1.addItem('1')
        self.c1.addItem('2')
        self.c1.addItem('3')
        self.c1.addItem('4')
        self.c1.addItem('5')

        v_box = QVBoxLayout()
        v_box.addWidget(self.le)
        v_box.addWidget(self.b1)
        v_box.addWidget(self.b2)
        v_box.addWidget(self.s1)
        v_box.addWidget(self.c1)

        self.b1.clicked.connect( lambda: self.btn_clk(self.b1, 'Hello from Clear') ) # Using Lamba as we have more than one func arg
        self.b2.clicked.connect( lambda: self.btn_clk(self.b2, 'Hello from Print') ) # Using Lamba as we have more than one func arg
        self.s1.valueChanged.connect( self.s_change )  # Using Callback as we only have one func arg
        self.c1.activated[str].connect( self.c_change )

        self.setLayout(v_box)
        # self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowTitle(self.title)
        self.show()

    def btn_clk(self, b, string):
        if b.text() == 'Print':
            print( self.le.text() )
        else: 
            self.le.clear()
        print(string)

    def s_change(self):
        s_val = str( self.s1.value() )
        self.le.setText(s_val)

    def c_change(self, text):
        box_val = str( text )
        self.le.setText(box_val)


app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())