# ---- QT 
import sys
import random   
from PyQt5.QtWidgets import (QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QSizePolicy, QApplication, QWidget)
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
        self.init_ui()
    
    def init_ui(self): 
        self.b1 = QPushButton('Push')
        self.graphCanvas = PlotCanvas(self, width=17, height=10)

        h_box = QHBoxLayout()
        h_box.addWidget(self.graphCanvas)
        h_box.addWidget(self.b1)

        self.b1.clicked.connect( lambda: self.btn_clk(self.b1, 'Hello from Push') ) # Using Lamba as we have more than one func arg

        self.setLayout(h_box)
        self.setGeometry(100, 100, self.width(), self.height())
        self.show()

    def btn_clk(self, b, string):
        print(string)
        
        self.graphCanvas.plot()



class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(1, 1, 1)
 
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.plot()

    def plot(self):  
        self.y = []
        for i in range(100):
            self.axes.clear()
            self.y.append(random.random())
            self.axes.plot(self.y)
        # self.axes.plot(self.y)


app = QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())