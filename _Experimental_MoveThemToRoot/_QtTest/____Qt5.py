import sys
import random
 
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton

# -------------------
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D
try:
    from scipy.constants import g, pi
except ImportError:
    g = 9.80665
    from math import pi
# ------------------
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGroupBox, QRadioButton, QSlider
# ------------------
 
class App(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.title = 'PyQt5 matplotlib example - pythonspot.com'
        self.width = 1920
        self.height = 1080
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        m = PlotCanvas(self, width=17, height=10.82)
        m.move(0,0)
 
        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This s an example button')
        button.move(1700,0)
        button.resize(200,100)

        # --
        slider = QSlider(Qt.Horizontal, self)
        slider.setFocusPolicy(Qt.NoFocus)
        slider.move(1700,120)
        slider.resize(200,50)

        self.show()

    def createExampleGroup(self):
        groupBox = QGroupBox("Slider Example")

        radio1 = QRadioButton("&Radio horizontal slider")

        slider = QSlider(Qt.Horizontal)
        slider.setFocusPolicy(Qt.StrongFocus)
        slider.setTickPosition(QSlider.TicksBothSides)
        slider.setTickInterval(10)
        slider.setSingleStep(1)

        radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radio1)
        vbox.addWidget(slider)
        vbox.addStretch(1)
        groupBox.setLayout(vbox)

        return groupBox
 
 
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.figure.add_subplot(1, 1, 1, projection='3d')
 
        FigureCanvas.__init__(self, self.figure)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):     
        xx = np.linspace(0.5, 50, 10)
        yy = np.linspace(1e-6, pi / 2, 10)
        xx, yy = np.meshgrid(xx, yy)
        points = np.stack([xx, yy], axis=2)
        def s(coords):
            v, angle = coords
            return v * v * np.sin(2 * angle) / g 
        dist = np.apply_along_axis(s, -1, points)

        self.axes = self.figure.add_subplot(1, 1, 1, projection='3d')
        self.axes.plot_surface(xx, yy, dist)
        self.axes.plot_surface(xx, yy, dist + 10, alpha=0.3)  # верхняя граница ответа "попал" в тренировочных данных
        self.axes.plot_surface(xx, yy, dist - 10, alpha=0.3)  # нижняя граница ответа "попал" в тренировочных данных

        plt.xlabel(r"$v_x$")
        plt.ylabel(r"$v_y$")
        xlim = plt.xlim()  # чтобы построить следующий график в том же масштабе
        ylim = plt.ylim()  # чтобы построить следующий график в том же масштабе
        zlim = self.axes.get_zlim()
        self.axes.set_zlim(0, 200)
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())