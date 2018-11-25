import sys

# ---------- Qt
from PyQt5.QtWidgets import (QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QApplication, QWidget)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# ---------- PyGame
import pygame


class Window(QWidget):
    def __init__(self, surface):
        super().__init__()

        self.title = 'Merging PyGame to PyQt5'
        self.surface = surface
        self.init_ui()
    
    def init_ui(self): 
        w = self.surface.get_width()
        h = self.surface.get_height()
        data = self.surface.get_buffer().raw
        q_image = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGB32)
        
        self.l = QLabel()
        self.l.setPixmap(QtGui.QPixmap.fromImage(q_image) )

        v_box = QVBoxLayout()
        v_box.addWidget(self.l)

        self.setLayout(v_box)
        self.setWindowTitle(self.title)
        self.show()


pygame.init()
# surface = pygame.display.set_mode((400, 300))
surface = pygame.Surface((400, 300))
pygame.draw.rect(surface, (0, 128, 255), pygame.Rect(30, 30, 60, 60))


# --------------------
app = QApplication(sys.argv)
a_window = Window(surface)
sys.exit(app.exec_())
# --------------------




