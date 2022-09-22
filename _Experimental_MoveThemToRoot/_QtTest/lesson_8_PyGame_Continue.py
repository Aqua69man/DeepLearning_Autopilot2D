import sys

# ---------- Qt
from PyQt5.QtWidgets import (QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QApplication, QWidget)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# ---------- PyGame
import pygame


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Merging PyGame to PyQt5'
        self.init_ui()
    
    def init_ui(self):     
        self.l = QLabel()

        v_box = QVBoxLayout()
        v_box.addWidget(self.l)

        self.setLayout(v_box)
        self.setWindowTitle(self.title)
        self.show()

    def update_image(self, surface):
        w = surface.get_width()
        h = surface.get_height()
        data = surface.get_buffer().raw
        self.q_image = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGB32)      # q_image should be a SELF member
        q_pixmap = QtGui.QPixmap.fromImage(self.q_image) 
        self.l.setPixmap(q_pixmap)


app = QApplication(sys.argv)
a_window = Window()


pygame.init()
# surface = pygame.display.set_mode((1920, 1080))
surface = pygame.Surface((1920, 1080))

x = 30; y = 30
done = False
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
    x += 1
    y += 1

    surface.fill((0,0,0))
    pygame.draw.rect(surface, (0, 128, 255), pygame.Rect(x, y, 60, 60))
    # pygame.display.flip()

    a_window.update_image(surface)


# --------------------
# sys.exit(app.exec_())
# --------------------




