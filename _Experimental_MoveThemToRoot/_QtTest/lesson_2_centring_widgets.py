import sys
from PyQt5 import QtWidgets, QtGui

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080


def window():
    app = QtWidgets.QApplication(sys.argv)
    
    w = QtWidgets.QWidget()
    w.setWindowTitle('Lesson 2')

    # ---- Lables: text and image 
    l = QtWidgets.QLabel('Look at Me')
    b = QtWidgets.QPushButton('Push me')
    
    # ---- Vertical Layout
    h_box = QtWidgets.QHBoxLayout()
    h_box.addStretch() # 1) To center widget 'l' from left
    h_box.addWidget(l)
    h_box.addStretch() # 2) To center widget 'l' from right

    # ---- Horizontal Layout
    v_box = QtWidgets.QVBoxLayout()
    v_box.addWidget(b)
    v_box.addLayout(h_box)  # Adding Layout to Layout to center Text_Label

    # ---- Set Layout and Show
    w.setLayout(v_box)
    w.show()
    
    # ----
    sys.exit(app.exec_())


window()