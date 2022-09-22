import sys
from PyQt5 import QtWidgets, QtGui

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080


def window():
    app = QtWidgets.QApplication(sys.argv)
    
    w = QtWidgets.QWidget()
    w.setWindowTitle('Lesson 2')
    w.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

    # ---- Lables: text and image 
    l1 = QtWidgets.QLabel(w)
    l1.setText('Look at Me')
    l1.move(WINDOW_WIDTH/2 - l1.width()/2, 20)

    l2 = QtWidgets.QLabel(w)
    l2.setPixmap(QtGui.QPixmap('flt.png'))
    l2.move(320, 100)

    # ---- Button
    b = QtWidgets.QPushButton(w)
    b.resize(100, 100)
    b.setText('Push me')

    w.show()
    sys.exit(app.exec_())

window()