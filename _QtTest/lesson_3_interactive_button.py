import sys
from PyQt5 import QtWidgets

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
    
    def init_ui(self):
        self.b = QtWidgets.QPushButton('Push me')
        self.l = QtWidgets.QLabel('I haven not been clicked yet')
        
        # Adding 'l' to H_Layout to ceter Horizontaly
        h_box = QtWidgets.QHBoxLayout()
        h_box.addStretch()
        h_box.addWidget(self.l)
        h_box.addStretch()

        # Adding H_Layout and 'l' to V_Layout 
        v_box = QtWidgets.QVBoxLayout()
        v_box.addWidget(self.b) 
        v_box.addLayout(h_box)

        self.b.clicked.connect(self.btn_click)

        self.setWindowTitle('Lesson 3 - interactive button')
        self.setLayout(v_box)
        self.show()

    def btn_click(self):
        self.l.setText('I have been clicked')


app = QtWidgets.QApplication(sys.argv)
a_window = Window()
sys.exit(app.exec_())