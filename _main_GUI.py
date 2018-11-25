import sys
import numpy as np
import random

# ---------- Qt
from PyQt5.QtWidgets import (QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QApplication, QWidget)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# --------- Game NN
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Car Training - Newral Network'
        # self.left = 100
        # self.top = 100
        # self.width = 1920
        # self.height = 1080
        self.init_ui()
    
    def init_ui(self): 
        self.l0_pygame_label = QLabel()

        self.l1 = QLabel('1st inner layer:')
        self.s1 = QSlider(Qt.Horizontal)
        self.s1.setMinimum(0)
        self.s1.setMaximum(10)
        self.s1.setValue(6)
        self.s1.setTickInterval(1) # discretization frequency
        self.s1.setTickPosition(QSlider.TicksRight) # QSlider.TicksBelow - determins the side where the stick points

        self.l2 = QLabel('2nd innder layer:')
        self.s2 = QSlider(Qt.Horizontal)
        self.s2.setMinimum(0)
        self.s2.setMaximum(10)
        self.s2.setValue(0)

        self.l3 = QLabel('3rd innder layer:')
        self.s3 = QSlider(Qt.Horizontal)
        self.s3.setMinimum(0)
        self.s3.setMaximum(10)
        self.s3.setValue(0)
        self.s3.setTickInterval(1)
        self.s3.setTickPosition(QSlider.TicksRight)


        self.l4 = QLabel('Batch size:')
        self.le4 = QLineEdit()
        self.le4.setText('50')

        self.l5 = QLabel('Learning rate:')
        self.c5 = QComboBox()
        self.c5.addItem('0.01')
        self.c5.addItem('0.05')
        self.c5.addItem('0.1')
        self.c5.addItem('0.5')
        self.c5.addItem('1')
        self.c5.addItem('5')
        self.c5.addItem('10')
        self.c5.setCurrentIndex(1)

        self.l6 = QLabel('L1:')
        self.c6 = QComboBox()
        self.c6.addItem('0')
        self.c6.addItem('0.0001')
        self.c6.addItem('0.0005')
        self.c6.addItem('0.001')
        self.c6.addItem('0.005')
        self.c6.addItem('0.01')
        self.c6.addItem('0.05')
        self.c6.addItem('0.1')

        self.l7 = QLabel('L2:')
        self.c7 = QComboBox()
        self.c7.addItem('0')
        self.c7.addItem('0.0001')
        self.c7.addItem('0.0005')
        self.c7.addItem('0.001')
        self.c7.addItem('0.005')
        self.c7.addItem('0.01')
        self.c7.addItem('0.05')
        self.c7.addItem('0.1')

        self.b8 = QPushButton('Train')

        self.b9 = QPushButton('Evaluate')

        v_box = QVBoxLayout()
        v_box.addWidget(self.l1)
        v_box.addWidget(self.s1)
        v_box.addWidget(self.l2)
        v_box.addWidget(self.s2)
        v_box.addWidget(self.l3)
        v_box.addWidget(self.s3)
        v_box.addWidget(self.l4)
        v_box.addWidget(self.le4)
        v_box.addWidget(self.l5)
        v_box.addWidget(self.c5)
        v_box.addWidget(self.l6)
        v_box.addWidget(self.c6)
        v_box.addWidget(self.l7)
        v_box.addWidget(self.c7)
        v_box.addWidget(self.b8)
        v_box.addWidget(self.b9)

        h_box = QHBoxLayout()
        h_box.addWidget(self.l0_pygame_label)
        h_box.addLayout(v_box)

        self.s1.valueChanged.connect(self.s1_reaction)
        self.s2.valueChanged.connect(self.s2_reaction)
        self.s3.valueChanged.connect(self.s3_reaction)
        self.b8.clicked.connect(self.b8_reaction)
        self.b9.clicked.connect(self.b9_reaction)

        self.setLayout(h_box)
        self.setWindowTitle(self.title)
        self.show()

    def s1_reaction(self, text):
        new_text = '1st inner layer: ' + str(text)  
        self.l1.setText(new_text)
    def s2_reaction(self, text):
        new_text = '2nd inner layer: ' + str(text)  
        self.l2.setText(new_text)
    def s3_reaction(self, text):
        new_text = '3rd inner layer: ' + str(text)  
        self.l3.setText(new_text)

    def b8_reaction(self):
        layer1 = int( self.s1.value() ) 
        layer2 = int( self.s2.value() )
        layer3 = int( self.s3.value() )
        batch_size = int( self.le4.text() )
        learning_rate = float( self.c5.currentText() )
        l1_reg = float( self.c6.currentText() )
        l2_reg = float( self.c7.currentText() )

        print( '1st inner layer: ', str(layer1) )
        print( '2nd inner layer: ', str(layer2) )
        print( '3rd inner layer: ', str(layer3) )
        print( 'Batch size: ', str(batch_size) )
        print( 'Learning rate: ', str(learning_rate) )
        print( 'L1: ', str(l1_reg) )
        print( 'L2: ', str(l2_reg) )
        
        # -------- Init NN and PyGame
        # learning_curve_by_network_structure(layer1=layer1, layer2=layer2, layer3=layer3, batch_size=batch_size, learning_rate=learning_rate)
        steps = 5000
        seed = 15
        m = generate_map(8, 5, 3, 3)
        np.random.seed(seed)
        random.seed(seed)

        # agent = SimpleCarAgent(epochs=15, mini_batch_size=50, eta=0.05)
        agent = SimpleCarAgent(epochs=15, mini_batch_size=batch_size, eta=learning_rate)
        scw = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
        scw.set_agents([agent])
        scw.run(steps, window=self)
    
    
    def b9_reaction(self):
        import os, glob
        work_dir = os.getcwd()

        agent_fils_names = []
        os.chdir(work_dir)
        for file in glob.glob("*.txt"):
            agent_fils_names.append(file)
        
        steps = 5000
        seed = 15
        m = generate_map(8, 5, 3, 3)
        np.random.seed(seed)
        random.seed(seed)

        agent = SimpleCarAgent.from_file(agent_fils_names[0])
        sw = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2)
        evl = sw.evaluate_agent(agent, steps, window=self)
        print(evl)
    
    def set_pygame_image(self, surface):
        w = surface.get_width()
        h = surface.get_height()
        data = surface.get_buffer().raw

        self.setGeometry(100, 100, w+300, h)
        self.l0_qimage = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGB32)
        self.l0_pygame_label.setPixmap(QtGui.QPixmap.fromImage(self.l0_qimage) )


    def closeEvent(self, event): 
        super().closeEvent()
        pygame.quit()  
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()   # window = Window(pygame.Surface((600,400)))
    sys.exit(app.exec_())
