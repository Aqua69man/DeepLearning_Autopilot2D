import sys
import numpy as np
import random

# ---------- Qt
from PyQt5.QtWidgets import (QLabel, QLineEdit, QSlider, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QApplication, QWidget, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui

# --------- Game NN
from cars.world import SimpleCarWorld
from cars.agent import SimpleCarAgent
from cars.physics import SimplePhysics
from cars.track import generate_map

# --------- MatplotLib
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# from mpl_toolkits.mplot3d import Axes3D
# from scipy.constants import g, pi

class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = 'Car Training - Newral Network'
        self.left = 100;  self.top = 100;  self.width = 1920;  self.height = 1080
        self.init_ui()
    def init_ui(self): 
        self.l_pygameBackbuff = QLabel()

        self.l_layer1 = QLabel('1st hidden layer:')
        self.s_layer1 = QSlider(Qt.Horizontal)
        self.s_layer1.setMinimum(0)
        self.s_layer1.setMaximum(20)
        self.s_layer1.setValue(9)
        self.s_layer1.setTickInterval(1) # discretization frequency
        self.s_layer1.setTickPosition(QSlider.TicksRight) # QSlider.TicksBelow - determins the side where the stick points

        self.l_layer2 = QLabel('2nd hidden layer:')
        self.s_layer2 = QSlider(Qt.Horizontal)
        self.s_layer2.setMinimum(0)
        self.s_layer2.setMaximum(20)
        self.s_layer2.setValue(0)
        self.s_layer2.setTickInterval(1)
        self.s_layer2.setTickPosition(QSlider.TicksRight)

        self.l_layer3 = QLabel('3rd hidden layer:')
        self.s_layer3 = QSlider(Qt.Horizontal)
        self.s_layer3.setMinimum(0)
        self.s_layer3.setMaximum(20)
        self.s_layer3.setValue(0)
        self.s_layer3.setTickInterval(1)
        self.s_layer3.setTickPosition(QSlider.TicksRight)

        self.l_epoches = QLabel('Epoches:')
        self.le_epoches = QLineEdit()
        self.le_epoches.setText('15')

        self.l_batchSize = QLabel('Batch size:')
        self.le_batchSize = QLineEdit()
        self.le_batchSize.setText('16')

        self.l_learnRate = QLabel('Learning rate:')
        self.cb_learnRate = QComboBox()
        self.cb_learnRate.addItem('0.01')
        self.cb_learnRate.addItem('0.05')
        self.cb_learnRate.addItem('0.1')
        self.cb_learnRate.addItem('0.5')
        self.cb_learnRate.addItem('1')
        self.cb_learnRate.addItem('5')
        self.cb_learnRate.addItem('10')
        self.cb_learnRate.setCurrentIndex(4)
        # self.cb_learnRate.setCurrentIndex(1)

        self.l_regL1 = QLabel('L1:')
        self.cb_regL1 = QComboBox()
        self.cb_regL1.addItem('0')
        self.cb_regL1.addItem('0.0001')
        self.cb_regL1.addItem('0.0005')
        self.cb_regL1.addItem('0.001')
        self.cb_regL1.addItem('0.005')
        self.cb_regL1.addItem('0.01')
        self.cb_regL1.addItem('0.05')
        self.cb_regL1.addItem('0.1')

        self.l_regL2 = QLabel('L2:')
        self.cb_regL2 = QComboBox()
        self.cb_regL2.addItem('0')
        self.cb_regL2.addItem('0.0001')
        self.cb_regL2.addItem('0.0005')
        self.cb_regL2.addItem('0.001')
        self.cb_regL2.addItem('0.005')
        self.cb_regL2.addItem('0.01')
        self.cb_regL2.addItem('0.05')
        self.cb_regL2.addItem('0.1')

        self.b_train = QPushButton('Train')
        self.b_eval = QPushButton('Evaluate')

        v_box_1 = QVBoxLayout()
        v_box_1.addWidget(self.l_pygameBackbuff)

        v_box_2 = QVBoxLayout()
        v_box_2.addWidget(self.l_layer1)
        v_box_2.addWidget(self.s_layer1)
        v_box_2.addWidget(self.l_layer2)
        v_box_2.addWidget(self.s_layer2)
        v_box_2.addWidget(self.l_layer3)
        v_box_2.addWidget(self.s_layer3)
        v_box_2.addWidget(self.l_epoches)
        v_box_2.addWidget(self.le_epoches)
        v_box_2.addWidget(self.l_batchSize)
        v_box_2.addWidget(self.le_batchSize)
        v_box_2.addWidget(self.l_learnRate)
        v_box_2.addWidget(self.cb_learnRate)
        v_box_2.addWidget(self.l_regL1)
        v_box_2.addWidget(self.cb_regL1)
        v_box_2.addWidget(self.l_regL2)
        v_box_2.addWidget(self.cb_regL2)
        v_box_2.addWidget(self.b_train)
        v_box_2.addWidget(self.b_eval)

        h_box = QHBoxLayout()
        h_box.addLayout(v_box_1)
        h_box.addLayout(v_box_2)

        self.s_layer1.valueChanged.connect(self.s_layer1_changed)
        self.s_layer2.valueChanged.connect(self.s_layer2_changed)
        self.s_layer3.valueChanged.connect(self.s_layer3_changed)
        self.b_train.clicked.connect(self.b_train_changed)
        self.b_eval.clicked.connect(self.b_eval_changed)

        self.setLayout(h_box)
        self.setWindowTitle(self.title)
        self.show()


    def b_train_changed(self):
        # parse gui prams
        hidden_layer1 = int( self.s_layer1.value() ) 
        hidden_layer2 = int( self.s_layer2.value() )
        hidden_layer3 = int( self.s_layer3.value() )
        hidden_layers = [hidden_layer1, hidden_layer2, hidden_layer3]

        epoches = int( self.le_epoches.text() )

        batch_size = int( self.le_batchSize.text() )
        learning_rate = float( self.cb_learnRate.currentText() )
        reg_L1 = float( self.cb_regL1.currentText() )
        reg_L2 = float( self.cb_regL2.currentText() )
        
        # init NN and PyGame
        steps = 3000
        seed = 15
        np.random.seed(seed)
        random.seed(seed)
        m = generate_map(8, 5, 3, 3)

        # agent = SimpleCarAgent(epochs=15, mini_batch_size=50, eta=0.05)
        agent = SimpleCarAgent(hidden_layers=hidden_layers, epochs=epoches, mini_batch_size=batch_size, eta=learning_rate)
        scw = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2, window=self)
        scw.set_agents([agent])
        scw.run(steps)

    def b_eval_changed(self):
        # get inferance file name
        import os, glob
        work_dir = os.getcwd()

        agent_fils_names = []
        os.chdir(work_dir)
        for file in glob.glob("*.txt"):
            agent_fils_names.append(file)
        
        # init NN and PyGame
        steps = 1000
        seed = 15
        np.random.seed(seed)
        random.seed(seed)
        m = generate_map(8, 5, 3, 3)

        # run game
        agent = SimpleCarAgent.from_file(agent_fils_names[0])
        sw = SimpleCarWorld(1, m, SimplePhysics, SimpleCarAgent, timedelta=0.2, window=self)
        evl = sw.evaluate_agent(agent, steps)
        print(evl)
 

    def s_layer1_changed(self, text):
        new_text = '1st hidden layer: ' + str(text)  
        self.l_layer1.setText(new_text)
    def s_layer2_changed(self, text):
        new_text = '2nd hidden layer: ' + str(text)  
        self.l_layer2.setText(new_text)
    def s_layer3_changed(self, text):
        new_text = '3rd hidden layer: ' + str(text)  
        self.l_layer3.setText(new_text)

    def set_pygame_image(self, surface):
        w = surface.get_width()
        h = surface.get_height()
        data = surface.get_buffer().raw

        self.setGeometry(100, 100, w+300, h)
        self.l0_qimage = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGB32)
        self.l_pygameBackbuff.setPixmap(QtGui.QPixmap.fromImage(self.l0_qimage) )
    def closeEvent(self, event): 
        super().closeEvent(event)
        pygame.quit() 


if __name__ == "__main__": 
    app = QApplication(sys.argv)
    window = Window()   # window = Window(pygame.Surface((600,400)))
    sys.exit(app.exec_())