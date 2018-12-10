from network import *

class RegularizedNetwork(Network):
    def __init__(self, sizes, output_log=True, output_function=sigmoid, output_derivative=sigmoid_prime, l1=0, l2=0):
        super().__init__(sizes, output_log, output_function, output_derivative)
        self.l1 = l1
        self.l2 = l2
        # ======================================================
        self.firstTime = True # delete me
        self.old_weights = []
        self.new_weights = []
        self.j1_weights = []
        self.l1_weights = []
        self.l2_weights = []
        # ======================================================
        
        
    def update_mini_batch(self, mini_batch, eta):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch. Учесть штрафы за L1 и L2.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate).
        """
            
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        eps = eta / len(mini_batch)
        
        # --------------------------- Start: Log weights
        # self.old_weights.append(self.weights)

        # j1_weights = [np.zeros(w.shape) for w in self.weights]
        # l1_weights = [np.zeros(w.shape) for w in self.weights]
        # l2_weights = [np.zeros(w.shape) for w in self.weights]
        # #----
        # j1_weights = [eps * nw for w, nw in zip(self.weights, nabla_w)]
        # l1_weights = [self.l1 * np.sign(w) for w in self.weights]
        # l2_weights = [self.l2 * w for w in self.weights]
        # self.j1_weights.append(j1_weights)
        # self.l1_weights.append(l1_weights)
        # self.l2_weights.append(l2_weights)
        # --------------------------- End
        
        self.weights = [w - eps * nw - self.l1 * np.sign(w) - self.l2 * w for w, nw in zip(self.weights, nabla_w)]
        #self.weights = [w - eta * (nw/len(mini_batch) + self.l1 * np.sign(w) + self.l2 * w) for w, nw in zip(self.weights, nabla_w)]
        
        # --------------------------- Start: Log weights ------------------------
        # self.new_weights.append(self.weights)
        # --------------------------- End ---------------------------------------
        
        self.biases  = [b - eps * nb for b, nb in zip(self.biases,  nabla_b)]