from network import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = np.loadtxt("data.csv", delimiter=",")

means = data.mean(axis=0)
means[-1] = 0  # правильные ответы мы нормализовывать не будем: это качественные переменные
stds = data.std(axis=0)
stds[-1] = 1
data = (data - means) / stds

test_index = np.random.choice([True, False], len(data), replace=True, p=[0.25, 0.75])
test  = data[test_index]
train = data[np.logical_not(test_index)]

train = [(d[:3][:, np.newaxis], np.eye(3, 1, k=-int(d[-1]))) for d in train]  
test =  [(d[:3][:, np.newaxis], d[-1]) for d in test]

input_count  = 3  # 3 нейрона входного слоя
hidden_count = 6  # 6 нейронов внутреннего слоя 
output_count = 3  # 3 нейрона выходного слоя, по индикатору для каждого из классов "попал", "недолёт" и "перелёт"



def cost_function(network, test_data, onehot=True, showLog=False, weights=None, l1=None, l2=None):
    c = 0
    for example, y in test_data:
        if not onehot:
            y = np.eye(3, 1, k=-int(y))
        yhat = network.feedforward(example)
        c += np.sum((y - yhat)**2)
    c = c / (2 * len(test_data)) # не забудем поделить на 2, т.к. у нас выйдет двойка после дифференцирования
    
    # -------- Farid ----------
    # ARGs: weights=None, l1=None, l2=None, were also added by Farid to test    
    if weights is not None:
        if showLog is True:
            print("---------------------------------")
            print("J1: ", c)
        
        wAbsSum = 0
        for arr in weights:
            for row in arr:
                for cell in row:
                    wAbsSum += abs(cell)  
        l1_reg = l1 * wAbsSum
        c += l1_reg
        if showLog is True:
            print("L1_sum: ", wAbsSum)
            print("L1: ", l1_reg)
        
        w2sum = 0
        for arr in weights:
            for row in arr:
                for cell in row:
                    w2sum += abs(cell) ** 2
        l2_reg = (l2 / 2) * w2sum
        c += l2_reg
        if showLog is True:        
            print("L2_sum: ", w2sum)
            print("L2: ", l2_reg)
        
        if showLog is True:
            print("_J: ", c)
    # -------- ~ Farid ----------
    
    return c

def learning_curve_by_network_structure(layer1, layer2, layer3, batch_size, learning_rate):
    layers = [x for x in [input_count, layer1, layer2, layer3, output_count] if x > 0]
    nn = Network(layers)
    learning_rate=float(learning_rate)
    
    CER = []
    cost_train = []
    cost_test  = []
    for _ in range(150):
        nn.SGD(training_data=train, epochs=1, mini_batch_size=batch_size, eta=learning_rate)
        CER.append(1 - nn.evaluate(test) / len(test))
        cost_test.append(cost_function(nn, test, onehot=False))
        cost_train.append(cost_function(nn, train, onehot=True))
    
    for i, key in enumerate(CER):
        print("Эпоха {0}: ошибка = {1}".format(i, key))

    fig = plt.figure(figsize=(15,5))
    fig.add_subplot(1,2,1)
    plt.ylim(0, 1)
    plt.plot(CER)
    plt.title("Classification error rate")
    plt.ylabel("Percent of incorrectly identified observations")
    plt.xlabel("Epoch number")
    
    fig.add_subplot(1,2,2)
    plt.plot(cost_train, label="Training error", color="orange")
    plt.plot(cost_test, label="Test error", color="blue")
    plt.title("Learning curve")
    plt.ylabel("Cost function")
    plt.xlabel("Epoch number")
    plt.legend()
    plt.show()

    
# ------------------------------------ Main 2 ---------------------------------------------------
# learning_curve_by_network_structure(layer1=6, layer2=0, layer3=0, batch_size=4, learning_rate=1)
# -----------------------------------------------------------------------------------------------