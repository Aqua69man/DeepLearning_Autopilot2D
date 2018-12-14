import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network, cost_function
from learning_algorithms.regularized_network import RegularizedNetwork

import matplotlib.pyplot as plt


class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    def __init__(self, hidden_layers=None, epochs=15, mini_batch_size=50, eta=0.05, l1=0, l2=0, history_data=int(50000)):
        """
        Создаёт машинку
        :param history_data: количество хранимых нами данных о результатах предыдущих шагов
        """
        # self.train_sensor_data_history = []
        # self.train_chosen_actions_history = []
        # self.train_reward_history = []
        # # --
        # self.test_sensor_data_history = []
        # self.test_chosen_actions_history = []
        # self.test_reward_history = []
        # # --
        self.cost_train = []
        self.cost_test = []

        self.evaluate_mode = False  # этот агент учится или экзаменутеся? если учится, то False
        self._rays = 5 # выберите число лучей ладара; например, 5
        

        self.layers = []
        self.layers.append(self._rays + 4) # INPUT layer: rays + (velosity + headingAngle + acceleration + steeering) 
        if hidden_layers:                 # HIDDEN layer: chooose, how many and in which ratio you need, for example, (self._rays + 4) * 2 or just an digit
            preproc_hidden_layers = [x for x in hidden_layers if x > 0]
            self.layers += preproc_hidden_layers 
        self.layers.append(1)             # OUTPUT layer

        self.epochs=epochs 
        self.train_every=mini_batch_size
        self.eta=eta
        self.l1 = l1
        self.l2 = l2

        # here +2 is for 2 inputs from elements of Action that we are trying to predict. 
        self.neural_net = RegularizedNetwork(sizes=self.layers, output_log=False,
                                  output_function=lambda x: x, output_derivative=lambda x: 1,
                                  l1=self.l1, l2=self.l2)
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0


    @classmethod
    def from_weights(cls, layers, epochs, train_every, eta, l1, l2, weights, biases):
        """
        Создание агента по параметрам его нейронной сети. Разбираться не обязательно.
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        agent.layers = layers
        agent.epochs = epochs 
        agent.train_every = train_every
        agent.eta = eta
        agent.l1 = l1
        agent.l2 = l2

        nn = RegularizedNetwork(sizes=layers, output_log=False,
                     output_function=lambda x: x, output_derivative=lambda x: 1,
                     l1=agent.l1, l2=agent.l2)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # это важный импорт, без него не пройдёт нормально eval
        layers, epochs, train_every, eta, l1, l2, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, epochs, train_every, eta, l1, l2, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_params(self):
        params = self.neural_net.sizes, self.epochs, self.train_every, self.eta, self.l1, self.l2,  self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_params()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):
        # хотим предсказать награду за все действия, доступные из текущего состояния
        rewards_to_controls_map = {}
        # дискретизируем множество значений, так как все возможные мы точно предсказать не сможем
        for steering in np.linspace(-1, 1, 3):  # выбирать можно и другую частоту дискретизации, но
            for acceleration in np.linspace(-0.75, 0.75, 3):  # в наших тестах будет именно такая
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map[predicted_reward] = action

        # ищем действие, которое обещает максимальную награду
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Добавим случайности, дух авантюризма. Иногда выбираем совершенно
        # рандомное действие
        if (not self.evaluate_mode) and (random.random() < 0.05):
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]
        # следующие строки помогут вам понять, что предсказывает наша сеть
        #     print("Chosen random action w/reward: {}".format(highest_reward))
        # else:
        #     print("Chosen action w/reward: {}".format(highest_reward))

        # запомним всё, что только можно: мы хотим учиться на своих ошибках
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  # мы пока не знаем, какая будет награда, это
        # откроется при вызове метода receive_feedback внешним миром

        return best_action

    def receive_feedback(self, reward, reward_depth=7):
        """
        Получить реакцию на последнее решение, принятое сетью, и проанализировать его
        :param reward: оценка внешним миром наших действий
        :param train_every: сколько нужно собрать наблюдений, прежде чем запустить обучение на несколько эпох
        :param reward_depth: на какую глубину по времени распространяется полученная награда
        """
        # считаем время жизни сети; помогает отмерять интервалы обучения
        self.step += 1

        # начиная с полной полученной истинной награды,
        # размажем её по предыдущим наблюдениям
        # чем дальше каждый раз домножая её на 1/2
        # (если мы врезались в стену - разумно наказывать не только последнее
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        # Если у нас накопилось хоть чуть-чуть данных, давайте потренируем нейросеть
        # прежде чем собирать новые данные
        # (проверьте, что вы в принципе храните достаточно данных (параметр `history_data` в `__init__`),
        # чтобы условие len(self.reward_history) >= train_every выполнялось
        if not self.evaluate_mode and (len(self.reward_history) >= self.train_every) and not (self.step % self.train_every):
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y_train = self.reward_history
            
            # self.neural_net.SGD(training_data=train_data, epochs=15, mini_batch_size=train_every, eta=0.05)
            train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]
            self.neural_net.SGD(training_data=train_data, epochs=self.epochs, mini_batch_size=self.train_every, eta=self.eta)
            print('Train costfunc: ', cost_function(self.neural_net, train_data, onehot=True))

            ## ---------------------------------------- Test and train 
            # np_sensor_data_history = np.array(self.sensor_data_history)
            # np_chosen_actions_history = np.array(self.chosen_actions_history)
            # np_reward_history = np.array(self.reward_history)

            # lastBatch_sensor_data_history       = np_sensor_data_history[-self.train_every:]
            # lastBatch_chosen_actions_history    = np_chosen_actions_history[-self.train_every:]
            # lastBatch_reward_history            = np_reward_history[-self.train_every:]
            

            # test_index = np.random.choice([True, False], self.train_every, replace=True, p=[0.25, 0.75])
            # # --
            # test_sensor_data_history = lastBatch_sensor_data_history[test_index]
            # test_chosen_actions_history = lastBatch_chosen_actions_history[test_index]
            # test_reward_history = lastBatch_reward_history[test_index]
            # # --
            # train_sensor_data_history = lastBatch_sensor_data_history[np.logical_not(test_index)]
            # train_chosen_actions_history = lastBatch_chosen_actions_history[np.logical_not(test_index)]
            # train_reward_history = lastBatch_reward_history[np.logical_not(test_index)]
            


            # [self.test_sensor_data_history.append(x)    for x in test_sensor_data_history]
            # [self.test_chosen_actions_history.append(x) for x in test_chosen_actions_history]
            # [self.test_reward_history.append(x)         for x in test_reward_history]

            # [self.train_sensor_data_history.append(x)       for x in train_sensor_data_history]
            # [self.train_chosen_actions_history.append(x)    for x in train_chosen_actions_history]
            # [self.train_reward_history.append(x)            for x in train_reward_history]


            # x_test = np.concatenate([self.test_sensor_data_history, self.test_chosen_actions_history], axis=1)
            # y_test = self.test_reward_history
            # # --
            # x_train = np.concatenate([self.train_sensor_data_history, self.train_chosen_actions_history], axis=1)
            # y_train = self.train_reward_history


            # train = [(x[:, np.newaxis], y) for x, y in zip(x_test, y_test)]
            # test = [(x[:, np.newaxis], y) for x, y in zip(x_train, y_train)]


            # self.neural_net.SGD(training_data=train, epochs=self.epochs, mini_batch_size=self.train_every, eta=self.eta)
            # # print('Train costfunc: ', cost_function(self.neural_net, train, onehot=True))

            ## ---------------------------------------- Plot graph
            # self.cost_train.append(cost_function(self.neural_net, train, onehot=True))
            # self.cost_test.append(cost_function(self.neural_net, test, onehot=True))
            
            # plt.close()
            # fig = plt.figure(figsize=(15,5))
            # fig.add_subplot(1,1,1)
            # plt.plot(self.cost_train, label="Training error", color="orange")
            # plt.plot(self.cost_test, label="Test error", color="blue")
            # plt.title("Learning curve")
            # plt.ylabel("Cost function")
            # plt.xlabel("Epoch number")
            # plt.legend()
            # plt.pause(0.1)

            # # self.window.graphCanvas.plot(self.cost_train, self.cost_test)