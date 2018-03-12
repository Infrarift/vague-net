import math
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss, _assert_no_grad

from data_creator import DataCreator
from vague_net import VagueNet, VagueLoss
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Facade:

    K_NUM_DATA = 1000

    def __init__(self):
        self.creator = DataCreator()
        self.dataset_a = self.creator.create_data_set(self.creator.data_function_a, 0, self.K_NUM_DATA)
        self.dataset_b = self.creator.create_data_set(self.creator.data_function_b, 1, self.K_NUM_DATA)
        self.net = VagueNet()

    def run(self):

        # Loss Function
        criterion = VagueLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)

        train_set, test_set = self.creator.get_train_test(self.dataset_a, self.dataset_b)

        for epoch in range(100):

            batch_size = 10
            batch_steps = int(math.floor(len(train_set) // batch_size))

            for i in range(batch_steps):
                train_batch = train_set[i*batch_size:(i+2)*batch_size]

                features = train_batch[:, :2]
                labels = train_batch[:, 2]

                features, labels = Variable(torch.from_numpy(features)), Variable(torch.from_numpy(labels))
                optimizer.zero_grad()

                outputs, w = self.net(features)
                loss = criterion(outputs, labels, w)
                loss.backward()
                optimizer.step()
                print(loss)
                print("W: {}".format(w))

        correct_x = []
        correct_y = []
        wrong_x = []
        wrong_y = []

        correct_count = 0

        for i in range(len(test_set)):
            test_data = test_set[i]
            test_x = test_data[:2]
            test_label = test_data[2]
            x = Variable(torch.from_numpy(test_x))
            result, w = self.net(x)
            result = round(result.data[0])
            # print(result)

            is_correct = result == int(test_label)
            # print(is_correct)
            if is_correct:
                correct_count += 1
                correct_x.append(test_x[0])
                correct_y.append(test_x[1])
            else:
                wrong_x.append(test_x[0])
                wrong_y.append(test_x[1])
                pass

        print("Correct: {:.0f}%".format(100 * correct_count/len(test_set)))
        plt.plot(correct_x, correct_y, "gx")
        plt.plot(wrong_x, wrong_y, "rx")
        plt.show()

    def plot_original_sets(self):
        self.plot_dataset(self.dataset_a, "bx")
        self.plot_dataset(self.dataset_b, "rx")
        plt.show()

    def plot_dataset(self, dataset, style="bx"):
        px, py = self.creator.get_dataset_x_y(dataset)
        plt.plot(px, py, style)

