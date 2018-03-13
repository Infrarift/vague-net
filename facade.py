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
        criterion = nn.MSELoss()
        c_criterion = VagueLoss()
        p_optimizer = optim.SGD(self.net.n1_params, lr=0.001, momentum=0.9)
        c_optimizer = optim.SGD(self.net.n2_params, lr=0.001, momentum=0.1)
        train_set, test_set = self.creator.get_train_test(self.dataset_a, self.dataset_b)

        for epoch in range(50):

            batch_size = 10
            batch_steps = int(math.floor(len(train_set) // batch_size))

            for i in range(batch_steps):
                train_batch = train_set[i*batch_size:(i+1)*batch_size]

                features = train_batch[:, :2]
                labels = train_batch[:, 2]

                features, labels = Variable(torch.from_numpy(features)), Variable(torch.from_numpy(labels))
                p_optimizer.zero_grad()

                p, c = self.net(features)
                loss = criterion(p, labels)
                loss.backward()
                p_optimizer.step()
                print("Loss: {}".format(loss.data[0]))

        for epoch in range(50):

            batch_size = 10
            batch_steps = int(math.floor(len(train_set) // batch_size))

            for i in range(batch_steps):
                train_batch = train_set[i*batch_size:(i+1)*batch_size]

                features = train_batch[:, :2]
                labels = train_batch[:, 2]

                label_tensor = torch.from_numpy(labels)
                features, labels = Variable(torch.from_numpy(features)), Variable(label_tensor)
                c_optimizer.zero_grad()

                p, c = self.net(features)
                p.data = p.data.view(batch_size)
                cmp = []
                mask_list = []
                a_mask_list = []
                penalty_weight_wrong = 3
                penalty_weight_correct = 1

                for i in range(len(p.data)):
                    p_i = p.data[i]
                    l_i = labels.data[i]
                    p_f = round(p_i)
                    result = 1 if p_f == l_i else 0
                    mask_weight = penalty_weight_correct if p_f == l_i else penalty_weight_wrong
                    add_mask_weight = 1 if p_f == l_i else 0

                    cmp.append(result)
                    mask_list.append(mask_weight)
                    a_mask_list.append(mask_weight)

                q = Variable(torch.Tensor(cmp))
                m = Variable(torch.Tensor(mask_list))
                a = Variable(torch.Tensor(a_mask_list))

                # p.data = p.data.view(10)
                # q = (p - labels) ** 2
                # q = q.detach()
                loss = c_criterion(c, q, m, a)
                loss.backward()
                c_optimizer.step()
                print("Loss 2: {}".format(loss.data[0]))

        print("Net Training Complete")

        correct_x = []
        correct_y = []
        wrong_x = []
        wrong_y = []
        skipped_x = []
        skipped_y = []

        correct_count = 1
        running_count = 1

        for i in range(len(test_set)):
            test_data = test_set[i]
            test_x = test_data[:2]
            test_label = test_data[2]
            x = Variable(torch.from_numpy(test_x))
            result, w = self.net(x)
            print("Result: {} | {}".format(result.data[0], w.data[0]))

            if w.data[0] < 0.5:
                skipped_x.append(test_x[0])
                skipped_y.append(test_x[1])
                continue

            result = round(result.data[0])
            # print(result)

            running_count += 1
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

        print("Correct: {:.0f}%".format(100 * correct_count/running_count))
        plt.plot(correct_x, correct_y, "gx")
        plt.plot(wrong_x, wrong_y, "rx")
        plt.plot(skipped_x, skipped_y, "bx", color=(0.8, 0.8, 0.8))
        plt.show()

    def plot_original_sets(self):
        self.plot_dataset(self.dataset_a, "bx")
        self.plot_dataset(self.dataset_b, "rx")
        plt.show()

    def plot_dataset(self, dataset, style="bx"):
        px, py = self.creator.get_dataset_x_y(dataset)
        plt.plot(px, py, style)

