import math
import matplotlib.pyplot as plt
from torch.nn.modules.loss import _Loss, _assert_no_grad
import numpy as np
from data_creator import DataCreator
from vague_net import VagueNet, VagueLoss
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Facade:

    K_NUM_DATA = 3000

    def __init__(self):
        self.creator = DataCreator()
        self.dataset_a = self.creator.create_data_set(self.creator.data_function_a, 0, self.K_NUM_DATA)
        self.dataset_b = self.creator.create_data_set(self.creator.data_function_b, 1, self.K_NUM_DATA)
        self.plot_original_sets()
        self.net = VagueNet()

    def run(self):

        # Loss Functions
        p_criterion = nn.MSELoss()
        c_criterion = VagueLoss()
        e_loss = nn.CrossEntropyLoss()

        # Optimizers
        p_optimizer = optim.SGD(self.net.params[0], lr=0.002, momentum=0.9)
        c_optimizer = optim.SGD(self.net.params[1], lr=0.002, momentum=0.9)

        # Declare data-sets.
        train_set, test_set = self.creator.get_train_test(self.dataset_a, self.dataset_b)

        # Train the raw network to recognize the classes.
        self.train_first_net(epochs=100, train_set=train_set, optimizer=p_optimizer, criterion=e_loss)
        self.evaluate(test_set, test_confidence=False, mask=False, title="1st Net Evaluation")

        # Create a sub-sample for secondary training.
        sub_samples = self.create_sub_sample(train_set)

        # Train Second Net.
        self.train_second_net(epochs=100, train_set=sub_samples, optimizer=c_optimizer, criterion=e_loss)
        self.evaluate(sub_samples, test_confidence=True, mask=False, title="2nd Net Evaluation")

        # Evaluate results.
        self.evaluate(test_set, test_confidence=False, mask=True, title="Final Evaluation")

    def train_first_net(self, epochs, train_set, optimizer, criterion):
        for epoch in range(epochs):

            batch_size = 50
            batch_steps = int(math.floor(len(train_set) // batch_size))

            for i in range(batch_steps):
                train_batch = train_set[i*batch_size:(i+1)*batch_size]

                features = train_batch[:, :2]
                labels = train_batch[:, 2]

                one_hot_labels = self.create_one_hot(labels, 2)
                features, labels = Variable(torch.from_numpy(features)), Variable(torch.from_numpy(one_hot_labels))
                optimizer.zero_grad()

                p, c = self.net(features)
                # print("Output: {}".format(p))
                # print("Target: {}".format(torch.max(labels, 1)[1]))
                loss = criterion(p, torch.max(labels, 1)[1])
                loss.backward()
                optimizer.step()
                print("Loss: {}".format(loss.data[0]))

    def create_one_hot(self, labels, num_classes):
        num_samples = len(labels)
        labels_array = np.zeros((num_samples, num_classes), np.long)
        for i in range(num_samples):
            label = int(labels[i])
            labels_array[i, label] = 1
        return labels_array

    def train_second_net(self, epochs, train_set, optimizer, criterion):

        for epoch in range(epochs):

            batch_size = 10
            batch_steps = int(math.floor(len(train_set) // batch_size))

            for i in range(batch_steps):
                train_batch = train_set[i*batch_size:(i+1)*batch_size]

                features = train_batch[:, :2]
                labels = train_batch[:, 2]

                one_hot_labels = self.create_one_hot(labels, 2)
                label_tensor = torch.from_numpy(one_hot_labels)
                features, labels = Variable(torch.from_numpy(features)), Variable(label_tensor)
                optimizer.zero_grad()

                p, c = self.net(features)
                # p.data = p.data.view(batch_size)
                # cmp = []
                # q = Variable(torch.Tensor(cmp))
                # m = Variable(torch.Tensor(mask_list))
                # a = Variable(torch.Tensor(a_mask_list))

                loss = criterion(c, torch.max(labels, 1)[1])
                loss.backward()
                optimizer.step()
                print("Loss 2: {}".format(loss.data[0]))

        print("Net 2 Training Complete")



    def create_sub_sample(self, train_set):

        positive_samples = []
        negative_samples = []

        # Use the net to predict the training data.
        for i in range(len(train_set)):
            train_data = train_set[i]
            features = train_data[:2]
            label = train_data[2]
            x = Variable(torch.from_numpy(features))
            p, c = self.net(x)
            result = torch.max(p.data, 0)[1][0]
            is_correct = result == int(label)
            if is_correct:
                positive_samples.append(features)
            else:
                negative_samples.append(features)

        # Normalize the size of the two sets.
        size = min(len(positive_samples), len(negative_samples))
        print("Pos: {} | Neg: {} | Resample: {}".format(len(positive_samples), len(negative_samples), size))

        final_samples = np.zeros((size * 2, 3), np.float32)
        for i in range(size):
            j = i * 2
            final_samples[j][0] = positive_samples[i][0]
            final_samples[j][1] = positive_samples[i][1]
            final_samples[j][2] = 1

            final_samples[j + 1][0] = negative_samples[i][0]
            final_samples[j + 1][1] = negative_samples[i][1]
            final_samples[j + 1][2] = 0

        # Plot the output samples.
        pos_x = [sample[0] for sample in positive_samples[:size]]
        pos_y = [sample[1] for sample in positive_samples[:size]]

        neg_x = [sample[0] for sample in negative_samples[:size]]
        neg_y = [sample[1] for sample in negative_samples[:size]]

        plt.plot(pos_x, pos_y, "g.")
        plt.plot(neg_x, neg_y, "rx")
        plt.title("Sub Sample")
        plt.show()

        return final_samples

    def evaluate(self, test_set, test_confidence=False, mask=True, title="Evaluation"):
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
            p_out, c_out = self.net(x)

            p = torch.max(p_out.data, 0)[1][0]
            c = torch.max(c_out.data, 0)[1][0]  # round(c_out.data[0])
            # c_value = c[0]

            if mask and c < 0.5:
                skipped_x.append(test_x[0])
                skipped_y.append(test_x[1])
                continue

            if test_confidence:
                f = c
            else:
                f = p

            running_count += 1
            is_correct = test_label == int(f)

            if is_correct:
                correct_count += 1
                correct_x.append(test_x[0])
                correct_y.append(test_x[1])
            else:
                wrong_x.append(test_x[0])
                wrong_y.append(test_x[1])
                pass

        full_count = len(test_set)
        correct_final = "{:.2f}%".format(100 * correct_count / running_count)
        discard_final = "{:.2f}%".format(100 * (full_count - running_count) / full_count)
        raw_correct_final = "{:.2f}%".format(100 * correct_count / full_count)

        print("Correct (Filtered): {}".format(correct_final))
        print("Correct (Total): {}".format(raw_correct_final))
        print("Discarded: {}".format(discard_final))

        plt.plot(correct_x, correct_y, "gx")
        plt.plot(wrong_x, wrong_y, "rx")
        plt.plot(skipped_x, skipped_y, "bx", color=(0.8, 0.8, 0.8))
        plt.title(title)
        plt.show()

    def plot_original_sets(self):
        self.plot_dataset(self.dataset_a, "bx")
        self.plot_dataset(self.dataset_b, "rx")
        plt.show()

    def plot_dataset(self, dataset, style="bx"):
        px, py = self.creator.get_dataset_x_y(dataset)
        plt.plot(px, py, style)

