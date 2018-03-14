import random
import numpy as np


class DataCreator:
    def __init__(self):
        pass

    @staticmethod
    def get_train_test(set_a, set_b):
        complete_set = np.concatenate((set_a, set_b), axis=0)
        np.random.shuffle(complete_set)
        split_index = int(len(complete_set) * 0.7)

        train_set = complete_set[:split_index]
        test_set = complete_set[split_index:]
        return train_set, test_set

    @staticmethod
    def data_function_a(x):
        # chance = random.random()
        # if chance > 0.5:
        #     start = 0
        # else:
        #     start = 6
        start = 0
        f_range = 6
        return start + random.random() * f_range

    @staticmethod
    def data_function_b(x):
        start = 3
        f_range = 6
        return start + random.random() * f_range

    @staticmethod
    def create_data_set(func, label, num_samples=100):
        data_set = np.zeros((num_samples, 3), np.float32)
        for i in range(num_samples):
            x = 1 + random.random() * 9
            y = func(x)
            data_set[i][0] = x
            data_set[i][1] = y
            data_set[i][2] = label
        return data_set

    @staticmethod
    def get_dataset_x_y(data_set):
        x = data_set[:, 0]
        y = data_set[:, 1]
        return x, y
