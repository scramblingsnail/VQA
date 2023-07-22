import torch
import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt


class CurveFitting:
    def __init__(self, batch_size: int, train_step: float, test_step: float, backend: str = 'mindspore'):
        self.batch_size = batch_size
        self.train_step = train_step
        self.test_step = test_step
        self.backend = backend

        self.train_data, self.test_data = self.prepare_data()
        if self.backend == 'mindspore':
            self.train_data_num = self.train_data.shape[0]
            self.test_data_num = self.test_data.shape[0]
        else:
            self.train_data_num = self.train_data.size()[0]
            self.test_data_num = self.test_data.size()[0]

    def show(self):
        plt.cla()
        plt.scatter(self.train_data.numpy()[:, 0], self.train_data.numpy()[:, 1], label='train', c='red')
        plt.plot(self.test_data.numpy()[:, 0], self.test_data.numpy()[:, 1], label='target', c='blue')
        plt.legend()
        plt.savefig(r'fitting_data.png')
        return

    def prepare_data(self):
        def function(x):
            y = 0.2 * np.power(x, 2) + 0.2

            # y = 0.2 * np.sin(x) + 0.2
            return y

        train_x = np.arange(-1, 1 + self.train_step, self.train_step)
        train_y = function(train_x)

        test_x = np.arange(-1, 1 + self.test_step, self.test_step)
        test_y = function(test_x)

        train_data = np.stack((train_x, train_y), axis=1)
        test_data = np.stack((test_x, test_y), axis=1)
        if self.backend == 'mindspore':
            train_data = ms.Tensor(train_data, dtype=ms.float32)
            test_data = ms.Tensor(test_data, dtype=ms.float32)
        else:
            train_data = torch.tensor(train_data, dtype=torch.float)
            test_data = torch.tensor(test_data, dtype=torch.float)
        return train_data, test_data

    def get_train_data(self, shuffle: bool = True):
        data_indices = list(range(self.train_data_num))
        if shuffle:
            np.random.shuffle(data_indices)
        start_idx = 0
        while start_idx + self.batch_size < self.train_data_num:
            end_idx = start_idx + self.batch_size
            if end_idx > self.train_data_num:
                end_idx = self.train_data_num
            batch_data = self.train_data[data_indices[start_idx: end_idx]]
            yield batch_data[:, :1], batch_data[:, 1]
            start_idx = end_idx

    def get_test_data(self):
        data_indices = list(range(self.test_data_num))
        start_idx = 0
        while start_idx < self.test_data_num:
            end_idx = start_idx + self.batch_size
            if end_idx > self.test_data_num:
                end_idx = self.test_data_num
            batch_data = self.test_data[data_indices[start_idx: end_idx]]
            yield batch_data[:, :1], batch_data[:, 1]
            start_idx = end_idx


class Classification:
    def __init__(self, train_data_num: int, test_data_num: int, batch_size: int, backend: str = 'mindspore'):
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num
        self.batch_size = batch_size
        self.backend = backend
        self.train_data, self.test_data = self.prepare_data()

    def show(self):
        def plot(data_tensor, name):
            data_xy, data_label = data_tensor[:, :2], data_tensor[:, 2]
            class_1 = data_label == 1
            class_0 = data_label == 0
            plt.cla()
            plt.scatter(data_xy[class_1][:, 0], data_xy[class_1][:, 1], c='red')
            plt.scatter(data_xy[class_0][:, 0], data_xy[class_0][:, 1], c='blue')
            plt.savefig(r'{}.png'.format(name))

        plot(self.train_data, 'classify_train')
        plot(self.test_data, 'classify_test')
        return

    def prepare_data(self):

        def valid_data(dx, dy, area_idx):
            def coo_round(x, y):
                if x ** 2 + y ** 2 <= radius ** 2:
                    return True
                else:
                    return False

            if area_idx == 0:
                center_x = 0.25
                center_y = 0.25
            elif area_idx == 1:
                center_x = 0.25
                center_y = 0.75
            elif area_idx == 2:
                center_x = 0.75
                center_y = 0.75
            elif area_idx == 3:
                center_x = 0.75
                center_y = 0.25
            else:
                raise ValueError
            if coo_round(dx, dy):
                return center_x + dx, center_y + dy
            else:
                return None

        def collect_data(data_num, area_idx, label):
            data_list = []
            for i in range(data_num):
                valid = None
                while valid is None:
                    dx = np.random.uniform(-radius, radius)
                    dy = np.random.uniform(-radius, radius)
                    # label = data_class(x, y)
                    valid = valid_data(dx, dy, area_idx=area_idx)
                data_list.append([valid[0], valid[1], label])
            return data_list

        radius = 0.1
        train_data = []
        test_data = []
        for idx in range(4):
            train_area_num = self.train_data_num // 4 + (self.train_data_num - (self.train_data_num // 4) * 4) * (idx == 3)
            test_area_num = self.test_data_num // 4 + (self.test_data_num - (self.test_data_num // 4) * 4) * (
                        idx == 3)
            train_data += collect_data(train_area_num, idx, idx % 2)
            test_data += collect_data(test_area_num, idx, idx % 2)

        if self.backend == 'mindspore':
            train_data = ms.Tensor(train_data, dtype=ms.float32)
            test_data = ms.Tensor(test_data, dtype=ms.float32)
        else:
            train_data = torch.tensor(train_data, dtype=torch.float)
            test_data = torch.tensor(test_data, dtype=torch.float)
        return train_data, test_data

    def get_train_data(self, shuffle: bool = False):
        data_indices = list(range(self.train_data_num))
        if shuffle:
            np.random.shuffle(data_indices)
        start_idx = 0
        while start_idx < self.train_data_num:
            end_idx = start_idx + self.batch_size
            if end_idx > self.train_data_num:
                end_idx = self.train_data_num
            batch_data = self.train_data[data_indices[start_idx: end_idx]]
            if self.backend == 'mindspore':
                yield batch_data[:, :2], batch_data[:, 2].int()
            else:
                yield batch_data[:, :2], batch_data[:, 2].long()
            start_idx = end_idx

    def get_test_data(self):
        data_indices = list(range(self.test_data_num))
        start_idx = 0
        while start_idx < self.test_data_num:
            end_idx = start_idx + self.batch_size
            if end_idx > self.test_data_num:
                end_idx = self.test_data_num
            batch_data = self.test_data[data_indices[start_idx: end_idx]]
            if self.backend == 'mindspore':
                yield batch_data[:, :2], batch_data[:, 2].int()
            else:
                yield batch_data[:, :2], batch_data[:, 2].long()
            start_idx = end_idx
