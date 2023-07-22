import mindspore as ms
from mindspore import nn
from mindspore.amp import all_finite
import numpy as np
from src import Classification, CurveFitting, Config
from src.qnn_mindspore import QNet
import matplotlib.pyplot as plt


class QNN(nn.Cell):
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def construct(self, batch_input):
        batch_output = []
        for idx in range(batch_input.shape[0]):
            expect_0, expect_1 = self.q_net(batch_input[idx])
            batch_output.append(expect_0)
        batch_output = ms.ops.stack(batch_output, axis=0)
        return batch_output


class BaseForward(nn.Cell):
    def __init__(self, qnn: QNN, loss_func):
        super().__init__()
        self.qnn = qnn
        self.loss_func = loss_func

    def construct(self, batch_input, batch_target):
        batch_output = self.qnn(batch_input)
        if isinstance(self.loss_func, nn.MSELoss):
            batch_output = batch_output.squeeze(1)
        loss = self.loss_func(batch_output, batch_target)
        return loss


class ClassificationForward(BaseForward):
    def __init__(self, q_net):
        qnn = QNN(q_net)
        super().__init__(qnn=qnn, loss_func=nn.CrossEntropyLoss())


class FittingForward(BaseForward):
    def __init__(self, q_net):
        qnn = QNN(q_net)
        super().__init__(qnn=qnn, loss_func=nn.MSELoss())


class BaseTrainer:
    def __init__(self, forward_net, data_loader, optimizer, epoch_num, evaluate_interval):
        self.epoch_num = epoch_num
        self.evaluate_interval = evaluate_interval
        self.forward_net = forward_net
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.value_and_grad = ms.value_and_grad(self.forward_net, None, weights=self.optimizer.parameters)
        self.save_test_result_interval = None

    def single_train(self, batch_input, batch_target):
        train_loss, grads = self.value_and_grad(batch_input, batch_target)
        if all_finite(grads):
            self.optimizer(grads)
        return train_loss

    def evaluate(self, train_data: bool = False):
        if train_data:
            evaluate_data = self.data_loader.get_train_data(shuffle=False)
        else:
            evaluate_data = self.data_loader.get_test_data()

        output = []
        target = []
        for batch_input, batch_target in evaluate_data:
            batch_output = self.forward_net.qnn(batch_input)
            output.append(batch_output)
            target.append(batch_target)
        output = ms.ops.cat(output, axis=0)
        target = ms.ops.cat(target, axis=0)
        return output, target

    def score(self, output, target):
        score = ms.ops.ones(1)
        return score

    def set_params(self):
        param_dict = {}
        with open('param.txt', 'r') as p_f:
            lines = p_f.readlines()
        for each_line in lines:
            name = each_line.split()[0]
            param = each_line.split()[1]
            param_dict[name] = ms.Tensor([float(param)], dtype=ms.float32)

        param_head = 'qnn.q_net.'
        for each_name in param_dict.keys():
            param_name = param_head + each_name
            self.forward_net.qnn.q_net.parameters_dict()[param_name].set_data(param_dict[each_name])
        return

    def train(self, early_stop=False, score_threshold=0):
        self.forward_net.set_train(True)
        loss_record = []
        train_score_record = []
        test_score_record = []
        intermediate_data = []
        for epoch in range(self.epoch_num):
            batch_loss = []
            # train
            for batch_input, batch_target in self.data_loader.get_train_data(shuffle=True):
                train_loss = self.single_train(batch_input, batch_target)
                batch_loss.append(train_loss.numpy())

            print('epoch {} train loss: {}'.format(epoch, np.mean(batch_loss)))
            loss_record.append(np.mean(batch_loss))
            # evaluate
            self.forward_net.set_train(False)

            if epoch % self.evaluate_interval == 0:
                train_output, train_target = self.evaluate(train_data=True)
                train_score = self.score(train_output, train_target)
                train_score_record.append(train_score)
                print('train score: ', train_score)
                if early_stop:
                    if train_score >= score_threshold:
                        break

            # save test intermediate result
            if self.save_test_result_interval is not None:
                if epoch % self.save_test_result_interval == 0:
                    print('testing ...')
                    test_output, test_target = self.evaluate(train_data=False)
                    test_score = self.score(test_output, test_target)
                    test_score_record.append(test_score)
                    print('test score: ', test_score)
                    intermediate_data.append(test_output.numpy())

            self.forward_net.set_train(True)
        return loss_record, train_score_record, test_score_record, intermediate_data


class ClassificationTrainer(BaseTrainer):
    def __init__(self, configs: Config):
        epoch_num = configs.classification_epoch_num
        evaluate_interval = configs.classification_evaluate_interval
        test_interval = configs.classification_save_interval

        r""" circuit definition """
        encoder_circuit, ansatze_circuit = configs.encoder_circuit, configs.ansatze_circuit

        r""" dataset """
        train_data_num = configs.classification_train_data_num
        test_data_num = configs.classification_test_data_num
        mini_batch_num = configs.classification_mini_batch_num
        data_loader = Classification(batch_size=mini_batch_num, train_data_num=train_data_num,
                                     test_data_num=test_data_num,
                                     backend='mindspore')
        data_loader.show()

        r""" variational quantum network """
        observe_bit = configs.classification_observe_bit
        observe_gate = configs.classification_observe_gate
        init_range = configs.init_range
        q_net = QNet(encoder_circuit=encoder_circuit, circuit=ansatze_circuit,
                     observe_gate=observe_gate, observe_bit=observe_bit,
                     param_init_range=init_range, input_scaling=configs.classification_input_scaling)
        q_net.show_circuit()

        r""" optimizer """
        lr = configs.classification_lr
        weight_decay = configs.classification_weight_decay
        momentum = configs.classification_momentum
        my_optimizer = nn.SGD(params=q_net.trainable_params(), learning_rate=lr,
                              momentum=momentum, weight_decay=weight_decay)

        forward_net = ClassificationForward(q_net)
        super().__init__(forward_net=forward_net, data_loader=data_loader, optimizer=my_optimizer,
                         epoch_num=epoch_num, evaluate_interval=evaluate_interval)
        self.save_test_result_interval = test_interval

    def score(self, output, target):
        predict = ms.ops.zeros(target.shape, dtype=ms.int32)
        predict[ms.ops.gt(output[:, 1], output[:, 0])] = 1
        correct = float(ms.ops.equal(predict, target).sum()) / target.shape[0] * 100
        return correct

    def plot(self, loss_record, train_acc_record, test_acc_record):
        plt.cla()
        plt.plot(loss_record)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('classify_loss_mindspore.png')

        plt.cla()
        plt.plot(train_acc_record)
        plt.xlabel('epoch')
        plt.ylabel('train_acc')
        plt.savefig('train_acc_mindspore.png')

        plt.cla()
        plt.plot(test_acc_record)
        plt.xlabel('epoch')
        plt.ylabel('test_acc')
        plt.savefig('test_acc_mindspore.png')
        return


class FittingTrainer(BaseTrainer):
    def __init__(self, configs: Config):
        epoch_num = configs.fitting_epoch_num
        save_interval = configs.fitting_save_interval
        evaluate_interval = configs.fitting_evaluate_interval

        r""" dataset """
        train_step = configs.fitting_train_data_step
        test_step = configs.fitting_test_data_step
        mini_batch_num = configs.fitting_mini_batch_num
        data_loader = CurveFitting(batch_size=mini_batch_num, train_step=train_step, test_step=test_step,
                                   backend='mindspore')
        data_loader.show()

        r""" circuit definition """
        encoder_circuit, ansatze_circuit = configs.encoder_circuit, configs.ansatze_circuit

        r""" variational quantum circuit """
        observe_bit = configs.fitting_observe_bit
        observe_gate = configs.fitting_observe_gate
        init_range = configs.init_range
        q_net = QNet(encoder_circuit=encoder_circuit, circuit=ansatze_circuit,
                     observe_gate=observe_gate, observe_bit=observe_bit,
                     param_init_range=init_range, input_scaling=configs.fitting_input_scaling)
        q_net.show_circuit()

        r""" optimizer """
        lr = configs.fitting_lr

        my_optimizer = nn.SGD(params=q_net.trainable_params(), learning_rate=lr,
                              momentum=configs.fitting_momentum, weight_decay=configs.fitting_weight_decay)

        forward_net = FittingForward(q_net)

        super().__init__(forward_net=forward_net, data_loader=data_loader, optimizer=my_optimizer,
                         epoch_num=epoch_num, evaluate_interval=evaluate_interval)
        self.save_test_result_interval = save_interval

    def score(self, output, target):
        r"""
        :param output: data_num * 1
        :param target: data_num
        :return:
        """
        mse = self.forward_net.loss_func(output.squeeze(1), target)
        return mse.numpy()

    def plot(self, loss_list, predict_list):
        test_x = self.data_loader.test_data.numpy()[:, 0]
        test_y = self.data_loader.test_data.numpy()[:, 1]
        plt.cla()
        plt.plot(test_x, test_y, label='correct')
        plt.scatter(self.data_loader.train_data.numpy()[:, 0], self.data_loader.train_data.numpy()[:, 1])

        for i in range(len(predict_list)):
            plt.plot(test_x, predict_list[i], label="epoch {}".format(i * self.save_test_result_interval))
        plt.legend()
        plt.savefig('fitting_results_mindspore.png')

        plt.cla()
        plt.plot(loss_list)
        plt.xlabel('epoch')
        plt.ylabel('MSE loss')
        plt.savefig('fitting_loss_mindspore.png')
        return


def init_env(configs: Config):
    if configs.device_target not in ['CPU', 'GPU', 'Ascend']:
        raise ValueError('invalid device target.')
    ms.set_context(device_target=configs.device_target)
    print('----------------------------- Env settings -----------------------------')
    print('running on {}'.format(configs.device_target))

    if configs.context_mode not in ['graph', 'pynative']:
        raise ValueError('invalid context mode.')

    if configs.context_mode == 'graph':
        ms.set_context(mode=ms.GRAPH_MODE)
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE)
    print('context mode: {}'.format(configs.context_mode))

    if isinstance(configs.device_id, int):
        ms.set_context(device_id=configs.device_id)
    print('device id: {}\n'.format(configs.device_id))
    return


if __name__ == '__main__':
    my_configs = Config()
    init_env(configs=my_configs)

    # fitting_trainer = FittingTrainer(configs=my_configs)
    # fitting_losses, fitting_train_scores, fitting_test_scores, fitting_predicts = fitting_trainer.train()
    # fitting_trainer.plot(loss_list=fitting_losses, predict_list=fitting_predicts)

    classification_trainer = ClassificationTrainer(configs=my_configs)
    losses, train_scores, test_scores, _ = classification_trainer.train()
    classification_trainer.plot(loss_record=losses, train_acc_record=train_scores, test_acc_record=test_scores)

