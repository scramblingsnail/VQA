import torch
import numpy as np
from src import Classification, CurveFitting, Config
from src.qnn_torch import QNet
import matplotlib.pyplot as plt


def train(q_model: QNet, train_loader, optimizer, loss_func, device):
    q_model.train()
    loss_record = []
    for batch_input, batch_label in train_loader.get_train_data(shuffle=True):
        # mini batch train
        batch_output = []
        batch_input, batch_label = batch_input.to(device), batch_label.to(device)
        optimizer.zero_grad()
        for idx in range(batch_input.size()[0]):
            expect_0, expect_1 = q_model.forward(batch_input[idx])
            batch_output.append(expect_0)
        batch_output = torch.stack(batch_output, dim=0)

        if isinstance(loss_func, torch.nn.MSELoss):
            batch_output = batch_output.squeeze(1)

        loss = loss_func(input=batch_output, target=batch_label)

        loss.backward()
        optimizer.step()
        loss_record.append(loss.detach().item())

    train_loss = np.mean(loss_record)
    print('train loss: ', train_loss)
    return train_loss


def evaluate(q_model: QNet, test_loader, device, flag='test'):
    r"""
    :param q_model:
    :param test_loader:
    :param device:
    :param flag:
    :return: output: data_num * 1; label: data_num
    """
    output = []
    label = []
    q_model.eval()

    with torch.no_grad():
        if flag == 'train':
            evaluate_data = test_loader.get_train_data(shuffle=False)
        else:
            evaluate_data = test_loader.get_test_data()
        for batch_input, batch_label in evaluate_data:
            batch_output = []
            batch_input, batch_label = batch_input.to(device), batch_label.to(device)
            for idx in range(batch_input.size()[0]):
                expect_0, expect_1 = q_model.forward(batch_input[idx])
                batch_output.append(expect_0)

            batch_output = torch.stack(batch_output, dim=0)
            output.append(batch_output)
            label.append(batch_label)
        output = torch.cat(output, dim=0)
        label = torch.cat(label, dim=0)
    return output, label


def fitting_demo(configs: Config):
    r""" device """
    device = configs.device
    observe_bit = configs.fitting_observe_bit
    observe_gate = configs.fitting_observe_gate
    mini_batch_num = configs.fitting_mini_batch_num
    epoch_num = configs.fitting_epoch_num
    lr = configs.fitting_lr
    init_range = configs.init_range
    save_interval = configs.fitting_save_interval

    r""" dataset """
    train_step = configs.fitting_train_data_step
    test_step = configs.fitting_test_data_step
    data_loader = CurveFitting(batch_size=mini_batch_num, train_step=train_step, test_step=test_step, backend='torch')
    data_loader.show()

    r""" circuit definition """
    encoder_circuit, ansatze_circuit = configs.encoder_circuit, configs.ansatze_circuit

    r""" variational quantum circuit """
    q_net = QNet(encoder_circuit=encoder_circuit, circuit=ansatze_circuit,
                 observe_gate=observe_gate, observe_bit=observe_bit,
                 device=device, param_init_range=init_range, input_scaling=configs.fitting_input_scaling)
    q_net.show_circuit()

    my_optimizer = torch.optim.SGD(params=q_net.parameters(), lr=lr,
                                   momentum=configs.fitting_momentum, weight_decay=configs.fitting_weight_decay)
    loss_func = torch.nn.MSELoss()

    predict_list = []

    r""" train """
    for epoch in range(epoch_num):
        print('epoch {}'.format(epoch))
        print('lr: ', my_optimizer.param_groups[0]['lr'])
        train(q_model=q_net, train_loader=data_loader, optimizer=my_optimizer, loss_func=loss_func, device=device)
        predict, label = evaluate(q_model=q_net, test_loader=data_loader, device=device)

        predict = predict.squeeze(1)

        if epoch % save_interval == 0:
            predict_list.append(predict.numpy())

        evaluate_loss = loss_func(input=predict, target=label)
        print('test loss: {}\n'.format(evaluate_loss))

    r""" save param """
    with open('param.txt', 'w+') as p_f:
        for name, param in q_net.named_parameters():
            if param.requires_grad:
                p_f.write('{}\t{:.9f}\n'.format(name, param.detach().item()))

    test_x = data_loader.test_data.numpy()[:, 0]
    test_y = data_loader.test_data.numpy()[:, 1]
    plt.cla()
    plt.plot(test_x, test_y, label='correct')
    plt.scatter(data_loader.train_data.numpy()[:, 0], data_loader.train_data.numpy()[:, 1])

    for i in range(len(predict_list)):
        plt.plot(test_x, predict_list[i], label="epoch {}".format(i * save_interval))
    plt.legend()
    plt.savefig(r'fitting_result_torch.png')
    return


def classification_demo(configs: Config):
    r""" device """
    device = configs.device
    observe_bit = configs.classification_observe_bit
    observe_gate = configs.classification_observe_gate
    train_data_num = configs.classification_train_data_num
    test_data_num = configs.classification_test_data_num
    mini_batch_num = configs.classification_mini_batch_num
    epoch_num = configs.classification_epoch_num
    lr = configs.classification_lr
    init_range = configs.init_range

    r""" circuit definition """
    encoder_circuit, ansatze_circuit = configs.encoder_circuit, configs.ansatze_circuit

    r""" dataset """
    data_loader = Classification(batch_size=mini_batch_num, train_data_num=train_data_num, test_data_num=test_data_num,
                                 backend='torch')
    data_loader.show()

    r""" variational quantum circuit """
    q_net = QNet(encoder_circuit=encoder_circuit, circuit=ansatze_circuit,
                 observe_gate=observe_gate, observe_bit=observe_bit,
                 device=device, param_init_range=init_range, input_scaling=configs.classification_input_scaling)
    q_net.show_circuit()

    my_optimizer = torch.optim.SGD(params=q_net.parameters(), lr=lr,
                                   momentum=configs.classification_momentum,
                                   weight_decay=configs.classification_weight_decay)
    loss_func = torch.nn.CrossEntropyLoss()

    r""" train """
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    for epoch in range(epoch_num):
        print('epoch {}'.format(epoch))
        print('lr: ', my_optimizer.param_groups[0]['lr'])
        train_loss = train(q_model=q_net, train_loader=data_loader, optimizer=my_optimizer,
                           loss_func=loss_func, device=device)
        prob, label = evaluate(q_model=q_net, test_loader=data_loader, device=device, flag='train')
        predict = torch.zeros(label.size(), device=device, dtype=torch.long)
        predict[torch.gt(prob[:, 1], prob[:, 0])] = 1

        correct = torch.eq(predict, label).sum() / label.size()[0] * 100
        train_acc_list.append(correct)

        prob_test, label_test = evaluate(q_model=q_net, test_loader=data_loader, device=device, flag='test')
        predict_test = torch.zeros(label_test.size(), device=device, dtype=torch.long)
        predict_test[torch.gt(prob_test[:, 1], prob_test[:, 0])] = 1

        correct_test = torch.eq(predict_test, label_test).sum() / label_test.size()[0] * 100
        test_acc_list.append(correct_test)

        train_loss_list.append(train_loss)
        print('train acc: {}\n'.format(correct))
        print('test acc: {}\n'.format(correct_test))
        # if correct_test >= 99:
        #     break

    plt.cla()
    plt.plot(train_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('classification_loss_torch.png')

    plt.cla()
    plt.plot(train_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('train acc')
    plt.savefig('classification_train_acc_torch.png')

    plt.cla()
    plt.plot(test_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('test acc')
    plt.savefig('classification_test_acc_torch.png')
    return


if __name__ == '__main__':
    my_configs = Config()
    print('---------------------- training fitting task ----------------------')
    fitting_demo(my_configs)

    print('---------------------- training classification task ----------------------')
    classification_demo(my_configs)


