
import torch
from src.qnn_torch import QGate, QNet, ConstantTensor
from src.qnn_torch import check_two_bit_gate, check_single_bit_gate, check_parameterized_two_bit_gate, check_circuit
from dataset import Classification
import matplotlib.pyplot as plt

from src.qcircuit import QCircuit


def circuit_0():
    c = QCircuit(qubit_num=4, circuit_depth=5)
    c.add_1_bit_gate(gate_name='H', bit_idx=0, depth=0)
    c.add_parameterized_1_bit_gate(gate_name='Ry', bit_idx=[1, 2, 3], depth=[0, 0, 0], param=[0.2, 0.3, 0.4])

    c.add_2_bit_gate(gate_name='CX', operate_bit_idx=0, control_bit_idx=3, depth=1)
    c.add_parameterized_2_bit_gate(gate_name='CRx', operate_bit_idx=[1, 2, 3],
                                   control_bit_idx=[0, 1, 2], depth=[4, 3, 2], param=[0.2, 0.3, 0.4])
    c.show()
    return c


def circuit_1():
    c = QCircuit(qubit_num=4, circuit_depth=10)
    c.add_parameterized_1_bit_gate(gate_name='Ry', bit_idx=[0, 1, 2, 3], depth=[0, 0, 0, 0])
    c.add_parameterized_2_bit_gate(gate_name='CRx', operate_bit_idx=[0, 1, 2, 3],
                                   control_bit_idx=[3, 0, 1, 2], depth=[1, 4, 3, 2])

    c.add_parameterized_1_bit_gate(gate_name='Ry', bit_idx=[0, 1, 2, 3], depth=[5, 5, 5, 5])
    c.add_parameterized_2_bit_gate(gate_name='CRx', operate_bit_idx=[0, 1, 2, 3],
                                   control_bit_idx=[1, 2, 3, 0], depth=[8, 9, 6, 7])
    c.show()
    return c


def single_circuit():
    c = QCircuit(qubit_num=1, circuit_depth=1)
    c.add_1_bit_gate(gate_name='Y', bit_idx=[0], depth=[0])
    print(c.circuit_dict)
    c.show()
    return c


def two_bit_gate():
    c = QCircuit(qubit_num=2, circuit_depth=1)
    c.add_2_bit_gate(gate_name='CX', operate_bit_idx=[0], control_bit_idx=[1], depth=[0])
    print(c.circuit_dict)
    c.show()
    return c


def cx_circuit():
    c = QCircuit(qubit_num=3, circuit_depth=4)
    c.add_2_bit_gate(gate_name='CX', operate_bit_idx=[1, 2, 1, 0], control_bit_idx=[0, 1, 2, 1], depth=[0, 1, 2, 3])
    c.show()
    return c


circuit_0()


# device = torch.device('cpu')
# my_circuit = cx_circuit()
# check_circuit(circuit=my_circuit, device=device)

# g = QGate(device=torch.device('cpu'))
# g.check()
#


#
# c = circuit_1()
#
# check_circuit(c, device)

# dataload = Classification(device=device)
# train_data, test_data = dataload.get_data()
# train_data, test_data = train_data.numpy(), test_data.numpy()
# train_0 = train_data[:, 2] == 0
# train_1 = train_data[:, 2] == 1
#
# test_0 = test_data[:, 2] == 0
# test_1 = test_data[:, 2] == 1
#
#
# plt.scatter(train_data[:, 0][train_0], train_data[:, 1][train_0], color='red')
# plt.scatter(train_data[:, 0][train_1], train_data[:, 1][train_1], color='blue')
# plt.show()
#
# plt.scatter(test_data[:, 0][test_0], test_data[:, 1][test_0], color='red')
# plt.scatter(test_data[:, 0][test_1], test_data[:, 1][test_1], color='blue')
# plt.show()


# check_two_bit_gate(device=device)

# check_parameterized_two_bit_gate(device=device, angle_denominator=2)

# check_single_bit_gate()


# n = QNet(encoder_circuit=my_c, circuit=my_c, device=torch.device('cpu'))
#
# cons = ConstantTensor(device=torch.device('cpu'))
#
# state_0 = cons.state_0()
# state_1 = cons.state_1()
#
# input_state = torch.kron(state_1, state_1)
# state = n.check_encoder(state=input_state)
# print(state)
# state = n.check_ansatze(state=input_state)
# print(state)
# s, p = n.extract_circuit_structure(my_c)



#
# ground_state = torch.complex(torch.tensor([0., 1.], dtype=torch.float),
#                              torch.zeros((2,), dtype=torch.float))
# ground_state = ground_state.unsqueeze(1)
# s = n.forward(state=ground_state)
# print(s)







