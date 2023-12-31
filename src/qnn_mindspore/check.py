import mindspore as ms
import numpy as np
from ..qcircuit import QCircuit
from .gates import QGate
from .qnetwork import QNet


def check_single_bit_gate():
    g = QGate()
    g.check()
    return


def check_two_bit_gate():
    for gate_name in ['CX', 'CY', 'CZ', 'CH']:
        for operate_bit in range(2):
            c = QCircuit(qubit_num=2, circuit_depth=1)
            c.add_2_bit_gate(gate_name=gate_name,
                             operate_bit_idx=[operate_bit],
                             control_bit_idx=[1 - operate_bit],
                             depth=[0])
            print('\nGate name: {}'.format(gate_name))
            check_circuit(circuit=c)
    return


def check_parameterized_two_bit_gate(angle_denominator=2):
    for gate_name in ['CRx', 'CRy', 'CRz']:
        for operate_bit in range(2):
            c = QCircuit(qubit_num=2, circuit_depth=1)
            c.add_parameterized_2_bit_gate(gate_name=gate_name,
                                           operate_bit_idx=[operate_bit],
                                           control_bit_idx=[1 - operate_bit],
                                           depth=[0],
                                           param=[np.pi / angle_denominator])
            print('\nGate name: {} rotation angle: pi / {}'.format(gate_name, angle_denominator))
            check_circuit(circuit=c)
    return


def check_cx_circuit():
    c = QCircuit(qubit_num=3, circuit_depth=4)

    # circuit 1
    c.add_2_bit_gate(gate_name='CX', operate_bit_idx=[1, 2, 1, 0], control_bit_idx=[0, 1, 2, 1], depth=[0, 1, 2, 3])

    # circuit 2
    # c.add_2_bit_gate(gate_name='CX', operate_bit_idx=[0, 2, 1, 0], control_bit_idx=[2, 0, 0, 1], depth=[0, 1, 2, 3])
    # c.add_1_bit_gate(gate_name='X', bit_idx=[1, ], depth=[0, ])

    check_circuit(circuit=c, if_observe=True, observe_gate='Z')
    return


def check_circuit(circuit: QCircuit, if_observe: bool = False, observe_gate: str = 'Z'):
    circuit.show()
    qubit_num = circuit.qubit_num
    n = QNet(encoder_circuit=circuit, circuit=circuit, constant_real_mode=False,
             observe_bit=list(range(qubit_num)), observe_gate=[observe_gate for i in range(qubit_num)])
    state_0 = n.constant.state_0()
    state_1 = n.constant.state_1()
    basic_states = [state_0, state_1]
    qubit_num = circuit.qubit_num

    states = []
    for state_idx in range(2**qubit_num):
        bin_idx = "{:0{}b}".format(state_idx, qubit_num)
        each_state = n.constant.empty()
        state_vector = ""
        for bit_idx in range(qubit_num):
            bit = int(bin_idx[bit_idx])
            each_state = ms.ops.kron(each_state, basic_states[bit])
            state_vector += "|{}>".format(bit)
        print("\ninput state: ", state_vector)
        print(each_state)
        states.append(each_state)
        output = n.check_ansatze(state=each_state, if_observe=if_observe)

        if if_observe:
            expectation_0, expectation_1 = output
            print("gate {} expectation 0: ".format(observe_gate))
            print(expectation_0.real())
            print("gate {} expectation 1: ".format(observe_gate))
            print(expectation_1.real())

            output_state = ""
            for i in range(expectation_0.shape[0]):
                output_state += "({}|0> + {}|1>) ".format(expectation_0.real()[i], expectation_1.real()[i])
            print('output state:')
            print(output_state)
        else:
            print('output state:')
            print(output)
    return

