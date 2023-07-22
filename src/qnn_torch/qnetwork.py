import torch
import torch.nn as nn
import numpy as np
from ..qcircuit import QCircuit
from .gates import QGate
from .constant import ConstantTensor
from .observe import Observer
from typing import Union, List, Tuple


class QNet(nn.Module):
    def __init__(self,
                 encoder_circuit: QCircuit,
                 circuit: QCircuit,
                 observe_gate: Union[str, List[str]],
                 observe_bit: Union[int, List[int]],
                 device: torch.device,
                 param_init_range: Union[List, Tuple] = None,
                 input_scaling=np.pi):
        super().__init__()
        self.encoder_circuit = encoder_circuit
        self.circuit = circuit
        self.device = device
        self.input_scaling = input_scaling
        if param_init_range is None:
            param_init_range = (- np.pi / circuit.circuit_depth, np.pi / circuit.circuit_depth)
        self.param_init_range = tuple(param_init_range)
        if self.encoder_circuit.qubit_num != self.circuit.qubit_num:
            raise ValueError('encoder qu-bit num do not match the ansatze qu-bit num')

        self.observer = Observer(device=self.device, qubit_num=self.circuit.qubit_num,
                                 observe_bit=observe_bit, observe_gate=observe_gate)
        self.gates_lib = QGate(device=self.device)
        self.constant = ConstantTensor(device=self.device)
        self.input_param_position = []
        self._register_circuit_param()
        encoder_circuit_info = self.extract_circuit_structure(self.encoder_circuit)
        circuit_info = self.extract_circuit_structure(self.circuit)
        self.encoder_circuit_structure, self.encoder_control_direction = encoder_circuit_info
        self.circuit_structure, self.control_direction = circuit_info
        self._circuit_flags = ('encoder', 'ansatze')

    def _init_param(self):
        return np.random.uniform(self.param_init_range[0], self.param_init_range[1])

    def _register_circuit_param(self):
        # input encoder circuit:
        for depth in range(self.encoder_circuit.circuit_depth):
            for bit in self.encoder_circuit.circuit_dict[depth].keys():
                param = self.encoder_circuit.circuit_dict[depth][bit]['param']
                if param is not None:
                    self.register_parameter('encoder_depth{}bit{}'.format(depth, bit),
                                            nn.Parameter(torch.tensor([param], dtype=torch.float, device=self.device)))
                    # params do not require grad.
                    self._parameters['encoder_depth{}bit{}'.format(depth, bit)].requires_grad = False
                    self.input_param_position.append((depth, bit))

        # variational ansatze
        for depth in range(self.circuit.circuit_depth):
            for bit in self.circuit.circuit_dict[depth].keys():
                param = self.circuit.circuit_dict[depth][bit]['param']
                if param is not None:
                    param = self._init_param()
                    self.register_parameter('ansatze_depth{}bit{}'.format(depth, bit),
                                            nn.Parameter(torch.tensor([param], dtype=torch.float, device=self.device)))
        return

    def show_circuit(self):
        print('Encoder circuit:')
        self.encoder_circuit.show()
        print('Ansatze circuit:')
        self.circuit.show()

    @staticmethod
    def extract_circuit_structure(circuit: QCircuit):
        r"""
        extract circuit structure for generating Hamiltonian. restore circuit structure as follows:
            [ [gate structure at each depth], ... ]
        for each depth:
            [ [each structure part], ... ]
        structure part is defined as follow:
            [bit]: operation bit of single bit gate
            [start_bit, bit0, ..., end_bit]: structure part defined by a 2-bit controlled gate.
                start_bit: min(operation_bit, control_bit)
                bit0, ... : single bit gate between start_bit and end_bit.
                end_bit: max(operation_bit, control_bit)
        """
        # (control_pair): upper_control_bit: True or False
        control_pairs = {}
        circuit_structure = {}
        for depth in range(circuit.circuit_depth):
            control_pairs[depth] = {}
            gate_dict = circuit.circuit_dict[depth]
            for bit in gate_dict.keys():
                control_bit = gate_dict[bit]['control_bit']
                if control_bit is not None:
                    control_pairs[depth][(min(bit, control_bit), max(bit, control_bit))] = bit < control_bit

        for depth in range(circuit.circuit_depth):
            circuit_structure[depth] = []
            gate_dict = circuit.circuit_dict[depth]
            control_pairs_keys = list(control_pairs[depth].keys())
            two_bit_part_idx = 0
            parts_num = len(control_pairs_keys)
            for bit in range(circuit.qubit_num):
                if two_bit_part_idx < parts_num:
                    two_bit_start = control_pairs_keys[two_bit_part_idx][0]
                    two_bit_end = control_pairs_keys[two_bit_part_idx][1]
                    if bit == two_bit_start:
                        circuit_structure[depth].append([two_bit_start])
                        continue
                    elif two_bit_start < bit < two_bit_end:
                        if bit in gate_dict.keys():
                            circuit_structure[depth][-1].append(bit)
                            continue
                    elif bit == two_bit_end:
                        circuit_structure[depth][-1].append(two_bit_end)
                        circuit_structure[depth][-1] = tuple(circuit_structure[depth][-1])
                        two_bit_part_idx += 1
                        continue
                if bit in gate_dict.keys():
                    circuit_structure[depth].append((bit,))
            circuit_structure[depth] = tuple(circuit_structure[depth])
        return circuit_structure, control_pairs

    def _single_step(self, input_state: torch.complex, gate_dict: dict, depth: int, flag: str):
        r"""
        :param input_state:
        :param gate_dict: gate dict of specific depth:
                    {operation bit: {'name': gate name, 'param': gate param, 'control_bit': control bit}, ...}
        :return:
        """

        def get_tensor(bit):
            if bit not in gate_dict.keys():
                return self.constant.eye()
            else:
                gate_name = gate_dict[bit]['name']
                control_bit = gate_dict[bit]['control_bit']
                param = gate_dict[bit]['param']

                if param is not None:
                    param = self._parameters['{}_depth{}bit{}'.format(flag, depth, bit)]
                assert control_bit is None, 'wrong gate type'
                return self.gates_lib.gate_tensor(gate_name=gate_name, param=param)

        def get_controlled_tensor(structure_part: tuple, upper_control_bit: bool):
            r"""
            if control_bit > bit:
                tensor = I (kron) I (kron)... I (kron) |0><0| + inside_gate_tensor (kron) I (kron) ... I (kron) |1><1|
            else:
                tensor = |0><0| (kron) I (kron)... I (kron) I + |1><1| (kron) I (kron) ... I (kron) inside_gate_tensor
            """
            tensor_0 = self.constant.empty()
            tensor_1 = self.constant.empty()
            eye = self.constant.eye()
            projection_0 = self.constant.projection_0()
            projection_1 = self.constant.projection_1()

            if upper_control_bit:
                bit = structure_part[0]
                control_bit = structure_part[-1]
            else:
                bit = structure_part[-1]
                control_bit = structure_part[0]

            gate_name = gate_dict[bit]['name']
            param = gate_dict[bit]['param']
            if param is not None:
                param = self._parameters['{}_depth{}bit{}'.format(flag, depth, bit)]
            son_gate_tensor = self.gates_lib.gate_tensor(gate_name=gate_name, param=param)

            if upper_control_bit:
                start_0 = eye
                start_1 = son_gate_tensor
                end_0 = projection_0
                end_1 = projection_1
            else:
                start_0 = projection_0
                start_1 = projection_1
                end_0 = eye
                end_1 = son_gate_tensor
            start_bit = min(bit, control_bit)
            end_bit = max(bit, control_bit)

            tensor_0 = torch.kron(tensor_0, start_0)
            tensor_1 = torch.kron(tensor_1, start_1)
            for passby_bit in range(start_bit + 1, end_bit):
                passby_tensor = get_tensor(passby_bit)
                tensor_0 = torch.kron(tensor_0, passby_tensor)
                tensor_1 = torch.kron(tensor_1, passby_tensor)

            tensor_0 = torch.kron(tensor_0, end_0)
            tensor_1 = torch.kron(tensor_1, end_1)
            return tensor_0 + tensor_1

        assert flag in self._circuit_flags
        hamiltonian = self.constant.empty()
        if flag == 'encoder':
            layer_structure = self.encoder_circuit_structure[depth]
            layer_control_direction = self.encoder_control_direction[depth]
        else:
            layer_structure = self.circuit_structure[depth]
            layer_control_direction = self.control_direction[depth]

        start_bit = 0
        for layer_part in layer_structure:
            for bit_idx in range(start_bit, layer_part[0]):
                each_tensor = get_tensor(bit_idx)
                hamiltonian = torch.kron(hamiltonian, each_tensor)
            if len(layer_part) > 1:
                control_pair = (layer_part[0], layer_part[-1])
                part_tensor = get_controlled_tensor(layer_part, layer_control_direction[control_pair])
                hamiltonian = torch.kron(hamiltonian, part_tensor)
            else:
                part_tensor = get_tensor(layer_part[0])
                hamiltonian = torch.kron(hamiltonian, part_tensor)
            start_bit = layer_part[-1] + 1
        for bit_idx in range(start_bit, self.circuit.qubit_num):
            each_tensor = get_tensor(bit_idx)
            hamiltonian = torch.kron(hamiltonian, each_tensor)
        return torch.mm(hamiltonian, input_state)

    def input_encoding(self, input_tensor):
        # the input tensor size should be less than the number of params in the first layer acting as input encoding.
        input_param_size = len(self.input_param_position)
        input_size = input_tensor.size()[0]
        input_tensor = input_tensor * self.input_scaling
        assert input_size <= input_param_size
        for input_idx in range(input_param_size):
            param_name = 'encoder_depth{}bit{}'.format(self.input_param_position[input_idx][0],
                                                       self.input_param_position[input_idx][1])
            self._parameters[param_name].data = input_tensor[input_idx % input_size]
        return

    def observe(self, ket: torch.complex):
        r"""
        :param ket: (2 ** qubit_num) * 1
        :return:
        """
        # observe:
        projection_0, projection_1 = self.observer.get_observe_tensor()

        ket = ket.unsqueeze(0).expand((projection_0.size()[0], ket.size()[0], ket.size()[1]))
        # bra vector
        bra = torch.transpose(ket.clone(), 1, 2)
        bra.imag = - bra.imag

        expectation_0 = torch.matmul(bra, projection_0)
        expectation_0 = torch.matmul(expectation_0, ket).squeeze(1).squeeze(1)

        expectation_1 = torch.matmul(bra, projection_1)
        expectation_1 = torch.matmul(expectation_1, ket).squeeze(1).squeeze(1)
        return expectation_0, expectation_1

    def check_encoder(self, state: torch.complex, if_observe: bool = False):
        for depth in range(self.encoder_circuit.circuit_depth):
            state = self._single_step(state, self.encoder_circuit.circuit_dict[depth], depth, flag='encoder')
        if if_observe:
            expectation_0, expectation_1 = self.observe(state)
            return expectation_0, expectation_1
        else:
            return state

    def check_ansatze(self, state: torch.complex, if_observe: bool = False):
        for depth in range(self.circuit.circuit_depth):
            state = self._single_step(state, self.circuit.circuit_dict[depth], depth, flag='ansatze')
        if if_observe:
            expectation_0, expectation_1 = self.observe(state)
            return expectation_0, expectation_1
        else:
            return state

    def forward(self, inputs):
        r"""
        :param inputs: (qubit_num, )
        :return:
        """
        # input encoding.
        self.input_encoding(inputs)

        ground_state = torch.complex(torch.tensor([1., 0.], dtype=torch.float, device=self.device),
                                     torch.zeros((2,), dtype=torch.float, device=self.device))
        # initial state
        state = torch.ones((1,), dtype=torch.float, device=self.device)
        for i in range(self.circuit.qubit_num):
            state = torch.kron(state, ground_state)
        state = state.unsqueeze(1)
        # print('initial state: ', state)

        # encoded state:
        for depth in range(self.encoder_circuit.circuit_depth):
            state = self._single_step(state, self.encoder_circuit.circuit_dict[depth], depth, flag='encoder')

        # evolution along the quantum circuit.
        for depth in range(self.circuit.circuit_depth):
            state = self._single_step(state, self.circuit.circuit_dict[depth], depth, flag='ansatze')

        # observe:
        expectation_0, expectation_1 = self.observe(state)
        return expectation_0.real, expectation_1.real
