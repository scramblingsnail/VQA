import mindspore as ms
from mindspore import nn, Parameter
import numpy as np
from ..qcircuit import QCircuit
from .gates import QGate
from .constant import ConstantTensor
from .observe import Observer
from typing import Union, List, Tuple


class QNet(nn.Cell):
    def __init__(self,
                 encoder_circuit: QCircuit,
                 circuit: QCircuit,
                 observe_gate: Union[str, List[str]],
                 observe_bit: Union[int, List[int]],
                 constant_real_mode: bool = False,
                 param_init_range: Union[List, Tuple] = None,
                 input_scaling=np.pi):
        super().__init__()
        self.encoder_circuit = encoder_circuit
        self.circuit = circuit
        self.input_scaling = input_scaling
        if param_init_range is None:
            param_init_range = (- np.pi / circuit.circuit_depth, np.pi / circuit.circuit_depth)
        self.param_init_range = tuple(param_init_range)
        if self.encoder_circuit.qubit_num != self.circuit.qubit_num:
            raise ValueError('encoder qu-bit num do not match the ansatze qu-bit num')

        self.observer = Observer(qubit_num=self.circuit.qubit_num, observe_bit=observe_bit, observe_gate=observe_gate)
        self.gates_lib = QGate()
        self.constant = ConstantTensor(real_mode=constant_real_mode)
        self.input_param_position = []
        self._register_circuit_param()
        encoder_circuit_info = self.extract_circuit_structure(self.encoder_circuit)
        circuit_info = self.extract_circuit_structure(self.circuit)
        self.encoder_circuit_structure, self.encoder_control_direction = encoder_circuit_info
        self.circuit_structure, self.control_direction = circuit_info
        self._circuit_flags = ('encoder', 'ansatze')
        self.complex = ms.ops.Complex()
        self.param_name_head = None

    def _init_param(self):
        return np.random.uniform(self.param_init_range[0], self.param_init_range[1])

    def _register_circuit_param(self):
        # input encoder circuit:
        for depth in range(self.encoder_circuit.circuit_depth):
            for bit in self.encoder_circuit.circuit_dict[depth].keys():
                param = self.encoder_circuit.circuit_dict[depth][bit]['param']
                if param is not None:
                    register_name = 'encoder_depth{}bit{}'.format(depth, bit)
                    self.insert_param_to_cell(param_name=register_name,
                                              param=Parameter(name=register_name,
                                                              default_input=ms.Tensor([param], dtype=ms.float32)))
                    # params do not require grad.
                    self.parameters_dict()[register_name].requires_grad = False
                    self.input_param_position.append((depth, bit))

        # variational ansatze
        for depth in range(self.circuit.circuit_depth):
            for bit in self.circuit.circuit_dict[depth].keys():
                param = self.circuit.circuit_dict[depth][bit]['param']
                if param is not None:
                    register_name = 'ansatze_depth{}bit{}'.format(depth, bit)
                    param = self._init_param()
                    self.insert_param_to_cell(param_name=register_name,
                                              param=Parameter(name=register_name,
                                                              default_input=ms.Tensor([param], dtype=ms.float32)))
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
            { depth0: (gate structure at each depth), ... }
        for each depth:
            ( (each structure part), ... )
        structure part is defined as follow:
            (bit): operation bit of single bit gate
            (start_bit, passby_gate_0, ..., end_bit): structure part defined by a 2-bit controlled gate.
                start_bit: min(operation_bit, control_bit)
                passby_gate_0, ... : single bit gate between start_bit and end_bit.
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

    def limited_kron(self, complex_tensor, real_tensor, real_first: bool):
        if real_first:
            new_real = ms.ops.kron(real_tensor, complex_tensor.real())
            new_imag = ms.ops.kron(real_tensor, complex_tensor.imag())
        else:
            new_real = ms.ops.kron(complex_tensor.real(), real_tensor)
            new_imag = ms.ops.kron(complex_tensor.imag(), real_tensor)
        new_tensor = self.complex(new_real, new_imag)
        return new_tensor

    def _limited_single_step(self, input_state, gate_dict: dict, depth: int, flag: str):
        assert flag == 'ansatze', 'temporary single step for ansatze circuit.'

        if len(self.circuit_structure[depth]) > 1:
            raise ValueError('Sorry, the mindspore version only support single gate per depth. '
                             'because mindspore do not support MUl_grad for complex tensor now.')

        def get_tensor(bit):
            if bit not in gate_dict.keys():
                return self.constant.eye(real_mode=True)
            else:
                gate_name = gate_dict[bit]['name']
                control_bit = gate_dict[bit]['control_bit']
                param = gate_dict[bit]['param']

                if param is not None:
                    param = self.parameters_dict()['{}{}_depth{}bit{}'.format(self.param_name_head, flag, depth, bit)]
                assert control_bit is None, 'wrong gate type'
                return self.gates_lib.gate_tensor(gate_name=gate_name, param=param)

        def get_controlled_tensor(structure_part: tuple, upper_control_bit: bool):
            r"""
            if control_bit > bit:
                tensor = I (kron) I (kron)... I (kron) |0><0| + inside_gate_tensor (kron) I (kron) ... I (kron) |1><1|
            else:
                tensor = |0><0| (kron) I (kron)... I (kron) I + |1><1| (kron) I (kron) ... I (kron) inside_gate_tensor
            """
            tensor_0 = self.constant.empty(real_mode=True)
            tensor_1 = self.constant.empty(real_mode=True)
            eye = self.constant.eye(real_mode=True)
            complex_empty = self.constant.empty(real_mode=False)
            projection_0 = self.constant.projection_0(real_mode=True)
            projection_1 = self.constant.projection_1(real_mode=True)

            if upper_control_bit:
                bit = structure_part[0]
                control_bit = structure_part[-1]
            else:
                bit = structure_part[-1]
                control_bit = structure_part[0]

            gate_name = gate_dict[bit]['name']
            param = gate_dict[bit]['param']
            if param is not None:
                param = self.parameters_dict()['{}{}_depth{}bit{}'.format(self.param_name_head, flag, depth, bit)]
            son_gate_tensor = self.gates_lib.gate_tensor(gate_name=gate_name, param=param)
            start_bit = min(bit, control_bit)
            end_bit = max(bit, control_bit)

            if upper_control_bit:
                start_0 = eye
                start_1 = son_gate_tensor
                end_0 = projection_0
                end_1 = projection_1

                tensor_0 = ms.ops.kron(tensor_0, start_0)
                tensor_1 = self.limited_kron(complex_tensor=start_1, real_tensor=tensor_1, real_first=True)
                for passby_bit in range(start_bit + 1, end_bit):
                    tensor_0 = ms.ops.kron(tensor_0, eye)
                    tensor_1 = self.limited_kron(complex_tensor=tensor_1, real_tensor=eye, real_first=False)

                tensor_0 = ms.ops.kron(tensor_0, end_0)
                # to complex:
                tensor_0 = self.limited_kron(complex_tensor=complex_empty, real_tensor=tensor_0, real_first=False)
                tensor_1 = self.limited_kron(complex_tensor=tensor_1, real_tensor=end_1, real_first=False)
            else:
                start_0 = projection_0
                start_1 = projection_1
                end_0 = eye
                end_1 = son_gate_tensor

                tensor_0 = ms.ops.kron(tensor_0, start_0)
                tensor_1 = ms.ops.kron(tensor_1, start_1)
                for passby_bit in range(start_bit + 1, end_bit):
                    tensor_0 = ms.ops.kron(tensor_0, eye)
                    tensor_1 = ms.ops.kron(tensor_1, eye)

                tensor_0 = ms.ops.kron(tensor_0, end_0)
                # to complex:
                tensor_0 = self.limited_kron(complex_tensor=complex_empty, real_tensor=tensor_0, real_first=False)
                tensor_1 = self.limited_kron(complex_tensor=end_1, real_tensor=tensor_1, real_first=True)

            return tensor_0 + tensor_1

        layer_control_direction = self.control_direction[depth]
        hamiltonian = self.constant.empty(real_mode=True)
        real_eye = self.constant.eye(real_mode=True)
        start_bit = 0

        for layer_part in self.circuit_structure[depth]:
            for bit_idx in range(start_bit, layer_part[0]):
                hamiltonian = ms.ops.kron(hamiltonian, real_eye)
            if len(layer_part) > 1:
                control_pair = (layer_part[0], layer_part[-1])
                part_tensor = get_controlled_tensor(layer_part, layer_control_direction[control_pair])
            else:
                part_tensor = get_tensor(layer_part[0])

            hamiltonian = self.limited_kron(complex_tensor=part_tensor, real_tensor=hamiltonian, real_first=True)
            start_bit = layer_part[-1] + 1

        for bit_idx in range(start_bit, self.circuit.qubit_num):
            hamiltonian = self.limited_kron(complex_tensor=hamiltonian, real_tensor=real_eye, real_first=False)
        return ms.ops.mm(hamiltonian, input_state)

    def _single_step(self, input_state, gate_dict: dict, depth: int, flag: str):

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
                    param = self.parameters_dict()['{}{}_depth{}bit{}'.format(self.param_name_head, flag, depth, bit)]
                assert control_bit is None, 'wrong gate type'
                return self.gates_lib.gate_tensor(gate_name=gate_name, param=param)

        def get_controlled_tensor(structure_part: tuple, upper_control_bit: bool):
            r"""
            Args:
                structure_part: (start bit, ... end bit)
                upper_control_bit: if control bit > operation bit
            if control_bit > bit:
                tensor = I (kron) Passby_1 (kron)... Passby_n (kron) |0><0| +
                            inside_gate_tensor (kron) Passby_1 (kron) ... Passby_n (kron) |1><1|
            else:
                tensor = |0><0| (kron) Passby_1 (kron)... Passby_n (kron) I +
                            |1><1| (kron) Passby_1 (kron) ... Passby_n (kron) inside_gate_tensor
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
                param = self.parameters_dict()['{}{}_depth{}bit{}'.format(self.param_name_head, flag, depth, bit)]
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

            tensor_0 = ms.ops.kron(tensor_0, start_0)
            tensor_1 = ms.ops.kron(tensor_1, start_1)
            for passby_bit in range(start_bit + 1, end_bit):
                passby_tensor = get_tensor(passby_bit)
                tensor_0 = ms.ops.kron(tensor_0, passby_tensor)
                tensor_1 = ms.ops.kron(tensor_1, passby_tensor)

            tensor_0 = ms.ops.kron(tensor_0, end_0)
            tensor_1 = ms.ops.kron(tensor_1, end_1)
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
                hamiltonian = ms.ops.kron(hamiltonian, each_tensor)
            if len(layer_part) > 1:
                control_pair = (layer_part[0], layer_part[-1])
                part_tensor = get_controlled_tensor(layer_part, layer_control_direction[control_pair])
                hamiltonian = ms.ops.kron(hamiltonian, part_tensor)
            else:
                part_tensor = get_tensor(layer_part[0])
                hamiltonian = ms.ops.kron(hamiltonian, part_tensor)
            start_bit = layer_part[-1] + 1
        for bit_idx in range(start_bit, self.circuit.qubit_num):
            each_tensor = get_tensor(bit_idx)
            hamiltonian = ms.ops.kron(hamiltonian, each_tensor)

        return ms.ops.mm(hamiltonian, input_state)

    def _set_global_name(self):
        # get the param head
        if self.param_name_head is None:
            for name, param in self.parameters_and_names():
                global_name = param.name
                if len(global_name.split('encoder_depth')) > 1:
                    self.param_name_head = global_name.split('encoder_depth')[0]
                    break
        return

    def input_encoding(self, input_tensor):
        # the input tensor size should be less than the number of params in the first layer acting as input encoding.
        if len(input_tensor.shape) >= 2:
            raise ValueError('expect 1-dim input tensor.')
        input_param_size = len(self.input_param_position)
        input_size = input_tensor.shape[0]
        assert input_size <= input_param_size

        input_tensor = input_tensor.unsqueeze(1) * self.input_scaling
        for input_idx in range(input_param_size):
            param_name = '{}encoder_depth{}bit{}'.format(self.param_name_head,
                                                         self.input_param_position[input_idx][0],
                                                         self.input_param_position[input_idx][1])
            self.parameters_dict()[param_name].set_data(input_tensor[input_idx % input_size])
        return

    def observe(self, ket):
        r"""
        :param ket: (2 ** qubit_num) * 1
        :return:
        """
        # observe:
        projection_0, projection_1 = self.observer.get_observe_tensor()

        ket = ket.unsqueeze(0).broadcast_to((projection_0.shape[0], ket.shape[0], ket.shape[1]))
        # bra vector
        bra = self.complex(ket.real(), - ket.imag())
        bra = ms.ops.swapaxes(bra, 1, 2)

        expectation_0 = ms.ops.matmul(bra, projection_0)
        expectation_0 = ms.ops.matmul(expectation_0, ket).squeeze(1).squeeze(1)

        expectation_1 = ms.ops.matmul(bra, projection_1)
        expectation_1 = ms.ops.matmul(expectation_1, ket).squeeze(1).squeeze(1)
        return expectation_0, expectation_1

    def check_encoder(self, state, if_observe: bool = False):
        self._set_global_name()
        for depth in range(self.encoder_circuit.circuit_depth):
            state = self._single_step(state, self.encoder_circuit.circuit_dict[depth], depth, flag='encoder')
        if if_observe:
            expectation_0, expectation_1 = self.observe(state)
            return expectation_0, expectation_1
        else:
            return state

    def check_ansatze(self, state, if_observe: bool = False):
        self._set_global_name()

        for depth in range(self.circuit.circuit_depth):
            r""" Temporary: limited single step """
            # limited single step
            state = self._limited_single_step(input_state=state,
                                              gate_dict=self.circuit.circuit_dict[depth], depth=depth, flag='ansatze')

            # normal single step
            # state = self._single_step(state, self.circuit.circuit_dict[depth], depth, flag='ansatze')
        if if_observe:
            expectation_0, expectation_1 = self.observe(state)
            return expectation_0, expectation_1
        else:
            return state

    def construct(self, inputs):
        r"""
        :param inputs: (qubit_num, )
        :return:
        """

        self._set_global_name()
        # input encoding.
        self.input_encoding(inputs)

        ground_state = self.constant.state_0(real_mode=False)
        # initial state
        state = self.constant.empty(real_mode=False)
        for i in range(self.circuit.qubit_num):
            state = ms.ops.kron(state, ground_state)

        # encoded state:
        for depth in range(self.encoder_circuit.circuit_depth):
            state = self._single_step(state, self.encoder_circuit.circuit_dict[depth], depth, flag='encoder')

        # evolution along the quantum circuit.
        for depth in range(self.circuit.circuit_depth):
            r""" Temporary: limited single step """
            # limited single step
            state = self._limited_single_step(input_state=state, gate_dict=self.circuit.circuit_dict[depth],
                                              depth=depth, flag='ansatze')

            # Normal single step
            # state = self._single_step(state, self.circuit.circuit_dict[depth], depth, flag='ansatze')

        # observe:
        expectation_0, expectation_1 = self.observe(state)
        return expectation_0.real(), expectation_1.real()
