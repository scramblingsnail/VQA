import numpy as np
from typing import Union, Tuple, List


class QCircuit:
    r"""
    this class is for quantum circuit definition
    the circuit structure is recorded as:
        {(depth) 0: {operation_bit: {'name': gate_name, 'param': gate_param, 'control_bit': control_bit}, ...}, ...}
    for simplicity, only single bit gate and two bit controlled gate are taken into consideration.
    Furthermore, constrain that the control link do not overlap along the qu-bit dimension.
    """
    def __init__(self, qubit_num: int, circuit_depth: int):
        self._angle_range = (0, np.pi)
        self._single_bit_gate_name = ('X', 'Y', 'Z', 'H')
        self._para_single_bit_gate_name = ('Rx', 'Ry', 'Rz')
        self._two_bit_gate_name = ('CX', 'CZ', 'CY', 'CH')
        self._para_two_bit_gate_name = ('CRx', 'CRy', 'CRz')
        self.qubit_num = qubit_num
        self.circuit_depth = circuit_depth
        self.circuit_dict = {}
        self.bit_occupied = {}
        self.controlled_link_occupied = {}
        self.reset_circuit()

    def _update_net(self,
                    operate_bits: List[int],
                    control_bits: List,
                    depths: List[int],
                    gate_names: List,
                    gate_params: List):
        for idx in range(len(depths)):
            depth = depths[idx]
            op_bit = operate_bits[idx]
            c_bit = control_bits[idx]
            if op_bit in self.bit_occupied[depth]:
                raise ValueError('failed to add operation bit at depth: {}, qu-bit: {}'.format(depth, op_bit))
            elif c_bit in self.bit_occupied[depth]:
                raise ValueError('failed to add control bit at depth: {}, qu-bit: {}'.format(depth, c_bit))
            else:
                if control_bits[idx] == operate_bits[idx]:
                    raise ValueError('control bit can not be the same as operate bit.')
                if op_bit >= self.qubit_num:
                    raise ValueError('operation bit {} out of range'.format(op_bit))
                self.circuit_dict[depth][op_bit] = {'name': gate_names[idx],
                                                    'param': gate_params[idx],
                                                    'control_bit': c_bit}

                self.bit_occupied[depth].append(op_bit)
                if control_bits[idx] is not None:
                    if c_bit >= self.qubit_num:
                        raise ValueError('control bit {} out of range'.format(c_bit))
                    self.bit_occupied[depth].append(c_bit)
                    if self._check_control_link(op_bit=op_bit, c_bit=c_bit, depth=depth):
                        self.controlled_link_occupied[depth].append((op_bit, c_bit))
        return

    def _check_control_link(self, op_bit, c_bit, depth):
        check = True
        link_len = abs(op_bit - c_bit)
        for each_link in self.controlled_link_occupied[depth]:
            each_len = abs(each_link[0] - each_link[1])
            if abs(max(each_link) - min(op_bit, c_bit)) < each_len + link_len and abs(min(each_link) - max(op_bit, c_bit)) < each_len + link_len:
                check = False
                raise ValueError('Sorry, do not support overlapped control link at same depth now.\t'
                                 'at depth {}\t'
                                 'occupied gate: operation bit: {}, control bit: {};\t'
                                 'conflicting gate: operation bit: {}, control bit: {}.'.format(depth,
                                                                                                each_link[0],
                                                                                                each_link[1],
                                                                                                op_bit, c_bit))
        return check

    def reset_circuit(self):
        self.circuit_dict = {i: {} for i in range(self.circuit_depth)}
        self.bit_occupied = {i: [] for i in range(self.circuit_depth)}
        self.controlled_link_occupied = {i: [] for i in range(self.circuit_depth)}
        return

    def show(self):
        gate_len = 9
        line_str = [["-" * gate_len for i in range(self.circuit_depth)] for j in range(self.qubit_num)]
        space_str = [[" " * gate_len for i in range(self.circuit_depth)] for j in range(self.qubit_num)]
        control_str = 'o'

        for depth in range(self.circuit_depth):
            for bit in range(self.qubit_num):
                if bit in self.circuit_dict[depth].keys():
                    gate = self.circuit_dict[depth][bit]['name']
                    c_bit = self.circuit_dict[depth][bit]['control_bit']
                    line = "-" * ((gate_len - len(gate)) // 2)
                    line_str[bit][depth] = "{}{}{}".format(line, gate, line)
                    if c_bit is not None:
                        line = "-" * ((gate_len - len(control_str)) // 2)
                        line_str[c_bit][depth] = "{}{}{}".format(line, control_str, line)
                        for link in range(min(c_bit, bit), max(c_bit, bit)):
                            line = " " * ((gate_len - 1) // 2)
                            space_str[link][depth] = "{}{}{}".format(line, '|', line)

        c_f = open('circuit_schematic.txt', 'w+')
        print("\n----------------------------- Circuit schematic -----------------------------\n")
        c_f.write("\n----------------------------- Circuit schematic -----------------------------\n")
        for i in range(len(line_str)):
            each_line = line_str[i]
            each_space = space_str[i]
            string = ""
            space = ""
            for j in range(len(each_line)):
                string += each_line[j]
                space += each_space[j]
            c_f.write(string + "\n")
            c_f.write(space + "\n")
            c_f.write(space + "\n")
            print(string)
            print(space)
            print(space)
        c_f.close()
        return

    @staticmethod
    def _check_input(operate_bit_idx: Union[int, List[int]],
                     depth: Union[int, List[int]],
                     control_bit_idx: Union[int, List[int]] = None):
        if isinstance(operate_bit_idx, int):
            operate_bit_idx = [operate_bit_idx]
        if isinstance(depth, int):
            depth = [depth]
        if len(operate_bit_idx) != len(depth):
            raise ValueError('lengths of config list do not match. one is {}, another {}'.format(len(operate_bit_idx),
                                                                                                 len(depth)))
        if control_bit_idx is not None:
            if isinstance(control_bit_idx, int):
                control_bit_idx = [control_bit_idx]
        else:
            control_bit_idx = [None] * len(operate_bit_idx)

        if len(control_bit_idx) != len(depth):
            raise ValueError('lengths of config list do not match.')

        return operate_bit_idx, depth, control_bit_idx

    def add_1_bit_gate(self,
                       gate_name: str,
                       bit_idx: Union[int, List[int]],
                       depth: Union[int, List[int]]):
        r"""
        add single bit quantum gate to circuit net list.

        :param gate_name: 'X', 'Y', 'Z', 'H'
        :param bit_idx: operating qu-bit index of gate
        :param depth: depth index of gate
        :return:
        """
        if gate_name not in self._single_bit_gate_name:
            raise ValueError('unexpected gate. support: {}'.format(self._single_bit_gate_name))
        operate_bit_idx, depth, control_bit_idx = self._check_input(bit_idx, depth)
        self._update_net(operate_bits=operate_bit_idx, control_bits=control_bit_idx, depths=depth,
                         gate_names=[gate_name] * len(depth), gate_params=[None] * len(depth))
        return

    def add_parameterized_1_bit_gate(self,
                                     gate_name: str,
                                     bit_idx: Union[int, List[int]],
                                     depth: Union[int, List[int]],
                                     param: Union[float, List[float]] = None):
        r"""
        add parameterized single bit quantum gate to circuit net list.

        :param gate_name: 'Rx', 'Ry', 'Rz'
        :param bit_idx: operating qu-bit index of gate
        :param depth: depth index of gate
        :param param: initial value of parameter (specifically, for R - gate: rotation angle)
        :return:
        """
        if gate_name not in self._para_single_bit_gate_name:
            raise ValueError('unexpected gate. support: {}'.format(self._para_single_bit_gate_name))

        operate_bit_idx, depth, control_bit_idx = self._check_input(bit_idx, depth)

        if param is None:
            param = list(np.random.uniform(self._angle_range[0], self._angle_range[1], len(depth)))
        else:
            if isinstance(param, float):
                param = [param]

        if len(param) != len(depth):
            raise ValueError('wrong params length.')
        self._update_net(operate_bits=operate_bit_idx, control_bits=control_bit_idx, depths=depth,
                         gate_names=[gate_name] * len(depth), gate_params=param)
        return

    def add_2_bit_gate(self,
                       gate_name: Union[str, List[str]],
                       operate_bit_idx: Union[int, List[int]],
                       control_bit_idx: Union[int, List[int]],
                       depth: Union[int, List[int]]):
        r"""
        add 2 bit controlled gate to circuit net list.

        :param gate_name: 'CX', 'CY', 'CZ', 'CH'
        :param control_bit_idx: control qu-bit
        :param operate_bit_idx: qu-bit under control
        :param depth: depth index of gate
        :return:
        """
        if gate_name not in self._two_bit_gate_name:
            raise ValueError('unexpected gate. support: {}'.format(self._two_bit_gate_name))

        operate_bit_idx, depth, control_bit_idx = self._check_input(operate_bit_idx, depth,
                                                                    control_bit_idx=control_bit_idx)
        self._update_net(operate_bits=operate_bit_idx, control_bits=control_bit_idx, depths=depth,
                         gate_names=[gate_name] * len(depth), gate_params=[None] * len(depth))
        return

    def add_parameterized_2_bit_gate(self,
                                     gate_name: Union[str, List[str]],
                                     operate_bit_idx: Union[int, List[int]],
                                     control_bit_idx: Union[int, List[int]],
                                     depth: Union[int, List[int]],
                                     param: Union[float, List[float]] = None):
        r"""
        add parameterized 2 bit controlled gate to circuit net list.

        :param gate_name: 'CRx', 'CRy', 'CRz'
        :param control_bit_idx: control qu-bit
        :param operate_bit_idx: qu-bit under control
        :param depth: depth index of gate
        :param param: initial value of parameter
        :return:
        """
        if gate_name not in self._para_two_bit_gate_name:
            raise ValueError('unexpected gate. support: {}'.format(self._para_two_bit_gate_name))

        operate_bit_idx, depth, control_bit_idx = self._check_input(operate_bit_idx, depth,
                                                                    control_bit_idx=control_bit_idx)

        if param is None:
            param = list(np.random.uniform(self._angle_range[0], self._angle_range[1], len(depth)))
        else:
            if isinstance(param, float):
                param = [param]

        if len(param) != len(depth):
            raise ValueError('params length do not match with other configs.')

        self._update_net(operate_bits=operate_bit_idx, control_bits=control_bit_idx, depths=depth,
                         gate_names=[gate_name] * len(depth), gate_params=param)
        return
