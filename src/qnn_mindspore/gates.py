import mindspore as ms
from mindspore import nn, Parameter
import numpy as np
from .constant import ConstantTensor
from typing import Union


class QGate:
    def __init__(self):
        self._gate_set = ('X', 'Y', 'Z', 'H', 'Rx', 'Ry', 'Rz')
        self.complex = ms.ops.Complex()

    def gate_tensor(self, gate_name: str, param: Union[ms.Tensor, Parameter] = None):
        if gate_name in ['Rx', 'Ry', 'Rz'] and param is None:
            raise ValueError('param required for {}'.format(gate_name))
        if gate_name == 'X' or gate_name == 'CX':
            return self.X_tensor()
        elif gate_name == 'Y' or gate_name == 'CY':
            return self.Y_tensor()
        elif gate_name == 'Z' or gate_name == 'CZ':
            return self.Z_tensor()
        elif gate_name == 'H' or gate_name == 'CH':
            return self.H_tensor()
        elif gate_name == 'Rx' or gate_name == 'CRx':
            return self.Rx_tensor(param=param)
        elif gate_name == 'Ry' or gate_name == 'CRy':
            return self.Ry_tensor(param=param)
        elif gate_name == 'Rz' or gate_name == 'CRz':
            return self.Rz_tensor(param=param)
        else:
            raise ValueError('Sorry, now only support gates: {}'.format(self._gate_set))

    def check(self):
        delimiter = 2
        cons = ConstantTensor(real_mode=False)
        angle = ms.Tensor([np.pi / delimiter], dtype=ms.float32)
        state_0 = cons.state_0()
        state_1 = cons.state_1()
        for each_name in self._gate_set:
            each_tensor = self.gate_tensor(gate_name=each_name, param=angle)
            out_0 = ms.ops.mm(each_tensor, state_0)
            out_1 = ms.ops.mm(each_tensor, state_1)
            print('\n---------- gate name: {} ----------'.format(each_name))
            if each_name in ('Rx', 'Ry', 'Rz'):
                print('rotation angle: {} pi'.format(1 / delimiter))
            print('tensor: \n', each_tensor)
            print('\ninput: |0>\n', state_0)
            print('output: \n', out_0)
            print('\ninput: |1>\n', state_1)
            print('output: \n', out_1)
        return

    def X_tensor(self):
        real = ms.Tensor([[0., 1.],
                          [1., 0.]], dtype=ms.float32)
        imaginary = ms.ops.zeros((2, 2), dtype=ms.float32)
        return self.complex(real, imaginary)

    def Y_tensor(self):
        real = ms.ops.zeros((2, 2), dtype=ms.float32)
        imaginary = ms.Tensor([[0., -1.],
                               [1., 0.]], dtype=ms.float32)
        return self.complex(real, imaginary)

    def Z_tensor(self):
        real = ms.Tensor([[1., 0.],
                          [0., -1.]], dtype=ms.float32)
        imaginary = ms.ops.zeros((2, 2), dtype=ms.float32)
        return self.complex(real, imaginary)

    def H_tensor(self):
        real = ms.Tensor(np.array([[1., 1.],
                                   [1., -1.]]) / np.sqrt(2), dtype=ms.float32)
        imaginary = ms.ops.zeros((2, 2), dtype=ms.float32)
        return self.complex(real, imaginary)

    def Rx_tensor(self, param: Parameter):
        real = ms.ops.eye(2, dtype=ms.float32) * ms.ops.cos(param / 2)
        imaginary = ms.Tensor([[0, -1.],
                               [-1., 0.]], dtype=ms.float32) * ms.ops.sin(param / 2)
        return self.complex(real, imaginary)

    def Ry_tensor(self, param: Parameter):
        real = ms.ops.eye(2, dtype=ms.float32) * ms.ops.cos(param / 2)
        real = real + ms.Tensor([[0, -1.],
                                 [1., 0.]], dtype=ms.float32) * ms.ops.sin(param / 2)
        imaginary = ms.ops.zeros((2, 2), dtype=ms.float32)
        return self.complex(real, imaginary)

    def Rz_tensor(self, param: Parameter):
        real = ms.ops.eye(2, dtype=ms.float32) * ms.ops.cos(param / 2)
        imaginary = ms.Tensor([[-1, 0.],
                               [0., 1.]], dtype=ms.float32) * ms.ops.sin(param / 2)
        return self.complex(real, imaginary)
