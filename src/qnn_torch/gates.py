import torch
import numpy as np
from .constant import ConstantTensor
from typing import Union


class QGate:
    def __init__(self, device: torch.device):
        self.device = device
        self._gate_set = ('X', 'Y', 'Z', 'H', 'Rx', 'Ry', 'Rz')

    def gate_tensor(self, gate_name: str, param: Union[torch.tensor, torch.nn.Parameter] = None):
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
        cons = ConstantTensor(device=self.device)
        angle = torch.tensor([np.pi / delimiter], dtype=torch.float)
        state_0 = cons.state_0()
        state_1 = cons.state_1()
        for each_name in self._gate_set:
            each_tensor = self.gate_tensor(gate_name=each_name, param=angle)
            out_0 = torch.mm(each_tensor, state_0)
            out_1 = torch.mm(each_tensor, state_1)
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
        real = torch.tensor([[0., 1.],
                             [1., 0.]], dtype=torch.float, device=self.device)
        imaginary = torch.zeros((2, 2), dtype=torch.float, device=self.device)
        return torch.complex(real, imaginary)

    def Y_tensor(self):
        real = torch.zeros((2, 2), dtype=torch.float, device=self.device)
        imaginary = torch.tensor([[0., -1.],
                                  [1., 0.]], dtype=torch.float, device=self.device)
        return torch.complex(real, imaginary)

    def Z_tensor(self):
        real = torch.tensor([[1., 0.],
                             [0., -1.]], dtype=torch.float, device=self.device)
        imaginary = torch.zeros((2, 2), dtype=torch.float, device=self.device)
        return torch.complex(real, imaginary)

    def H_tensor(self):
        real = torch.tensor(np.array([[1., 1.],
                                      [1., -1.]]) / np.sqrt(2), dtype=torch.float, device=self.device)
        imaginary = torch.zeros((2, 2), dtype=torch.float, device=self.device)
        return torch.complex(real, imaginary)

    def Rx_tensor(self, param: torch.nn.Parameter):
        real = torch.eye(2, dtype=torch.float, device=self.device) * torch.cos(param / 2)
        imaginary = torch.tensor([[0, -1.],
                                  [-1., 0.]], dtype=torch.float, device=self.device) * torch.sin(param / 2)
        return torch.complex(real, imaginary)

    def Ry_tensor(self, param: torch.nn.Parameter):
        real = torch.eye(2, dtype=torch.float, device=self.device) * torch.cos(param / 2)
        real = real + torch.tensor([[0, -1.],
                                    [1., 0.]], dtype=torch.float, device=self.device) * torch.sin(param / 2)
        imaginary = torch.zeros((2, 2), dtype=torch.float, device=self.device)
        return torch.complex(real, imaginary)

    def Rz_tensor(self, param: torch.nn.Parameter):
        real = torch.eye(2, dtype=torch.float, device=self.device) * torch.cos(param / 2)
        imaginary = torch.tensor([[-1, 0.],
                                  [0., 1.]], dtype=torch.float, device=self.device) * torch.sin(param / 2)
        return torch.complex(real, imaginary)
