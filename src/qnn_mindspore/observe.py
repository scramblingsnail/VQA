import mindspore as ms
from .constant import ConstantTensor
from typing import List, Union


class Observer:
    def __init__(self,
                 qubit_num: int,
                 observe_bit: Union[List[int], int],
                 observe_gate: Union[List[str], str]):
        self.qubit_num = qubit_num
        if isinstance(observe_bit, int):
            observe_bit = [observe_bit]
        if isinstance(observe_gate, str):
            observe_gate = [observe_gate]
        self.observe_bit = observe_bit
        self.observe_gate = observe_gate
        if len(self.observe_bit) != len(self.observe_gate):
            raise ValueError('observe bit and gate do not match.')
        self.cons = ConstantTensor(real_mode=False)
        self.complex = ms.ops.Complex()

    def get_observe_tensor(self):
        r"""
        :return: observe bit num * (2 ** qubit_num) * (2 ** qubit_num)
        """
        projection_tensor_0, projection_tensor_1 = [], []
        for ob_idx in range(len(self.observe_bit)):
            gate_name = self.observe_gate[ob_idx]
            ob_bit = self.observe_bit[ob_idx]

            if gate_name == 'X':
                pro_0, pro_1 = self.X_observe()
            elif gate_name == 'Y':
                pro_0, pro_1 = self.Y_observe()
            elif gate_name == 'Z':
                pro_0, pro_1 = self.Z_observe()
            else:
                raise ValueError("Sorry, only support 'X', 'Y', 'Z' now.")

            projection_0 = self.cons.empty()
            projection_1 = self.cons.empty()
            eye = self.cons.eye()
            for bit in range(self.qubit_num):
                if bit == ob_bit:
                    projection_0 = ms.ops.kron(projection_0, pro_0)
                    projection_1 = ms.ops.kron(projection_1, pro_1)
                else:
                    projection_0 = ms.ops.kron(projection_0, eye)
                    projection_1 = ms.ops.kron(projection_1, eye)
            projection_tensor_0.append(projection_0)
            projection_tensor_1.append(projection_1)
        projection_tensor_0 = ms.ops.stack(projection_tensor_0, axis=0)
        projection_tensor_1 = ms.ops.stack(projection_tensor_1, axis=0)
        return projection_tensor_0, projection_tensor_1

    def X_observe(self):
        projection_0 = self.complex(ms.Tensor([[0.5, 0.5],
                                               [0.5, 0.5]], dtype=ms.float32),
                                    ms.ops.zeros((2, 2), dtype=ms.float32))
        projection_1 = self.complex(ms.Tensor([[0.5, -0.5],
                                               [-0.5, 0.5]], dtype=ms.float32),
                                    ms.ops.zeros((2, 2), dtype=ms.float32))
        return projection_0, projection_1

    def Y_observe(self):
        projection_0 = self.complex(ms.Tensor([[0.5, 0.],
                                               [0., -0.5]], dtype=ms.float32),
                                    ms.Tensor([[0., 0.5],
                                               [0.5, 0.]], dtype=ms.float32))
        projection_1 = self.complex(ms.Tensor([[0.5, 0.],
                                               [0., 0.5]], dtype=ms.float32),
                                    ms.Tensor([[0., -0.5],
                                               [-0.5, 0.]], dtype=ms.float32))
        return projection_0, projection_1

    def Z_observe(self):
        projection_0 = self.complex(ms.Tensor([[1., 0.],
                                               [0., 0.]], dtype=ms.float32),
                                    ms.ops.zeros((2, 2), dtype=ms.float32))
        projection_1 = self.complex(ms.Tensor([[0., 0.],
                                               [0., 1.]], dtype=ms.float32),
                                    ms.ops.zeros((2, 2), dtype=ms.float32))
        return projection_0, projection_1

