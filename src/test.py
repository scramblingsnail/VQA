# from pyvqnet.nn.module import Module
# from pyvqnet.optim.sgd import SGD
# from pyvqnet.nn.loss import CategoricalCrossEntropy
# from pyvqnet.tensor.tensor import QTensor
# import pyqpanda as pq
# from pyvqnet.qnn.quantumlayer import QuantumLayer
# import tensorcircuit as tc

import torch


a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 1], [1, 1]])
print(torch.kron(a, b))

# c = tc.Circuit(3)
# c.H(1)
# c.CNOT(0, 1)
# c.RX(2, theta=tc.num_to_tensor(1.))
# r = c.expectation([tc.gates.z(), (2, )])
# print(r)
