import torch


class ConstantTensor:
    def __init__(self, device: torch.device):
        self.device = device


    def eye(self):
        r"""
        1   0
        0   1
        """
        return torch.complex(torch.eye(2, dtype=torch.float, device=self.device),
                             torch.zeros((2, 2), dtype=torch.float, device=self.device))

    def empty(self):
        r"""
        1
        """
        return torch.ones((1,), dtype=torch.float, device=self.device)

    def projection_0(self):
        r"""
        1   0
        0   0
        """
        return torch.complex(torch.tensor([[1., 0.],
                                           [0., 0.]], dtype=torch.float, device=self.device),
                             torch.zeros((2, 2), dtype=torch.float, device=self.device))

    def projection_1(self):
        r"""
        0   0
        0   1
        """
        return torch.complex(torch.tensor([[0., 0.],
                                           [0., 1.]], dtype=torch.float, device=self.device),
                             torch.zeros((2, 2), dtype=torch.float, device=self.device))

    def state_0(self):
        r"""
        1
        0
        size: (2, 1)
        """
        return torch.complex(torch.tensor([1., 0.], dtype=torch.float, device=self.device),
                             torch.zeros((2,), dtype=torch.float, device=self.device)).unsqueeze(1)

    def state_1(self):
        r"""
        0
        1
        size: (2, 1)
        """
        return torch.complex(torch.tensor([0., 1.], dtype=torch.float, device=self.device),
                             torch.zeros((2,), dtype=torch.float, device=self.device)).unsqueeze(1)
