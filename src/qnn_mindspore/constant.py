import mindspore as ms


class ConstantTensor:
    def __init__(self, real_mode: bool = True):
        self.complex = ms.ops.Complex()
        self.real_mode = real_mode

    def eye(self, real_mode: bool = None):
        r"""
        1   0
        0   1
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.ops.eye(2, dtype=ms.float32)
        else:
            return self.complex(ms.ops.eye(2, dtype=ms.float32),
                                ms.ops.zeros((2, 2), dtype=ms.float32))

    def empty(self, real_mode: bool = None):
        r"""
        1
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.ops.ones((1,), dtype=ms.float32)
        else:
            return self.complex(ms.ops.ones((1,), dtype=ms.float32), ms.ops.zeros((1,), dtype=ms.float32))

    def projection_0(self, real_mode: bool = None):
        r"""
        1   0
        0   0
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.Tensor([[1., 0.],
                              [0., 0.]], dtype=ms.float32)
        else:
            return self.complex(ms.Tensor([[1., 0.],
                                           [0., 0.]], dtype=ms.float32),
                                ms.ops.zeros((2, 2), dtype=ms.float32))

    def projection_1(self, real_mode: bool = None):
        r"""
        0   0
        0   1
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.Tensor([[0., 0.],
                              [0., 1.]], dtype=ms.float32)
        else:
            return self.complex(ms.Tensor([[0., 0.],
                                           [0., 1.]], dtype=ms.float32),
                                ms.ops.zeros((2, 2), dtype=ms.float32))

    def state_0(self, real_mode: bool = None):
        r"""
        1
        0
        size: (2, 1)
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.Tensor([1., 0.], dtype=ms.float32).unsqueeze(1)
        else:
            return self.complex(ms.Tensor([1., 0.], dtype=ms.float32),
                                ms.ops.zeros((2,), dtype=ms.float32)).unsqueeze(1)

    def state_1(self, real_mode: bool = None):
        r"""
        0
        1
        size: (2, 1)
        """
        if real_mode is None:
            real_mode = self.real_mode
        if real_mode:
            return ms.Tensor([0., 1.], dtype=ms.float32).unsqueeze(1)
        else:
            return self.complex(ms.Tensor([0., 1.], dtype=ms.float32),
                                ms.ops.zeros((2,), dtype=ms.float32)).unsqueeze(1)
