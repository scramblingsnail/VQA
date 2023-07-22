import mindspore as ms
from mindspore import nn, Parameter
from mindspore import value_and_grad
from src.qnn_mindspore import ConstantTensor, QGate, Observer
from src.qnn_mindspore import check_single_bit_gate, check_two_bit_gate, check_parameterized_two_bit_gate
from src.qnn_mindspore import check_cx_circuit


class TestKron(nn.Cell):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(ms.ops.ones(shape=(1,), dtype=ms.float32))

    def construct(self, inputs):
        ms_complex = ms.ops.Complex()
        real = ms.ops.eye(2, dtype=ms.float32) * ms.ops.cos(self.weight)
        imaginary = ms.Tensor([[0, -1.],
                               [-1., 0.]], dtype=ms.float32) * ms.ops.sin(self.weight)
        matrix = ms_complex(real, imaginary)
        output = ms.ops.mm(inputs, matrix)
        return output


def test_grad():
    def test_func(input_tensor):
        empty = ms_complex(ms.ops.ones((1,), dtype=ms.float32), ms.ops.zeros((1,), dtype=ms.float32))
        # input_tensor = ms.ops.kron(empty, input_tensor)
        new_real = ms.ops.kron(empty.real(), input_tensor.real())
        new_imag = ms.ops.kron(empty.real(), input_tensor.imag())
        new_tensor = ms_complex(new_real, new_imag)
        print(input_tensor)
        output = ms.ops.mm(new_tensor, new_tensor).sum()
        # output = ms.ops.mm(input_tensor, input_tensor).real().sum()
        # output = ms.ops.matmul(new_tensor.unsqueeze(0), new_tensor.unsqueeze(0)).sum()
        return output

    ms_complex = ms.ops.Complex()
    real = ms.Tensor([[1, 2], [2, 1]], dtype=ms.float32)
    imag = ms.Tensor([[-1, -3], [1, -1]], dtype=ms.float32)
    a = ms_complex(real, imag)

    out = test_func(a)
    print(out)
    grad_func = ms.grad(test_func, grad_position=0)
    grad = grad_func(a)
    print(grad)


def define_complex():
    r = ms.Tensor([1, 2], dtype=ms.float32)
    i = ms.Tensor([-1, -3], dtype=ms.float32)
    ms_complex = ms.ops.Complex()
    a = ms_complex(r, i)
    b = ms.ops.kron(a, a)
    print(a.real())
    print(a * a)
    print(b)


def multiply_complex():
    real = ms.Tensor([1, 2], dtype=ms.float32)
    real = real.unsqueeze(1)
    imag = ms.Tensor([-1, -3], dtype=ms.float32)
    imag = imag.unsqueeze(0)
    print(imag.shape)
    a = ms.ops.mm(real, imag)
    print(a)


def test_kron():
    ms_complex = ms.ops.Complex()
    real = ms.Tensor([1, 2], dtype=ms.float32)
    imag = ms.Tensor([-1, -3], dtype=ms.float32)
    input_tensor = ms_complex(real, imag)

    # input_tensor = ms.Tensor([-1, -3], dtype=ms.float32)

    empty = ms_complex(ms.ops.ones((1,), dtype=ms.float32), ms.ops.zeros((1,), dtype=ms.float32))
    input_tensor = ms.ops.kron(empty, input_tensor)
    print(input_tensor)


def test_constant_py():
    a = ConstantTensor()
    print('eye:', a.eye())
    print('empty:', a.empty())
    print('projection_0:', a.projection_0())
    print('projection_1:', a.projection_1())
    print('state_0:', a.state_0())
    print('state_1:', a.state_1())
    c = ms.Tensor([1, 2], dtype=ms.float32)
    b = c / ms.Tensor([2], dtype=ms.float32)
    print(type(b))


def test_gate():
    g = QGate()
    g.check()


def test_observe():
    o = Observer(qubit_num=1, observe_bit=0, observe_gate='Z')
    print('X observe: ', o.X_observe())
    print('Y observe: ', o.Y_observe())
    print('Z observe: ', o.Z_observe())


class Test(nn.Cell):
    def __init__(self):
        super().__init__()
        a = Parameter(default_input=ms.Tensor([1], dtype=ms.float32), name='test')
        self.insert_param_to_cell(param=a, param_name='test')
        a = a.unsqueeze(1)
        print(a[0].shape)
        self.test.set_data(a[0])
        print(self.test.data)
        self.complex = ms.ops.Complex()

    def limited_kron(self, complex_tensor, real_tensor, real_first: bool):
        if real_first:
            new_real = ms.ops.kron(real_tensor, complex_tensor.real())
            new_imag = ms.ops.kron(real_tensor, complex_tensor.imag())
        else:
            new_real = ms.ops.kron(complex_tensor.real(), real_tensor)
            new_imag = ms.ops.kron(complex_tensor.imag(), real_tensor)
        new_tensor = self.complex(new_real, new_imag)
        return new_tensor


# check_single_bit_gate()

# check_two_bit_gate()

# check_parameterized_two_bit_gate()

print(len("abc".split('d')))

# check_cx_circuit()

# TODO: 解决 single_step 内的梯度问题。

# ms_complex = ms.ops.Complex()
# real = ms.Tensor([[1, 2], [2, 1]], dtype=ms.float32)
# imag = ms.Tensor([[-1, 1], [1, -1]], dtype=ms.float32)
# input_tensor = ms_complex(real, imag)
#
# b = input_tensor.unsqueeze(0).broadcast_to((3, 2, 2))
# c = input_tensor.unsqueeze(0).broadcast_to((3, 2, 2))
#
# d = ms.ops.matmul(b, c)
# print(d.shape)







