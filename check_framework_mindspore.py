from src.qnn_mindspore import check_single_bit_gate, check_two_bit_gate, check_parameterized_two_bit_gate, check_cx_circuit


if __name__ == '__main__':
    check_single_bit_gate()
    check_two_bit_gate()
    check_parameterized_two_bit_gate()
    check_cx_circuit()
