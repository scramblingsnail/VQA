import torch
import numpy as np
from .qcircuit import QCircuit


r""" configurations """


class Config:
    def __init__(self):
        r""" for torch env """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        r""" for mindspore env """
        self.device_target = 'GPU'      # 'CPU' 'Ascend'
        self.context_mode = 'pynative'
        self.device_id = 0

        r""" circuit definition """
        self.encoder_circuit, self.ansatze_circuit = self.circuit_definition()

        r""" param init range """
        self.init_range = (- np.pi / 8, np.pi / 8)

        r""" ---------- for classification demo ----------"""
        r""" observer """
        self.classification_observe_bit = [0, 1]
        self.classification_observe_gate = ['Z', 'Z']
        r""" dataset """
        self.classification_train_data_num = 40
        self.classification_test_data_num = 12
        r""" input encoding """
        self.classification_input_scaling = np.pi
        r""" training """
        self.classification_mini_batch_num = 10
        self.classification_epoch_num = 100
        self.classification_evaluate_interval = 1
        self.classification_lr = 0.03
        r""" optimizer """
        self.classification_momentum = 0.9
        self.classification_weight_decay = 1e-4
        r""" visualization """
        self.classification_save_interval = 1

        r""" ---------- for fitting demo ---------- """
        self.fitting_observe_bit = [0, ]
        self.fitting_observe_gate = ['Z', ]
        r""" dataset """
        self.fitting_train_data_step = 0.1
        self.fitting_test_data_step = 0.01
        r""" input encoding """
        self.fitting_input_scaling = np.pi / 4
        r""" training """
        self.fitting_mini_batch_num = 10
        self.fitting_epoch_num = 201
        self.fitting_evaluate_interval = 50
        self.fitting_lr = 0.03
        r""" optimizer """
        self.fitting_momentum = 0.9
        self.fitting_weight_decay = 1e-4
        r""" visualization """
        self.fitting_save_interval = 50

    @staticmethod
    def circuit_definition():
        r""" circuit definition """
        qubit_num = 4
        encoder_circuit_depth = 1
        ansatze_circuit_depth = 40

        encoder_circuit = QCircuit(qubit_num=qubit_num, circuit_depth=encoder_circuit_depth)
        ansatze_circuit = QCircuit(qubit_num=qubit_num, circuit_depth=ansatze_circuit_depth)

        encoder_circuit.add_parameterized_1_bit_gate(gate_name='Ry',
                                                     bit_idx=[0, 1, 2, 3],
                                                     depth=[0, 0, 0, 0])

        # ansatze_circuit.add_parameterized_1_bit_gate(gate_name='Ry',
        #                                              bit_idx=[0, 1, 2, 3] + [0, 1, 2, 3] + [0, 1, 2, 3] + [0, 1, 2, 3] +
        #                                                      [0, 1, 2, 3],
        #                                              depth=[4, 4, 4, 4] + [9, 9, 9, 9] + [14, 14, 14, 14] +
        #                                                    [19, 19, 19, 19] + [24, 24, 24, 24])
        #
        # ansatze_circuit.add_parameterized_2_bit_gate(gate_name='CRx',
        #                                              operate_bit_idx=[0, 3, 2, 1] + [2, 3, 0, 1] + [3, 2, 0, 1] +
        #                                                              [2, 1, 0, 3] + [0, 3, 2, 1],
        #                                              control_bit_idx=[3, 2, 1, 0] + [3, 0, 1, 2] + [2, 1, 3, 0] +
        #                                                              [3, 2, 1, 0] + [3, 2, 1, 0],
        #                                              depth=[0, 1, 2, 3] + [5, 6, 7, 8] + [10, 11, 12, 13] +
        #                                                    [15, 16, 17, 18] + [20, 21, 22, 23])

        ansatze_circuit.add_parameterized_1_bit_gate(gate_name='Ry',
                                                     bit_idx=[0, 1, 2, 3] + [0, 1, 2, 3] + [0, 1, 2, 3] +
                                                             [0, 1, 2, 3] + [0, 1, 2, 3],
                                                     depth=[4, 5, 6, 7] + [12, 13, 14, 15] + [20, 21, 22, 23] +
                                                           [28, 29, 30, 31] + [36, 37, 38, 39])

        ansatze_circuit.add_parameterized_2_bit_gate(gate_name='CRx',
                                                     operate_bit_idx=[0, 3, 2, 1] + [2, 3, 0, 1] + [3, 2, 0, 1] +
                                                                     [2, 1, 0, 3] + [0, 3, 2, 1],
                                                     control_bit_idx=[3, 2, 1, 0] + [3, 0, 1, 2] + [2, 1, 3, 0] +
                                                                     [3, 2, 1, 0] + [3, 2, 1, 0],
                                                     depth=[0, 1, 2, 3] + [8, 9, 10, 11] + [16, 17, 18, 19] +
                                                           [24, 25, 26, 27] + [32, 33, 34, 35])

        return encoder_circuit, ansatze_circuit
