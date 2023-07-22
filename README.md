# VQA

(This is A simple framework for Variational Quantum Algorithm)

### 昇腾AI创新大赛2023-昇思赛道-算法创新赛题

第一批赛题 赛题八：使用MinsSpore机器学习框架实现变分量子线路模拟


# Overview

* ### Backends:

  * Torch version
  * MindSpore version
* ### Check framework:
  > cd VQA
  > 
  > Torch version:
  > run_check_framework_torch.sh
  > 
  > MindSpore Version:
  > run_check_framework_mindspore.sh
* ### Run demo:
  > Torch version: 
  > run_train_torch.sh
  > 
  > MindSpore version:
  > run_train_mindspore.sh

# Contents:
  * ### Quantum Circuit
    * Supported Quantum Gates now

      >  1-bit gates:  X,  Y,  Z,  H.
      > 
      >  parameterized 1-bit gates: Rx($\theta$), Ry($\theta$), Rz($\theta$).
      >
      >  2-bit (controlled) gates: CX, CY, CZ, CH.
      > 
      >   parameterized 2-bit gates: CRx($\theta$), CRy($\theta$), CRz($\theta$).
    * define a quantum circuit
      >``` 
      > # define a circuit of 4 qubits, 5 layers depth
      > c = QCircuit(qubit_num=4, circuit_depth=5)
      > ```
    * add 1-bit gate
      > ```
      > # add 1-bit quantum gate at bit 0, depth 0
      > c.add_1_bit_gate(gate_name='H', bit_idx=0, depth=0)
      > ```
    * add parameterized 1-bit gate
      > ```
      > # add parameterized 1-bit quantum gate at (bit, depth) equal: (1, 0), (2, 0), (3, 0); 
      > # corresponding parameters are (0.2, 0.3, 0.4)
      > c.add_parameterized_1_bit_gate(gate_name='Ry', bit_idx=[1, 2, 3], depth=[0, 0, 0], param=[0.2, 0.3, 0.4])
      > ```
    * add 2-bit gate
    * >```
      > # add 2-bit gate at bit 0, depth 1, controlled by bit 3
      > c.add_2_bit_gate(gate_name='CX', operate_bit_idx=0, control_bit_idx=3, depth=1)
      >```
    * add parameterized 2-bit gate
      >```
      > # add parameterized 2-bit quantum gate at (bit, depth) equal: (1, 4), (2, 3), (3, 2);
      > # controlled by bit (0, 1, 2) and parameterized (0.2, 0.3, 0.4) individually.
      > c.add_parameterized_2_bit_gate(gate_name='CRx', operate_bit_idx=[1, 2, 3], control_bit_idx=[0, 1, 2], depth=[4, 3, 2], param=[0.2, 0.3, 0.4])
      >```
    * circuit visualization
      >```
      > c.show()
      >
      > ----------------------------- Circuit schematic -----------------------------
      > ----H-------CX-------------------------o----
      >              |                          |    
      >              |                          |    
      > ---Ry-------------------------o-------CRx---
      >              |                 |          
      >              |                 |       
      > ---Ry----------------o-------CRx------------
      >              |        |   
      >              |        |  
      > ---Ry-------o-------CRx---------------------
      >```
    * #### Note
      #### topological constraints of circuit definition
      * Only one gate can be defined at a certain position (bit, depth).
      * If a position is defined as the control bit of a controlled gate, it can not be defined as another gate's position.
      * At the same depth, Controlled gates should not overlap with each other.
      * (Temporary), for MindSpore Version:
        * only one gate can be defined at a certain depth.
  * ### Quantum network
    * #### Observer
      observe the quantum state by a specific quantum gate, the quantum state will collapse to one of the gate's eigenstates.
      > support X, Y, Z Now. 
    * #### Encoder Circuit
      A parameterized circuit is constructed for input encoding, the input data are mapped to the parameters of quantum gates.
    * #### Ansatze circuit
      A parameterized circuit with trainable parameters. Use gradient descent to optimize its parameters.
    * #### Define a quantum network
      > ```
      > # define a quantum network based on the predefined encoder circuit and ansatze circuit.
      > # observe bit 0 with quantum gate 'Z'.
      > n = QNet(encoder_circuit=encoder_circuit, circuit=ansatze_circuit, observe_bit=[0, ], observe_gate=['Z', ])
      > ```
    * #### Forward
        > ```
        > # torch version
        > # expectation_0: the possibility of state collapsing to eigenstate 0 of the observer gate.
        > # expectation_1: ~ of eigenstate 1.
        > expectation_0, expectation_1 = n.forward(inputs)
        > # mindspore version
        > expectation_0, expectation_1 = n(inputs)
        > ```
    
  * ### Some Simple Demonstrations
    * #### Classification
      * classify two class of points
    * #### Curve Fitting
      * fit a specific curve

