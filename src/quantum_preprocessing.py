import numpy as np
import pennylane as qml
from pennylane import numpy as np
import os
from multiprocessing import Pool
import pickle
import argparse

## Parameters of the script
parser = argparse.ArgumentParser(description="Run quantum processing on given images in format train/val/test.")
parser.add_argument("--data_folder", required=True, help="Input folder for data.")
parser.add_argument("--out_folder", required=True, help="Output folder for results.")

args = parser.parse_args()
data = args.data_folder
out = args.out_folder

# Creción de los circuitos
n_qubits = 12
graphs = {
    "closed-nn" : [[0,6], [6,9], [9,3], [3,0], [0,1], [6,7], [9,10], [3,4], [1,7], [7,10], [10,4], [4,1], [1,2], [7,8], [10,11], [4,5], [2,8], [8,11], [11,5], [5,2]],
    "open-nn": [[0,1],[1,2],[2,8],[8,7],[7,6],[6,9],[9,10],[10,11],[11,5],[5,4],[4,3]]
}

dev = qml.device('default.qubit', wires=n_qubits)

## Circuito cerrado
@qml.qnode(dev, interface="autograd")
def closed_circuit(params):
    """creates a quantum kernel based on a digital-analog quantum encoding.

    Args:
        params (torch.Parameters): trainable parameters

    Returns:
        list: expectation values of <Z> on each qubit
    """
    
    # DIGITAL ENCODING OF THE PIXELS
    for idx in range(n_qubits):
        qml.RY(params[idx], wires=idx)
        qml.Hadamard(wires=idx)
        
    # RYDBERG HAMILTONIAN EVOLUTION
    for t in [0.05, 0.1]:
        for qubit in range(n_qubits):
            qml.RX(t, wires=qubit)
            qml.RZ(t/2, wires=qubit)
        for pair in graphs["closed-nn"]:
            qml.CNOT(wires=[pair[0], pair[1]])
            qml.RZ(phi=np.pi/3, wires=pair[1])
            qml.CNOT(wires=[pair[0], pair[1]])
    
    # DIGITAL SINGLE-QUBIT OPERATIONS
    for qubit in range(n_qubits):
        qml.RY(np.pi/2, wires=qubit) 
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

## Circuito abierto
@qml.qnode(dev, interface="autograd")
def open_circuit(params):
    """creates a quantum kernel based on a digital-analog quantum encoding.

    Args:
        params (torch.Parameters): trainable parameters

    Returns:
        list: expectation values of <Z> on each qubit
    """
    
    # DIGITAL ENCODING OF THE PIXELS
    for idx in range(n_qubits):
        qml.RY(params[idx], wires=idx)
        qml.Hadamard(wires=idx)
        
    # RYDBERG HAMILTONIAN EVOLUTION
    for t in [0.05, 0.1]:
        for qubit in range(n_qubits):
            qml.RX(t, wires=qubit)
            qml.RZ(t/2, wires=qubit)
        for pair in graphs["open-nn"]:
            qml.CNOT(wires=[pair[0], pair[1]])
            qml.RZ(phi=np.pi/3, wires=pair[1])
            qml.CNOT(wires=[pair[0], pair[1]])
    
    # DIGITAL SINGLE-QUBIT OPERATIONS
    for qubit in range(n_qubits):
        qml.RY(np.pi/2, wires=qubit) 
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

## Definimos la convolución
array_shape = (64, 64)
def quanv(idx, image, folder):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = np.zeros((array_shape[0]-1, array_shape[1]-1, 24))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, array_shape[0]-1, 1):
        for k in range(0, array_shape[1]-1, 1):
            a0=image[j, k, 0]
            a1=image[j, k, 1]
            a2=image[j, k, 2]
            
            b0=image[j, k + 1, 0]
            b1=image[j, k + 1, 1]
            b2=image[j, k + 1, 2]
            
            c0=image[j + 1, k, 0]
            c1=image[j + 1, k, 1]
            c2=image[j + 1, k, 2]
            
            d0=image[j + 1, k + 1, 0]
            d1=image[j + 1, k + 1, 1]
            d2=image[j + 1, k + 1, 2]

            q1_results = closed_circuit( # circuit 2
                [
                    a0,
                    a1,
                    a2,
                    b0,
                    b1,
                    b2,
                    c0,
                    c1,
                    c2,
                    d0,
                    d1,
                    d2
                ]
            )
            q2_results = open_circuit( # circuit 3
                [
                    a0,
                    a1,
                    a2,
                    b0,
                    b1,
                    b2,
                    c0,
                    c1,
                    c2,
                    d0,
                    d1,
                    d2
                ]
            )

            for c in range(12):
                out[j, k, c] = q1_results[c]
            for c in range(12): 
                out[j, k, c+12] = q2_results[c]

    if folder=="train":
        with open(f"{out}/train/train_{idx}.pkl", "wb") as file:
            pickle.dump(out, file)
    elif folder=="validation":
        with open(f"{out}/validation/val_{idx}.pkl", "wb") as file:
            pickle.dump(out, file)
    elif folder=="test":
        with open(f"{out}/test/test_{idx}.pkl", "wb") as file:
            pickle.dump(out, file)

# Ejecutamos el preprocesado

num_cores = os.cpu_count() - 2 # Elegir número de cores

# pool_tr = Pool(processes=num_cores)
# pool_val = Pool(processes=num_cores)
pool_test = Pool(processes=num_cores)

# # PROCCESS TRAIN DATA SET
# print("Quantum pre-processing of train images")
# with open(f"{data}/train_imgs_tf.pkl", "rb") as f:
#     train_imgs_tf = pickle.load(f)

# for idx, img in enumerate(train_imgs_tf):
#     pool_tr.apply_async(quanv, args=(idx,img,"train"))
# pool_tr.close()
# pool_tr.join()

# # PROCCESS VALIDATION DATA SET
# print("Quantum pre-processing of validation images")
# with open(f"{data}/val_imgs_tf.pkl", "rb") as f:
#     val_imgs_tf = pickle.load(f)

# for idx, img in enumerate(val_imgs_tf):
#     pool_val.apply_async(quanv, args=(idx,img,"validation"))
# pool_val.close()
# pool_val.join()

# PROCCESS TEST DATA SET
print("Quantum pre-processing of test images")
with open(f"{data}/test_imgs_tf.pkl", "rb") as f:
    test_imgs_tf = pickle.load(f)

for idx, img in enumerate(test_imgs_tf):
    pool_test.apply_async(quanv, args=(idx,img,"test"))
pool_test.close()
pool_test.join()