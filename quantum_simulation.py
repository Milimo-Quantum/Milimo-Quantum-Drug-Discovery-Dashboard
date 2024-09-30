# quantum_simulation.py

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA, COBYLA, ADAM
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer.noise import NoiseModel
from qiskit.primitives import Estimator
from qiskit.converters import circuit_to_gate
from qiskit import transpile
from qiskit.exceptions import QiskitError
from qiskit.circuit import Parameter
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import io
import MDAnalysis as mda

logger = logging.getLogger(__name__)

def create_custom_circuit(num_qubits, depth):
    circuit = QuantumCircuit(num_qubits)
    params = []
    for d in range(depth):
        for q in range(num_qubits):
            theta = Parameter(f'θ_{d}_{q}')
            params.append(theta)
            circuit.ry(theta, q)
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)
    return circuit, params

def run_vqe(num_qubits, depth, optimization_method, hamiltonian):
    try:
        H = hamiltonian

        # Create the custom circuit
        ansatz, params = create_custom_circuit(num_qubits, depth)
        
        # Choose the optimizer
        if optimization_method == "spsa":
            optimizer = SPSA(maxiter=100)
        elif optimization_method == "cobyla":
            optimizer = COBYLA(maxiter=100)
        else:  # adam
            optimizer = ADAM(maxiter=100)

        # Create the estimator
        estimator = Estimator()

        # Create the VQE instance
        vqe = VQE(estimator, ansatz, optimizer)

        # Run VQE
        result = vqe.compute_minimum_eigenvalue(H)

        # Get the optimized parameters
        optimized_params = result.optimal_point

        # Create the optimized circuit with bound parameters
        optimized_circuit = ansatz.assign_parameters({param: value for param, value in zip(params, optimized_params)})

        # Create a measured circuit by adding measurements
        measured_circuit = optimized_circuit.copy()
        measured_circuit.measure_all()

        # Calculate expectation values for each term in the Hamiltonian
        expectation_values = []
        for term in H.paulis:
            expectation = estimator.run(optimized_circuit, term).result().values[0]
            expectation_values.append(expectation)

        return result, expectation_values, optimized_params, optimized_circuit, measured_circuit
    except Exception as e:
        logger.error(f"Error in run_vqe: {str(e)}", exc_info=True)
        raise

def apply_error_mitigation(circuit, error_mitigation):
    try:
        backend = AerSimulator()
        if error_mitigation == "readout":
            # Create a basic noise model
            noise_model = NoiseModel()
            # Add readout error to the noise model (you may need to adjust these probabilities)
            for qubit in range(circuit.num_qubits):
                noise_model.add_readout_error([[0.98, 0.02], [0.02, 0.98]], [qubit])
            backend = AerSimulator(noise_model=noise_model)
        
        # Transpile the circuit for the backend
        transpiled_circuit = transpile(circuit, backend)
        return backend, transpiled_circuit
    except Exception as e:
        logger.error(f"Error in apply_error_mitigation: {str(e)}", exc_info=True)
        raise

def generate_hamiltonian_from_file(file_path, num_qubits):
    try:
        universe = mda.Universe(file_path)
        atoms = universe.atoms

        if len(atoms) == 0:
            raise ValueError("No atoms found in the molecular structure.")

        # Simplified example: use atomic numbers as features
        masses = atoms.masses
        if len(masses) == 0 or max(masses) == 0:
            raise ValueError("Atom masses are not available or zero.")

        features = masses / max(masses)

        # Pad or truncate features to match num_qubits
        features = list(features[:num_qubits]) + [0.0] * max(0, num_qubits - len(features))

        # Create a simple Hamiltonian based on these features
        H = SparsePauliOp.from_list([
            ("Z" * i + "X" + "I" * (num_qubits - i - 1), feature)
            for i, feature in enumerate(features)
        ])

        return H
    except Exception as e:
        logger.error(f"Error in generate_hamiltonian_from_file: {str(e)}")
        raise

def run_quantum_simulation(file_path, num_qubits, depth, error_mitigation, optimization_method):
    try:
        # Generate Hamiltonian from file
        hamiltonian = generate_hamiltonian_from_file(file_path, num_qubits)

        # Run VQE
        start_time = time.time()
        vqe_result, expectation_values, optimized_params, optimized_circuit, measured_circuit = run_vqe(num_qubits, depth, optimization_method, hamiltonian)
        quantum_time = time.time() - start_time

        # Apply error mitigation and get transpiled circuit
        backend, transpiled_circuit = apply_error_mitigation(measured_circuit, error_mitigation)

        # Run the circuit
        job = backend.run(transpiled_circuit)
        result = job.result()
        
        try:
            counts = result.get_counts(transpiled_circuit)
        except QiskitError:
            logger.warning("No counts available for the circuit. This might be expected if no measurements were performed.")
            counts = {}

        return {
            "vqe_result": vqe_result,
            "expectation_values": expectation_values,
            "optimized_params": optimized_params,
            "optimized_circuit": optimized_circuit,
            "counts": counts,
            "quantum_time": quantum_time
        }
    except Exception as e:
        logger.error(f"Error in run_quantum_simulation: {str(e)}", exc_info=True)
        raise

def circuit_to_image(circuit):
    try:
        # Add measurements for visualization purposes
        measured_circuit = circuit.copy()
        measured_circuit.measure_all()

        # Set custom style options
        style = {
            'backgroundcolor': 'white',
            'textcolor': 'black',
            'subtextcolor': 'gray',
            'linecolor': 'black',
            'creglinecolor': 'black',
            'gatetextcolor': 'white',
            'gatefacecolor': '#800080',  # Purple color for gates
            'barrierfacecolor': '#CCCCCC',
            'barrieredgecolor': '#CCCCCC',
            'circuitfacecolor': 'white',
            'circuitedgecolor': 'black',
            'measure': {
                'color': 'black',
                'textcolor': 'black',
                'fontsize': 9,
            },
        }

        # Convert the circuit to an image
        fig, ax = plt.subplots(figsize=(12, 6))
        circuit_drawer(measured_circuit, output='mpl', style=style, ax=ax)
        
        # Remove axis labels and ticks
        ax.set_axis_off()
        
        # Adjust layout and save
        plt.tight_layout()
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='svg', bbox_inches='tight', pad_inches=0.1, facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        svg_image = img_buffer.getvalue().decode()
        plt.close(fig)

        return svg_image
    except Exception as e:
        logger.error(f"Error in circuit_to_image: {str(e)}", exc_info=True)
        return "Error generating circuit image"