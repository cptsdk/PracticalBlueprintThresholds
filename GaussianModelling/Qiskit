import numpy as np                           
from qiskit import QuantumCircuit       
from qiskit.providers.aer import Aer  



#################### Functions for generating the Qiskit data ####################

def calculate_branch_fidelity(n, m, num_runs, prob_error):    
    # Construct the ideal encoded (aka "branched") cluster state circuit
    qca = QuantumCircuit(n + 1)
    qca.h(0)  # Apply Hadamard to the root qubit

    cnot_count = 0  # Counter for CNOTs 

    for i in range(1, n + 1):
        qca.cx(0, i)  # Apply CNOT from root to qubit i
        cnot_count += 1

        # Apply Hadamard after every m CNOTs, except after the last CNOT
        if cnot_count == m and i < n:
            qca.h(0)
            cnot_count = 0  # Reset the counter

    qca.save_statevector()

    # Simulate the ideal state
    backend = Aer.get_backend('aer_simulator')  
    job1 = backend.run(qca)
    result1 = job1.result()
    ideal_state = result1.get_statevector(qca)

    # Function to introduce errors based on m
    def construct_noisy_circuit():
        qcb = QuantumCircuit(n + 1)
        qcb.h(0)  # Apply Hadamard to the root qubit
        if np.random.binomial(1, prob_error):
                qcb.z(0)

        cnot_count = 0  # Counter for CNOTs 

        for i in range(1, n + 1):
            qcb.cx(0, i)  # Apply CNOT from root to qubit i
            cnot_count += 1

            # Introduce a Z error after CNOT with probability prob_error
            if np.random.binomial(1, prob_error):
                qcb.z(0)

            # Apply Hadamard after every m CNOTs, except after the last CNOT
            if cnot_count == m and i < n:
                qcb.h(0)
                # Introduce a Z error after Hadamard with probability prob_error
                if np.random.binomial(1, prob_error):
                    qcb.z(0)
                cnot_count = 0  # Reset the counter

        qcb.save_statevector()
        return qcb

    # Run simulations to calculate infidelities
    infidelities = []
    for _ in range(num_runs):
        noisy_circuit = construct_noisy_circuit()
        job2 = backend.run(noisy_circuit)
        result2 = job2.result()
        error_state = result2.get_statevector(noisy_circuit)

        # Direct infidelity calculation for pure states
        infidelity = 1 - abs(np.vdot(ideal_state, error_state))**2
        infidelities.append(infidelity)

    # Calculate mean infidelity and standard deviation
    mean_infidelity = np.mean(infidelities)
    std_infidelity = np.std(infidelities) / np.sqrt(num_runs)

    return mean_infidelity, std_infidelity
