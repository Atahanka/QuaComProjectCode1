# Reproducing the Results
Final Project — Quantum Computation

This repository contains the code used to generate the results presented in the final project
“Analyzing the Impact of Noise and Error Mitigation on the Convergence of Variational Quantum Algorithms”.
All experiments were performed using Qiskit simulators with controlled noise models.

---

## Environment Setup

The code was tested with Python 3.10 or newer.

(Optional) Create and activate a virtual environment:

```bash
python -m venv .venv  
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pandas
```

No access to real quantum hardware is required.

---

## Running the Experiments

All experiments are controlled from a single script.

```bash
cd code  
python vqe_experiments.py
```

This script performs the following steps:
- Defines a two-qubit transverse-field Ising Hamiltonian
- Runs a VQE-style optimization loop
- Compares three execution regimes:
  - noiseless
  - noisy
  - noisy with zero-noise extrapolation (ZNE)
- Sweeps over:
  - shot budgets B ∈ {300, 900, 2700}
  - base error rates p₀ ∈ {0.001, 0.003, 0.01}
- Repeats each configuration for three random seeds

All numerical results are saved to:

results.csv

---

## Generating the Figures

After the experiments finish, generate the figures used in the report by running:

```bash
python plotting.py
```

This script reads results.csv and produces the following plots:
- Energy versus iteration
- Iterations to reach a fixed accuracy threshold versus shot budget
- Final energy error versus base error rate

The figures are saved as PDF files in the directory.

---

## Notes on Experimental Design

- A fixed shot budget per iteration is enforced for all execution regimes.
- For zero-noise extrapolation:
  - Noise scaling is implemented using gate folding
  - Three noise scales are used
  - The total shot budget is split evenly across noise scales
- The optimizer is intentionally simple to isolate the effects of noise and sampling rather than optimizer sophistication.
- All experiments are performed on simulators with parameterized noise models.

---

## Reproducibility

- All random seeds are fixed and explicitly recorded.
- The Hamiltonian, ansatz, noise model, and optimizer settings are defined directly in vqe_experiments.py.
- All plots can be regenerated directly from results.csv.
