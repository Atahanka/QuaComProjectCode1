import numpy as np
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# ==========================
#  Hamiltonian & utilities
# ==========================

def hamiltonian_eigenvalues() -> Tuple[float, np.ndarray]:
    Z = np.array([[1, 0], [0, -1]], dtype=float)
    I = np.eye(2, dtype=float)
    H = -1.0 * np.kron(Z, Z) - 0.7 * (np.kron(Z, I) + np.kron(I, Z))
    w, _ = np.linalg.eigh(H)
    return float(w[0]), w


def bitstring_energy(bitstring: str) -> float:
    assert len(bitstring) == 2
    z = []
    for b in bitstring:
        z_val = 1.0 if b == '0' else -1.0
        z.append(z_val)
    z0, z1 = z
    energy = -1.0 * (z0 * z1) - 0.7 * (z0 + z1)
    return energy


def energy_from_counts(counts: Dict[str, int]) -> float:
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    e_sum = 0.0
    for bitstring, c in counts.items():
        p = c / total_shots
        e_sum += p * bitstring_energy(bitstring)
    return e_sum


# ==========================
#  Ansatz & folding
# ==========================

@dataclass
class AnsatzConfig:
    num_qubits: int = 2
    reps: int = 2


def build_ansatz(config: AnsatzConfig) -> Tuple[QuantumCircuit, ParameterVector]:
    num_qubits = config.num_qubits
    reps = config.reps
    num_params_per_layer = num_qubits * 2  # two rotations per qubit (ry, rz)
    num_params = reps * num_params_per_layer

    params = ParameterVector("theta", num_params)
    qc = QuantumCircuit(num_qubits)

    p_idx = 0
    for r in range(reps):
        for q in range(num_qubits):
            qc.ry(params[p_idx], q)
            p_idx += 1
            qc.rz(params[p_idx], q)
            p_idx += 1

        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

    return qc, params


def fold_circuit_global(base: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    if scale_factor < 1 or scale_factor % 2 != 1:
        raise ValueError("scale_factor must be an odd integer >= 1")

    k = (scale_factor - 1) // 2
    folded = QuantumCircuit(base.num_qubits)
    folded.compose(base, inplace=True)
    for _ in range(k):
        folded.compose(base.inverse(), inplace=True)
        folded.compose(base, inplace=True)
    return folded


# ==========================
#  Noise model
# ==========================

def build_noise_model(p0: float) -> NoiseModel:
    noise_model = NoiseModel()

    p1 = min(p0, 1.0)
    p2 = min(10 * p0, 1.0)
    pr = min(2 * p0, 0.5)

    dep1 = depolarizing_error(p1, 1)
    dep2 = depolarizing_error(p2, 2)
    readout = ReadoutError([[1 - pr, pr], [pr, 1 - pr]])

    one_q_gates = ['rx', 'ry', 'rz', 'u1', 'u2', 'u3']
    two_q_gates = ['cx']

    noise_model.add_all_qubit_quantum_error(dep1, one_q_gates)
    noise_model.add_all_qubit_quantum_error(dep2, two_q_gates)
    noise_model.add_all_qubit_readout_error(readout)

    return noise_model


# ==========================
#  Energy evaluation
# ==========================

def evaluate_energy_single(
    ansatz_base: QuantumCircuit,
    params: ParameterVector,
    theta: np.ndarray,
    shots: int,
    backend: AerSimulator,
    seed: Optional[int] = None,
) -> float:
    qc = ansatz_base.assign_parameters(
        {p: float(v) for p, v in zip(params, theta)},
        inplace=False
    )
    qc_meas = qc.copy()
    qc_meas.measure_all()

    job = backend.run(qc_meas, shots=shots, seed_simulator=seed)
    result = job.result()
    counts = result.get_counts()
    return energy_from_counts(counts)


def evaluate_energy_zne(
    ansatz_base: QuantumCircuit,
    params: ParameterVector,
    theta: np.ndarray,
    total_shots: int,
    backend: AerSimulator,
    noise_scales: List[int],
    seed: Optional[int] = None,
) -> float:
    n_scales = len(noise_scales)
    shots_per_scale = max(total_shots // n_scales, 1)

    energies = []
    scales_used = []

    for i, s in enumerate(noise_scales):
        folded = fold_circuit_global(ansatz_base, s)
        qc = folded.assign_parameters(
            {p: float(v) for p, v in zip(params, theta)},
            inplace=False
        )
        qc_meas = qc.copy()
        qc_meas.measure_all()

        seed_scale = None if seed is None else seed + i

        job = backend.run(qc_meas, shots=shots_per_scale, seed_simulator=seed_scale)
        result = job.result()
        counts = result.get_counts()
        e = energy_from_counts(counts)
        energies.append(e)
        scales_used.append(s)

    scales_arr = np.array(scales_used, dtype=float)
    energies_arr = np.array(energies, dtype=float)
    coeffs = np.polyfit(scales_arr, energies_arr, deg=1)
    a, b = coeffs
    return float(b)


# ==========================
#  Simple VQE-style optimizer
# ==========================

@dataclass
class VQERunConfig:
    regime: str  # 'noiseless', 'noisy', 'zne'
    p0: float
    shot_budget: int
    num_iterations: int
    seed: int
    noise_scales: List[int]


def run_vqe(
    ansatz_base: QuantumCircuit,
    params: ParameterVector,
    config: VQERunConfig,
    exact_ground: float,
) -> List[Dict]:
    np.random.seed(config.seed)

    if config.regime == "noiseless":
        backend = AerSimulator()  # no noise model
    else:
        noise_model = build_noise_model(config.p0)
        backend = AerSimulator(noise_model=noise_model)

    num_params = len(params)

    theta_best = np.random.uniform(-np.pi, np.pi, size=(num_params,))
    if config.regime in ("noiseless", "noisy"):
        e_best = evaluate_energy_single(
            ansatz_base, params, theta_best, config.shot_budget, backend, seed=config.seed
        )
    elif config.regime == "zne":
        e_best = evaluate_energy_zne(
            ansatz_base, params, theta_best, config.shot_budget,
            backend, config.noise_scales, seed=config.seed
        )
    else:
        raise ValueError(f"Unknown regime {config.regime}")

    history = []

    history.append({
        "iteration": 0,
        "energy_best": e_best,
        "theta": theta_best.copy(),
        "regime": config.regime,
        "p0": config.p0,
        "shot_budget": config.shot_budget,
        "seed": config.seed,
        "exact_ground": exact_ground,
    })

    step_size0 = 0.3

    for it in range(1, config.num_iterations + 1):
        step_size = step_size0 / np.sqrt(1.0 + it)

        delta = np.random.normal(loc=0.0, scale=step_size, size=(num_params,))
        theta_candidate = theta_best + delta

        if config.regime in ("noiseless", "noisy"):
            e_candidate = evaluate_energy_single(
                ansatz_base, params, theta_candidate, config.shot_budget, backend,
                seed=config.seed + it
            )
        else:  # zne
            e_candidate = evaluate_energy_zne(
                ansatz_base, params, theta_candidate, config.shot_budget, backend,
                config.noise_scales, seed=config.seed + it
            )

        if e_candidate < e_best:
            e_best = e_candidate
            theta_best = theta_candidate

        history.append({
            "iteration": it,
            "energy_best": e_best,
            "theta": theta_best.copy(),
            "regime": config.regime,
            "p0": config.p0,
            "shot_budget": config.shot_budget,
            "seed": config.seed,
            "exact_ground": exact_ground,
        })

    return history


# ==========================
#  Main experiment driver
# ==========================

def main():
    regimes = ["noiseless", "noisy", "zne"]
    p0_list = [0.001, 0.003, 0.01]
    shot_budgets = [300, 900, 2700]
    seeds = [0, 1, 2]
    num_iterations = 60
    noise_scales = [1, 3, 5]

    exact_ground, spectrum = hamiltonian_eigenvalues()
    print("Exact ground energy:", exact_ground)
    print("Spectrum:", spectrum)

    ansatz_config = AnsatzConfig(num_qubits=2, reps=2)
    ansatz_base, params = build_ansatz(ansatz_config)

    out_file = "results.csv"
    fieldnames = [
        "regime", "p0", "shot_budget", "seed",
        "iteration", "energy_best", "exact_ground"
    ]

    with open(out_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for regime in regimes:
            for shot_budget in shot_budgets:
                for seed in seeds:
                    # For noiseless, p0 is irrelevant; store p0=0.0
                    if regime == "noiseless":
                        p0_values = [0.0]
                    else:
                        p0_values = p0_list

                    for p0 in p0_values:
                        cfg = VQERunConfig(
                            regime=regime,
                            p0=p0,
                            shot_budget=shot_budget,
                            num_iterations=num_iterations,
                            seed=seed,
                            noise_scales=noise_scales,
                        )
                        print(
                            f"Running VQE: regime={regime}, p0={p0}, "
                            f"shots={shot_budget}, seed={seed}"
                        )
                        history = run_vqe(ansatz_base, params, cfg, exact_ground)
                        for rec in history:
                            writer.writerow({
                                "regime": rec["regime"],
                                "p0": rec["p0"],
                                "shot_budget": rec["shot_budget"],
                                "seed": rec["seed"],
                                "iteration": rec["iteration"],
                                "energy_best": rec["energy_best"],
                                "exact_ground": rec["exact_ground"],
                            })

    print(f"Results written to {out_file}")


if __name__ == "__main__":
    main()
