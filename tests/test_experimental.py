from typing import Sequence
import pytest
import numpy as np
from jax import numpy as jnp, jit, grad, random

from pytket.extensions.qiskit.backends.aer import AerDensityMatrixBackend

from pytket.circuit import Circuit, CircBox

from pytket.extensions.qujax.experimental import tk_to_qujax_args

from qujax.densitytensor import all_zeros_densitytensor
from qujax.experimental.densitytensor import get_params_to_densitytensor_func


def test_measure_and_reset() -> None:

    n_qubits = 2
    n_bits = 1
    circuit = Circuit(n_qubits, n_bits)
    circuit.H(0)
    circuit.CX(0, 1)
    circuit.Measure(0, 0)
    circuit.Reset(1)

    op_seq, op_metaparams_seq, param_inds_seq, n_qb, n_b = tk_to_qujax_args(
        circuit, simulator="statetensor"
    )

    assert op_seq == ["H", "CX", "Measure", "Reset"]
    assert op_metaparams_seq == [(0,), (0, 1), (0, 0), (1,)]
    assert param_inds_seq == [(), (), {"prng_keys": 0}, {"prng_keys": 1}]
    assert n_qubits == n_qb
    assert n_bits == n_b


def test_subcircuit_parameters_dt() -> None:

    n_qubits = 3
    n_bits = 0
    circuit = Circuit(n_qubits, n_bits)
    np.random.seed(0)

    subcircuit_1_repeats = 3
    subcircuit_2_repeats = 5
    subsubcircuit_repeats = 2

    def get_subsubcircuit(params: np.ndarray) -> Circuit:
        subsubcircuit = Circuit(n_qubits - 1, n_bits)
        subsubcircuit.Ry(params[0], 0)
        subsubcircuit.Ry(params[1], 1)
        subsubcircuit.YYPhase(params[2], 0, 1)
        return subsubcircuit

    def get_subcircuit_1(params: np.ndarray) -> Circuit:
        subcircuit_1 = Circuit(n_qubits - 1, n_bits)
        subcircuit_1.Rx(params[0], 0)
        subcircuit_1.Rx(params[1], 1)
        subcircuit_1.ZZPhase(params[2], 0, 1)
        subcircuit_1.Reset(0)
        return subcircuit_1

    def get_subcircuit_2(params: np.ndarray) -> Circuit:
        subcircuit_2 = Circuit(n_qubits - 1, n_bits)
        subcircuit_2.Rz(params[0], 0)
        subcircuit_2.Ry(params[1], 1)
        subcircuit_2.XXPhase(params[2], 0, 1)
        for i in range(subsubcircuit_repeats):
            sc = get_subsubcircuit(np.random.rand(3))
            sc.name = f"c_{i}"
            subcircuit_2.add_circbox(CircBox(sc), [0, 1])
        return subcircuit_2

    for i in range(subcircuit_1_repeats):
        sc = get_subcircuit_1(np.random.rand(3))
        sc.name = f"a_{i}"
        circuit.add_circbox(CircBox(sc), [0, 1])

    for i in range(subcircuit_2_repeats):
        sc = get_subcircuit_2(np.random.rand(3))
        sc.name = f"b_{i}"
        circuit.add_circbox(CircBox(sc), [1, 2])

    (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        n_qb,
        n_b,
        params,
        rng_param_index,
        pytket_to_qujax_qubit_map,
    ) = tk_to_qujax_args(circuit, simulator="densitytensor")

    expected_op_sequence = ["RepeatingSubcircuit", "RepeatingSubcircuit"]
    expected_metaparam_sequence = []
    expected_param_ind_sequence = [
        {"repeating_parameters": "a"},
        {"repeating_parameters": "b"},
    ]

    sub_op_seq_1 = ["Rx", "Rx", "ZZPhase", "Reset"]
    sub_metaparam_seq_1 = [[0], [1], [0, 1], [0]]
    sub_param_ind_seq_1 = [
        {"gate_parameters": (0,)},
        {"gate_parameters": (1,)},
        {"gate_parameters": (2,)},
        [],
    ]
    expected_metaparam_sequence.append(
        (sub_op_seq_1, sub_metaparam_seq_1, sub_param_ind_seq_1)
    )

    sub_op_seq_2 = ["Rz", "Ry", "XXPhase", "RepeatingSubcircuit"]
    sub_metaparam_seq_2 = [[1], [2], [1, 2]]
    sub_param_ind_seq_2 = [
        {"gate_parameters": (0,)},
        {"gate_parameters": (1,)},
        {"gate_parameters": (2,)},
        {"repeating_parameters": "c"},
    ]

    sub_sub_op_seq = ["Ry", "Ry", "YYPhase"]
    sub_sub_metaparam_seq = [[1], [2], [1, 2]]
    sub_sub_param_ind_seq = [
        {"gate_parameters": (0,)},
        {"gate_parameters": (1,)},
        {"gate_parameters": (2,)},
    ]
    sub_metaparam_seq_2.append(
        (sub_sub_op_seq, sub_sub_metaparam_seq, sub_sub_param_ind_seq)
    )

    expected_metaparam_sequence.append(
        (sub_op_seq_2, sub_metaparam_seq_2, sub_param_ind_seq_2)
    )

    assert op_seq == expected_op_sequence
    assert op_metaparams_seq == expected_metaparam_sequence
    assert param_inds_seq == expected_param_ind_sequence
    assert n_qubits == n_qb
    assert n_bits == n_b


def test_subcircuit_dt_one_level() -> None:
    n_qubits = 3
    n_bits = 0
    circuit = Circuit(n_qubits, n_bits)
    np.random.seed(0)

    subcircuit_1_repeats = 3

    def get_subcircuit_1(params: np.ndarray) -> Circuit:
        subcircuit_1 = Circuit(n_qubits - 1, n_bits)
        subcircuit_1.Rx(params[0], 0)
        subcircuit_1.Rx(params[1], 1)
        subcircuit_1.ZZPhase(params[2], 0, 1)
        subcircuit_1.Reset(0)
        return subcircuit_1

    for i in range(subcircuit_1_repeats):
        sc = get_subcircuit_1(np.random.rand(3))
        sc.name = f"a_{i}"
        circuit.add_circbox(CircBox(sc), [0, 1])

    (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        n_qb,
        n_b,
        params,
        rng_param_index,
        pytket_to_qujax_qubit_map,
    ) = tk_to_qujax_args(circuit, simulator="densitytensor")

    backend = AerDensityMatrixBackend()
    compiled = backend.get_compiled_circuit(circuit)
    handle = backend.process_circuit(compiled)
    result = backend.get_result(handle)
    dm = result.get_density_matrix().reshape((2,) * n_qubits * 2)

    params_to_densitytensor_func = get_params_to_densitytensor_func(
        op_seq, op_metaparams_seq, param_inds_seq
    )
    densitytensor_in = all_zeros_densitytensor(n_qubits)
    densitytensor, _ = params_to_densitytensor_func(params, densitytensor_in)

    assert jnp.allclose(dm, densitytensor)


def test_subcircuit_dt() -> None:
    n_qubits = 3
    n_bits = 0
    circuit = Circuit(n_qubits, n_bits)
    np.random.seed(0)

    subcircuit_1_repeats = 3
    subcircuit_2_repeats = 5
    subsubcircuit_repeats = 2

    def get_subsubcircuit(params: np.ndarray) -> Circuit:
        subsubcircuit = Circuit(n_qubits - 1, n_bits)
        subsubcircuit.Ry(params[0], 0)
        subsubcircuit.Ry(params[1], 1)
        subsubcircuit.YYPhase(params[2], 0, 1)
        return subsubcircuit

    def get_subcircuit_1(params: np.ndarray) -> Circuit:
        subcircuit_1 = Circuit(n_qubits - 1, n_bits)
        subcircuit_1.Rx(params[0], 0)
        subcircuit_1.Rx(params[1], 1)
        subcircuit_1.ZZPhase(params[2], 0, 1)
        subcircuit_1.Reset(0)
        return subcircuit_1

    def get_subcircuit_2(params: np.ndarray) -> Circuit:
        subcircuit_2 = Circuit(n_qubits - 1, n_bits)
        subcircuit_2.Rz(params[0], 0)
        subcircuit_2.Ry(params[1], 1)
        subcircuit_2.XXPhase(params[2], 0, 1)
        for i in range(subsubcircuit_repeats):
            sc = get_subsubcircuit(np.random.rand(3))
            sc.name = f"c_{i}"
            subcircuit_2.add_circbox(CircBox(sc), [0, 1])
        return subcircuit_2

    circuit.X(1)
    circuit.X(2)

    for i in range(subcircuit_1_repeats):
        sc = get_subcircuit_1(np.random.rand(3))
        sc.name = f"a_{i}"
        circuit.add_circbox(CircBox(sc), [0, 1])

    for i in range(subcircuit_2_repeats):
        sc = get_subcircuit_2(np.random.rand(3))
        sc.name = f"b_{i}"
        circuit.add_circbox(CircBox(sc), [1, 2])

    (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        n_qb,
        n_b,
        params,
        rng_param_index,
        pytket_to_qujax_qubit_map,
    ) = tk_to_qujax_args(circuit, simulator="densitytensor")

    backend = AerDensityMatrixBackend()
    compiled = backend.get_compiled_circuit(circuit)
    handle = backend.process_circuit(compiled)
    result = backend.get_result(handle)
    dm = result.get_density_matrix().reshape((2,) * n_qubits * 2)

    params_to_densitytensor_func = get_params_to_densitytensor_func(
        op_seq, op_metaparams_seq, param_inds_seq
    )
    densitytensor_in = all_zeros_densitytensor(n_qubits)
    densitytensor, _ = params_to_densitytensor_func(params, densitytensor_in)

    assert jnp.allclose(dm, densitytensor)
