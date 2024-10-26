from typing import Sequence, Tuple, Union
import pytest
import numpy as np
import jax
from jax import numpy as jnp, jit, grad, random

from pytket.extensions.qiskit.backends.aer import AerDensityMatrixBackend

from pytket.circuit import Circuit, CircBox, PauliExpBox
from pytket.pauli import Pauli

from pytket.extensions.qujax.experimental import tk_to_qujax_args

from qujax.densitytensor import all_zeros_densitytensor
from qujax.experimental.densitytensor import get_params_to_densitytensor_func


def _test_circuit_experimental(circuit: Circuit) -> None:
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
    dm = result.get_density_matrix().reshape((2,) * n_qb * 2)

    params_to_densitytensor_func = get_params_to_densitytensor_func(
        op_seq, op_metaparams_seq, param_inds_seq
    )
    densitytensor_in = all_zeros_densitytensor(n_qb)
    densitytensor, _ = params_to_densitytensor_func(params, densitytensor_in)

    assert jnp.allclose(dm, densitytensor)


# def _test_circuit_experimental(
#     circuit: Circuit, param: Union[None, jnp.ndarray], test_two_way: bool = False
# ) -> None:
#     true_sv = circuit.get_statevector()
#     true_probs = jnp.square(jnp.abs(true_sv))

#     apply_circuit = tk_to_qujax(circuit)
#     jit_apply_circuit = jit(apply_circuit)

#     apply_circuit_dt = tk_to_qujax(circuit, simulator="densitytensor")
#     jit_apply_circuit_dt = jit(apply_circuit_dt)

#     if param is None:
#         test_sv = apply_circuit().flatten()
#         test_jit_sv = jit_apply_circuit().flatten()

#         test_dt = apply_circuit_dt()
#         n_qubits = test_dt.ndim // 2
#         test_dm_diag = jnp.diag(test_dt.reshape(2**n_qubits, 2**n_qubits))
#         test_jit_dm_diag = jnp.diag(
#             jit_apply_circuit_dt().reshape(2**n_qubits, 2**n_qubits)
#         )
#     else:
#         test_sv = apply_circuit(param).flatten()
#         test_jit_sv = jit_apply_circuit(param).flatten()
#         test_dt = apply_circuit_dt(param)
#         n_qubits = test_dt.ndim // 2
#         test_dm_diag = jnp.diag(test_dt.reshape(2**n_qubits, 2**n_qubits))
#         test_jit_dm_diag = jnp.diag(
#             jit_apply_circuit_dt(param).reshape(2**n_qubits, 2**n_qubits)
#         )

#         assert jnp.allclose(param, tk_to_param(circuit))

#     assert jnp.allclose(test_sv, true_sv)
#     assert jnp.allclose(test_jit_sv, true_sv)
#     assert jnp.allclose(test_dm_diag.real, true_probs)
#     assert jnp.allclose(test_jit_dm_diag, true_probs)

#     if param is not None:
#         cost_func = lambda p: jnp.square(apply_circuit(p)).real.sum()
#         grad_cost_func = grad(cost_func)
#         assert isinstance(grad_cost_func(param), jnp.ndarray)

#         cost_jit_func = lambda p: jnp.square(jit_apply_circuit(p)).real.sum()
#         grad_cost_jit_func = grad(cost_jit_func)
#         assert isinstance(grad_cost_jit_func(param), jnp.ndarray)

#     if test_two_way:
#         circuit_commands = [
#             com for com in circuit.get_commands() if str(com.op) != "Barrier"
#         ]
#         circuit_2 = qujax_args_to_tk(*tk_to_qujax_args(circuit), param)  # type: ignore
#         assert all(
#             g.op.type == g2.op.type
#             for g, g2 in zip(circuit_commands, circuit_2.get_commands())
#         )
#         assert all(
#             g.qubits == g2.qubits
#             for g, g2 in zip(circuit_commands, circuit_2.get_commands())
#         )


def get_circuit1(
    n_qubits: int, depth: int, param_seed: int
) -> Tuple[Circuit, jax.Array]:
    circuit = Circuit(n_qubits)

    param = random.uniform(random.PRNGKey(param_seed), (n_qubits * (depth + 1),)) * 2

    k = 0
    for i in range(n_qubits):
        circuit.Ry(float(param[k]), i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            circuit.CX(i, i + 1)
        circuit.add_barrier(list(range(0, n_qubits)))
        for i in range(n_qubits):
            circuit.Ry(float(param[k]), i)
            k += 1

    return circuit, param


def get_circuit2(
    n_qubits: int, depth: int, param_seed: int
) -> Tuple[Circuit, jax.Array]:
    circuit = Circuit(n_qubits)

    param = (
        random.uniform(random.PRNGKey(param_seed), (2 * n_qubits * (depth + 1),)) * 2
    )

    k = 0
    for i in range(n_qubits):
        circuit.H(i)
    for i in range(n_qubits):
        circuit.Rz(float(param[k]), i)
        k += 1
    for i in range(n_qubits):
        circuit.Rx(float(param[k]), i)
        k += 1

    for _ in range(depth):
        for i in range(0, n_qubits - 1):
            circuit.CZ(i, i + 1)
        circuit.add_barrier(list(range(0, n_qubits)))
        for i in range(n_qubits):
            circuit.Rz(float(param[k]), i)
            k += 1
        for i in range(n_qubits):
            circuit.Rx(float(param[k]), i)
            k += 1

    return circuit, param


def test_pauli_exp_box() -> None:
    circuit = Circuit(3, 0)

    box = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], 0.22)
    circuit.add_pauliexpbox(box, [0,1,2])

    _test_circuit_experimental(circuit)


def test_multiple_pauli_exp_box() -> None:
    circuit = Circuit(3, 0)

    box = PauliExpBox([Pauli.X, Pauli.Y, Pauli.Z], 0.22)
    circuit.add_pauliexpbox(box, [0, 1, 2])
    box = PauliExpBox([Pauli.X, Pauli.X], 0.11)
    circuit.add_pauliexpbox(box, [0, 1])
    box = PauliExpBox([Pauli.X, Pauli.X], 0.33)
    circuit.add_pauliexpbox(box, [1, 2])
    box = PauliExpBox([Pauli.Z], 0.22)
    circuit.add_pauliexpbox(box, [0])
    box = PauliExpBox([Pauli.Z], 0.44)
    circuit.add_pauliexpbox(box, [2])

    _test_circuit_experimental(circuit)


def test_circ_box() -> None:
    circbox_qubits = 3
    extra_qubits = 1
    total_qubits = circbox_qubits + extra_qubits
    depth = 1

    circuit = Circuit(total_qubits, 0)

    circuit_1, param_1 = get_circuit1(circbox_qubits, depth, 0)
    circuit_2, param_2 = get_circuit2(circbox_qubits, depth, 1)

    param = jnp.concat([param_1, param_2])

    circuit.add_circbox(CircBox(circuit_1), list(range(circbox_qubits)))
    circuit.add_circbox(CircBox(circuit_2), list(range(extra_qubits, total_qubits)))

    _test_circuit_experimental(circuit)


def test_nested_circ_box() -> None:
    circbox_qubits = 3
    extra_qubits = 1
    total_qubits = circbox_qubits + extra_qubits
    depth = 1

    circuit = Circuit(total_qubits, 0)

    circuit_1, param_1 = get_circuit1(circbox_qubits, depth, 0)
    circuit_2, param_2 = get_circuit2(circbox_qubits, depth, 1)

    param = jnp.concat([param_1, param_2])

    sub_circuit = Circuit(total_qubits, 0)
    sub_circuit.name = "a_0"

    sub_circuit.add_circbox(CircBox(circuit_1), list(range(circbox_qubits)))
    sub_circuit.add_circbox(CircBox(circuit_2), list(range(extra_qubits, total_qubits)))
    circuit.add_circbox(CircBox(sub_circuit), list(range(total_qubits)))

    sub_circuit_1 = sub_circuit.copy()
    sub_circuit_1.name = "a_1"
    circuit.add_circbox(CircBox(sub_circuit_1), list(range(total_qubits)))

    _test_circuit_experimental(circuit)


def test_reset_dt() -> None:

    n_qubits = 2
    n_bits = 1
    circuit = Circuit(n_qubits, n_bits)
    circuit.H(0)
    circuit.CX(0, 1)
    circuit.Reset(1)

    _test_circuit_experimental(circuit)


def test_measure_and_reset() -> None:

    n_qubits = 2
    n_bits = 1
    circuit = Circuit(n_qubits, n_bits)
    circuit.H(0)
    circuit.CX(0, 1)
    circuit.Measure(0, 0)
    circuit.Reset(1)

    (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        n_qb,
        n_b,
        params,
        rng_param_inds,
        pytket_to_qujax_qubit_map,
    ) = tk_to_qujax_args(circuit, simulator="statetensor")

    assert op_seq == ["H", "CX", "Measure", "Reset"]
    assert op_metaparams_seq == [(0,), (0, 1), (0, 0), (1,)]
    assert param_inds_seq == [(), (), {"prng_keys": 0}, {"prng_keys": 1}]
    assert n_qubits == n_qb
    assert n_bits == n_b


def test_subcircuit_parameters_dt() -> None:

    n_qubits = 3
    n_bits = 0
    circuit = Circuit()
    anc = circuit.add_q_register("anc", 1)
    sys = circuit.add_q_register("sys", n_qubits - 1)
    qubits = anc.to_list() + sys.to_list()
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
        # subcircuit_1.Reset(0)
        return subcircuit_1

    def get_subcircuit_2(params: np.ndarray) -> Circuit:
        subcircuit_2 = Circuit()
        subsys = subcircuit_2.add_q_register("subsys", n_qubits - 1)
        subqubits = subsys.to_list()

        subcircuit_2.Rz(params[0], subqubits[0])
        subcircuit_2.Ry(params[1], subqubits[1])
        subcircuit_2.XXPhase(params[2], subqubits[0], subqubits[1])
        for i in range(subsubcircuit_repeats):
            sc = get_subsubcircuit(np.random.rand(3))
            sc.name = f"c_{i}"
            subcircuit_2.add_circbox(CircBox(sc), [subqubits[0], subqubits[1]])
        return subcircuit_2

    circuit.X(qubits[1])
    circuit.Reset(qubits[1])
    circuit.X(qubits[2])

    for i in range(subcircuit_1_repeats):
        sc = get_subcircuit_1(np.random.rand(3))
        sc.name = f"a_{i}"
        circuit.add_circbox(CircBox(sc), [qubits[0], qubits[1]])

    for i in range(subcircuit_2_repeats):
        sc = get_subcircuit_2(np.random.rand(3))
        sc.name = f"b_{i}"
        circuit.add_circbox(CircBox(sc), [qubits[1], qubits[2]])

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

    expected_op_sequence = [
        "X",
        "Reset",
        "X",
        "RepeatingSubcircuit",
        "RepeatingSubcircuit",
    ]
    expected_metaparam_sequence = [[1], [2]]
    expected_param_ind_sequence = [
        [],
        [],
        [],
        {"repeating_parameters": "a"},
        {"repeating_parameters": "b"},
    ]

    sub_op_seq_1 = ["Rx", "Rx", "ZZPhase"]
    sub_metaparam_seq_1 = [[0], [1], [0, 1], [0]]
    sub_param_ind_seq_1 = [
        {"gate_parameters": (0,)},
        {"gate_parameters": (1,)},
        {"gate_parameters": (2,)},
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
    circuit = Circuit()
    anc = circuit.add_q_register("anc", 1)
    sys = circuit.add_q_register("sys", n_qubits - 1)
    qubits = anc.to_list() + sys.to_list()
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
        # subcircuit_1.Reset(0)
        return subcircuit_1

    def get_subcircuit_2(params: np.ndarray) -> Circuit:
        subcircuit_2 = Circuit()
        subsys = subcircuit_2.add_q_register("subsys", n_qubits - 1)
        subqubits = subsys.to_list()

        subcircuit_2.Rz(params[0], subqubits[0])
        subcircuit_2.Ry(params[1], subqubits[1])
        subcircuit_2.XXPhase(params[2], subqubits[0], subqubits[1])
        wrappingsubsubcircuit = Circuit()
        wrappingsubsys = wrappingsubsubcircuit.add_q_register(
            "wrappingsubsys", n_qubits - 1
        )
        wrappingsubsys_qubits = wrappingsubsys.to_list()
        wrappingsubsubcircuit.name = "d"
        for i in range(subsubcircuit_repeats):
            sc = get_subsubcircuit(np.random.rand(3))
            sc.name = f"c_{i}"
            wrappingsubsubcircuit.add_circbox(
                CircBox(sc), [wrappingsubsys_qubits[0], wrappingsubsys_qubits[1]]
            )
        subcircuit_2.add_circbox(
            CircBox(wrappingsubsubcircuit), [subqubits[0], subqubits[1]]
        )
        return subcircuit_2

    circuit.X(qubits[1])
    circuit.Reset(qubits[1])
    circuit.X(qubits[2])

    for i in range(subcircuit_1_repeats):
        sc = get_subcircuit_1(np.random.rand(3))
        sc.name = f"a_{i}"
        circuit.add_circbox(CircBox(sc), [qubits[0], qubits[1]])

    circuit.Reset(qubits[1])

    for i in range(subcircuit_2_repeats):
        sc = get_subcircuit_2(np.random.rand(3))
        sc.name = f"b_{i}"
        circuit.add_circbox(CircBox(sc), [qubits[1], qubits[2]])

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
    densitytensor_jit, _ = jax.jit(params_to_densitytensor_func)(
        params, densitytensor_in
    )

    assert jnp.allclose(dm, densitytensor, atol=1e-7)
    assert jnp.allclose(dm, densitytensor_jit, atol=1e-7)
