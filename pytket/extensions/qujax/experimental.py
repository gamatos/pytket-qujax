from typing import (
    Tuple,
    Sequence,
    Optional,
    List,
    Union,
    Callable,
    Any,
    Literal,
    Mapping,
)

from copy import copy

import sympy

import jax
from jax import numpy as jnp

import qujax  # type: ignore
import pytket
import pytket.circuit
from pytket import Circuit  # type: ignore
from pytket import Qubit

from pytket.extensions.qujax.qujax_convert import (
    _symbolic_command_to_gate_and_param_inds,
)

def _get_pexb(tensor, d):
    identity = jnp.diag(jnp.ones(tensor.shape[0]))
    def _pexb(p) -> jax.Array:
        a = -1 / 2 * jnp.pi * p
        gate = jnp.cos(a) * identity + 1j * jnp.sin(a) * tensor
        gate = gate.reshape((2,) * 2 * d)
        return gate
    return _pexb


def try_get_repeat_identifier(name: str | None) -> Tuple[str, int] | Tuple[None, None]:
    if isinstance(name, str) and len(name) > 0:
        s = name.split("_")
        if len(s) == 2:
            identifier, nr = s
            try:
                return identifier, int(nr)
            except Exception:
                pass
    return (None, None)


def tk_to_qujax_args(
    circuit: Circuit,
    symbol_map: Optional[dict] = None,
    simulator: Literal["statetensor"] | Literal["densitytensor"] = "statetensor",
    root=True,
) -> Tuple[
    Sequence[Union[str, Callable[[jnp.ndarray], jnp.ndarray]]],
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    int,
    int,
    Mapping[str, Any],
    int,
    Mapping[pytket.Qubit, int],
]:
    """
    Converts a pytket circuit into a tuple of arguments representing
    a qujax quantum circuit.
    Assumes all circuit gates can be found in ``qujax.gates``
    The ``symbol_map`` argument controls the interpretation of any symbolic parameters
    found in ``circuit.free_symbols()``.

    - If ``symbol_map`` is ``None``, circuit.free_symbols() will be ignored.
      Parameterised gates will be determined based on whether they are stored as
      functions (parameterised) or arrays (non-parameterised) in qujax.gates. The order
      of qujax circuit parameters is the same as in circuit.get_commands().
    - If ``symbol_map`` is provided as a ``dict``, assign qujax circuit parameters to
      symbolic parameters in ``circuit.free_symbols()``; the order of qujax circuit
      parameters will be given by this dict. Keys of the dict should be symbolic pytket
      parameters as in ``circuit.free_symbols()`` and the values indicate
      the index of the qujax circuit parameter -- integer indices starting from 0.

    The conversion can also be checked with ``print_circuit``.

    :param circuit: Circuit to be converted (without any measurement commands).
    :type circuit: pytket.Circuit
    :param symbol_map:
        If ``None``, parameterised gates determined by ``qujax.gates``. \n
        If ``dict``, maps symbolic pytket parameters following the order in this dict.
    :type symbol_map: Optional[dict]
    :return: Tuple of arguments defining a (parameterised) quantum circuit
        that can be sent to ``qujax.get_params_to_statetensor_func``. The elements of
        the tuple (qujax args) are as follows

        - Sequence of gate name strings to be found in ``qujax.gates``.
        - Sequence of sequences describing which qubits gates act on.
        - Sequence of sequences of parameter indices that gates are using.
        - Number of qubits.

    :rtype: Tuple[Sequence[str], Sequence[Sequence[int]], Sequence[Sequence[int]], int]
    """
    if symbol_map:
        assert (
            set(symbol_map.keys()) == circuit.free_symbols()
        ), "Circuit keys do not much symbol_map"
        assert set(symbol_map.values()) == set(
            range(len(circuit.free_symbols()))
        ), "Incorrect indices in symbol_map"

    op_seq = []
    op_metaparams_seq = []
    param_inds_seq = []
    param_index = 0
    rng_param_index = 0
    params = {"gate_parameters": [], "repeating_parameters": {}}
    previous_op = None
    cur_op = None
    pytket_to_qujax_qubit_map = {q: i for i, q in enumerate(circuit.qubits)}
    pytket_to_qujax_bit_map = {b: i for i, b in enumerate(circuit.bits)}

    for c in circuit.get_commands():
        previous_op = cur_op
        cur_op = c.op

        op_name = c.op.type.name
        if op_name == "Barrier":
            continue
        elif type(c.op) is pytket.circuit.CircBox:
            sub_circuit = c.op.get_circuit()
            (
                sub_op_seq,
                sub_op_metaparams_seq,
                sub_param_inds_seq,
                sub_n_qubits,
                sub_n_bits,
                sub_params,
                sub_rng_param_index,
                sub_pytket_to_qujax_qubit_map,
            ) = tk_to_qujax_args(sub_circuit, symbol_map, simulator, root=False)

            sub_qujax_to_pytket_qubit_map = {
                v: k for k, v in sub_pytket_to_qujax_qubit_map.items()
            }

            # Check if previous operation was a CircBox with a repeating pattern
            previous_circuit_name = (
                None
                if not isinstance(previous_op, pytket.circuit.CircBox)
                else previous_op.circuit_name
            )
            previous_circuit_repeat_identifier, previous_circuit_repeat_nr = (
                try_get_repeat_identifier(previous_circuit_name)
            )

            starting_repeats = False
            continuing_repeats = False
            # Check if current operation is a CircBox with a repeating pattern
            repeat_identifier, repeat_nr = try_get_repeat_identifier(c.op.circuit_name)
            if repeat_identifier is not None:
                # Exists in dict
                if repeat_identifier in params["repeating_parameters"]:
                    # Check if continuing
                    if (
                        previous_circuit_repeat_identifier is not None
                        and previous_circuit_repeat_nr is not None
                    ):
                        if repeat_identifier == previous_circuit_repeat_identifier:
                            # In dict and continuing
                            if previous_circuit_repeat_nr + 1 == repeat_nr:
                                continuing_repeats = True
                else:  # does not exist in dict
                    starting_repeats = True

            # Map subcircuit qubits to circuit qubits
            # IMPORTANT TODO: filter metaparams that are qubits from ones that are not!
            circbox_qubit_map = {k: v for k, v in zip(sub_circuit.qubits, c.qubits)}

            def _remap_qubits(q):
                if not isinstance(q, int):
                    return q
                return pytket_to_qujax_qubit_map[
                    circbox_qubit_map[sub_qujax_to_pytket_qubit_map[q]]
                ]

            def _recursive_remap_qubits(op, mp):
                for i, (n, p) in enumerate(zip(op, mp)):
                    if n == "RepeatingSubcircuit":
                        _recursive_remap_qubits(p[0], p[1])
                    else:
                        mp[i] = jax.tree.map(_remap_qubits, p)

            _recursive_remap_qubits(sub_op_seq, sub_op_metaparams_seq)

            if symbol_map is not None:
                param_inds_seq_to_append = sub_param_inds_seq
            else:

                def _remap_param_inds(i):
                    if not isinstance(i, int):
                        return i
                    return i + param_index

                # Map subparameters to circuit parameters
                shifted_sub_param_inds_seq = jax.tree.map(
                    _remap_param_inds, sub_param_inds_seq
                )

                # Add circuit parameters to parameter count
                max_sub_param_ind = max(
                    [
                        x
                        for x in jax.tree.flatten(sub_param_inds_seq)[0]
                        if isinstance(x, int)
                    ]
                    + [-1]
                )

                param_inds_seq_to_append = shifted_sub_param_inds_seq

            if starting_repeats:
                op_seq += ("RepeatingSubcircuit",)
                param_inds_seq.append({"repeating_parameters": repeat_identifier})
                op_metaparams_seq += [
                    (
                        sub_op_seq,
                        sub_op_metaparams_seq,
                        sub_param_inds_seq,
                    )
                ]
                params["repeating_parameters"][repeat_identifier] = [sub_params]
            elif continuing_repeats:
                params["repeating_parameters"][repeat_identifier].append(sub_params)
            else:
                op_seq += sub_op_seq
                param_inds_seq += param_inds_seq_to_append
                op_metaparams_seq += sub_op_metaparams_seq
                params["gate_parameters"] += sub_params["gate_parameters"]
                params["repeating_parameters"] |= sub_params["repeating_parameters"]
                param_index += max_sub_param_ind + 1

        elif type(c.op) is pytket.circuit.PauliExpBox:
            paulis = c.op.get_paulis()
            tensor = jnp.ones(1)
            # Build tensor product of Pauli matrices
            for p in paulis:
                m = qujax.gates.__dict__[p.name]
                tensor = jnp.kron(tensor, m)            
            metaparams_seq = [pytket_to_qujax_qubit_map[q] for q in c.qubits]

            _pexb = _get_pexb(tensor, len(metaparams_seq))

            if symbol_map is not None:
                op_name, _param_inds = _symbolic_command_to_gate_and_param_inds(
                    c, symbol_map, _pexb
                )
                param_inds = {"gate_parameters": _param_inds}
            else:
                op_name = _pexb
                param_inds = {"gate_parameters": param_index}
                param_index += 1
                params["gate_parameters"].append(c.op.get_phase())

            op_seq.append(op_name)
            op_metaparams_seq.append(metaparams_seq)
            param_inds_seq.append(param_inds)
        elif op_name == "Measure":
            metaparams_seq = (c.qubits[0].index[0], c.bits[0].index[0])
            if simulator == "statetensor":
                param_inds = {"measurement_prng_keys": rng_param_index}
                rng_param_index += 1
            else:
                param_inds = {}

            op_seq.append(op_name)
            op_metaparams_seq.append(metaparams_seq)
            param_inds_seq.append(param_inds)
        elif op_name == "Reset":
            metaparams_seq = [pytket_to_qujax_qubit_map[q] for q in c.qubits]
            if simulator == "statetensor":
                param_inds = {"measurement_prng_keys": rng_param_index}
                rng_param_index += 1
            else:
                param_inds = []

            op_seq.append(op_name)
            op_metaparams_seq.append(metaparams_seq)
            param_inds_seq.append(param_inds)
        else:
            if symbol_map is not None:
                op_name, param_inds = _symbolic_command_to_gate_and_param_inds(
                    c, symbol_map
                )
            else:
                if op_name in qujax.experimental.statetensor.get_default_gates():
                    n_params = len(c.op.params)
                    params["gate_parameters"] += c.op.params

                    metaparams_seq = [pytket_to_qujax_qubit_map[q] for q in c.qubits]
                    param_inds = {
                        "gate_parameters": tuple(
                            range(param_index, param_index + n_params)
                        )
                    }
                    if n_params == 0:
                        param_inds = ()
                    param_index += n_params
                else:
                    raise TypeError(
                        f"{op_name} gate not found in qujax.gates. \n pytket-qujax "
                        "can automatically convert arbitrary non-parameterised gates "
                        "when specified in a symbolic circuit and absent from the "
                        "symbol_map argument.\n Arbitrary parameterised gates can be "
                        "added to a local qujax.gates installation and/or submitted "
                        "via pull request."
                    )

            op_seq.append(op_name)
            op_metaparams_seq.append(metaparams_seq)
            param_inds_seq.append(param_inds)

    def _is_leaf(l):
        return (
            isinstance(l, jax.Array)
            or isinstance(l, list)
            and all(isinstance(x, float) for x in l)
        )

    if root:
        params = jax.tree.map(jnp.array, params, is_leaf=_is_leaf)

    return (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        circuit.n_qubits,
        circuit.n_bits,
        params,
        rng_param_index,
        pytket_to_qujax_qubit_map,
    )
