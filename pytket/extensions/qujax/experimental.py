from typing import Tuple, Sequence, Optional, List, Union, Callable, Any, Literal

from jax import numpy as jnp

import qujax  # type: ignore
from pytket import Circuit  # type: ignore

from pytket.extensions.qujax.qujax_convert import (
    _tk_qubits_to_inds,
    _symbolic_command_to_gate_and_param_inds,
)


def tk_to_qujax_args(
    circuit: Circuit,
    symbol_map: Optional[dict] = None,
    simulator: Literal["statetensor"] | Literal["densitytensor"] = "statetensor",
) -> Tuple[
    Sequence[Union[str, Callable[[jnp.ndarray], jnp.ndarray]]],
    Sequence[Sequence[int]],
    Sequence[Sequence[int]],
    int,
    int,
    Sequence[float],
    int,
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
    params = []
    for c in circuit.get_commands():
        op_name = c.op.type.name
        if op_name == "Barrier":
            continue
        elif op_name == "Measure":
            metaparams_seq = (c.qubits[0].index[0], c.bits[0].index[0])
            if simulator == "statetensor":
                param_inds = {"measurement_prng_keys": rng_param_index}
                rng_param_index += 1
            else:
                param_inds = {}
        elif op_name == "Reset":
            metaparams_seq = (c.qubits[0].index[0],)
            if simulator == "statetensor":
                param_inds = {"measurement_prng_keys": rng_param_index}
                rng_param_index += 1
            else:
                param_inds = {}
        else:
            if symbol_map:
                op_name, param_inds = _symbolic_command_to_gate_and_param_inds(
                    c, symbol_map
                )
            else:
                if op_name not in qujax.experimental.statetensor.get_default_gates():
                    raise TypeError(
                        f"{op_name} gate not found in qujax.gates. \n pytket-qujax "
                        "can automatically convert arbitrary non-parameterised gates "
                        "when specified in a symbolic circuit and absent from the "
                        "symbol_map argument.\n Arbitrary parameterised gates can be "
                        "added to a local qujax.gates installation and/or submitted "
                        "via pull request."
                    )
            n_params = len(c.op.params)
            params += c.op.params

            metaparams_seq = _tk_qubits_to_inds(c.qubits)
            param_inds = {
                "gate_parameters": tuple(range(param_index, param_index + n_params))
            }
            if n_params == 0:
                param_inds = ()
            param_index += n_params

        op_seq.append(op_name)
        op_metaparams_seq.append(metaparams_seq)
        param_inds_seq.append(param_inds)

    return (
        op_seq,
        op_metaparams_seq,
        param_inds_seq,
        circuit.n_qubits,
        circuit.n_bits,
        params,
        rng_param_index,
    )
