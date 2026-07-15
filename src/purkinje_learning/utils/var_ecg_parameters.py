from __future__ import annotations

from typing import List

import numpy as np

from purkinje_learning.bo_ecg import OptimParam
from purkinje_learning.bo_purkinje_tree import BO_PurkinjeTreeConfig
from purkinje_learning.bo_utils import BOECGParameter, Prior


def var_ecg_parameters_list(
    var_parameters_names: List[BOECGParameter],
    dim: int = 2,
    prior: Prior = Prior.UNIFORM,
) -> List[OptimParam]:
    var_parameters: List[OptimParam] = []

    if BOECGParameter.INIT_LENGTH in var_parameters_names:
        lb = 30.0 * np.ones(dim)
        ub = 100.0 * np.ones(dim)
        var_parameters.append(OptimParam(BOECGParameter.INIT_LENGTH.value, lb, ub, prior))

    if BOECGParameter.LENGTH in var_parameters_names:
        lb = 4.0 * np.ones(1)
        ub = 12.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.LENGTH.value, lb, ub, prior))

    if BOECGParameter.W in var_parameters_names:
        lb = 0.05 * np.ones(1)
        ub = 0.25 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.W.value, lb, ub, prior))

    if BOECGParameter.L_SEGMENT in var_parameters_names:
        lb = 1.0 * np.ones(dim)
        ub = 15.0 * np.ones(dim)
        var_parameters.append(OptimParam(BOECGParameter.L_SEGMENT.value, lb, ub, prior))

    if BOECGParameter.FASCICLES_LENGTH in var_parameters_names:
        lb = 2.0 * np.ones(2 * dim)
        ub = 50.0 * np.ones(2 * dim)
        var_parameters.append(
            OptimParam(BOECGParameter.FASCICLES_LENGTH.value, lb, ub, prior)
        )

    if BOECGParameter.FASCICLES_ANGLES in var_parameters_names:
        lb = -1.0 / 4.0 * np.pi * np.ones(2 * dim)
        ub = 3.0 / 4.0 * np.pi * np.ones(2 * dim)
        var_parameters.append(
            OptimParam(BOECGParameter.FASCICLES_ANGLES.value, lb, ub, prior)
        )

    if BOECGParameter.BRANCH_ANGLE in var_parameters_names:
        lb = 5.0 * np.pi / 180.0 * np.ones(1)
        ub = 45.0 * np.pi / 180.0 * np.ones(1)
        var_parameters.append(
            OptimParam(BOECGParameter.BRANCH_ANGLE.value, lb, ub, prior)
        )

    if BOECGParameter.ROOT_TIME in var_parameters_names:
        lb = -75.0 * np.ones(1)
        ub = 50.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.ROOT_TIME.value, lb, ub, prior))

    if BOECGParameter.CV in var_parameters_names:
        lb = 2.0 * np.ones(1)
        ub = 4.0 * np.ones(1)
        var_parameters.append(OptimParam(BOECGParameter.CV.value, lb, ub, prior))

    return var_parameters


def _as_pair(value: object, default: List[float]) -> List[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return list(default)
        if len(value) == 2:
            return [float(value[0]), float(value[1])]
        mid = len(value) // 2
        left = float(np.mean(value[:mid])) if mid > 0 else float(np.mean(value))
        right = float(np.mean(value[mid:])) if mid > 0 else float(np.mean(value))
        return [left, right]
    return [float(value), float(value)]


def initial_values(
    var_parameters: List[OptimParam],
    N_it: int = 20,
) -> BO_PurkinjeTreeConfig:
    param_dict = {p.parameter: float(np.mean(p.lower)) for p in var_parameters}

    return BO_PurkinjeTreeConfig(
        init_length=param_dict.get("init_length", 30.0),
        length=param_dict.get("length", 8.0),
        w=param_dict.get("w", 0.1),
        l_segment=param_dict.get("l_segment", 1.0),
        fascicles_length=_as_pair(
            param_dict.get("fascicles_length"), [20.0, 20.0]
        ),
        fascicles_angles=_as_pair(param_dict.get("fascicles_angles"), [1.0, 1.0]),
        branch_angle=param_dict.get("branch_angle", 0.15),
        N_it=N_it,
    )
