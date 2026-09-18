import ast

import numpy as np
import torch

from aihwkit.nn import AnalogLinear
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs.devices import (
    ConstantStepDevice, ExpStepDevice, LinearStepDevice, PowStepDevice,
    SoftBoundsDevice, SoftBoundsPmaxDevice,
)
from aihwkit.simulator.parameters.inference import DriftParameter
from aihwkit.utils.fitting import fit_measurements
from aihwkit.utils.visualization import compute_pulse_response

VAR_DEF=[("Cycle to Cycle Step Standard Deviation","dw_min_std",0),
         ("Device to Device Step Standard Deviation","dw_min_dtod", 0),
         ("Device to Device Minium Conductance Standard Deviation","w_min_dtod",0),
         ("Device to Device Maximum Conductance Standard Deviation","w_max_dtod",0)]

# Unit the measurement file's conductance values are written in, keyed by
# the label shown to the user, valued by the multiplier that converts one
# unit into Siemens (e.g. a file written in mS has each value multiplied
# by 1e-3 to get Siemens). fitting() itself works in uS (see its docstring).
UNIT_MULTIPLIERS = {
    "S": 1.0,
    "mS": 1e-3,
    "µS": 1e-6,
    "nS": 1e-9,
    "pS": 1e-12,
}

# Device types tried by fitting() when asked to fit "different devices" --
# all share the same (w_min, w_max) constructor and the same dw_min/up_down/
# w_max fit parameters, so the exact same fit_measurements() call works for
# each of them.
DEVICE_CLASSES = {
    cls.__name__: cls for cls in (
        SoftBoundsDevice, SoftBoundsPmaxDevice, ExpStepDevice,
        LinearStepDevice, ConstantStepDevice, PowStepDevice,
    )
}

# Extra per-device-class fit parameters, on top of the common dw_min/up_down/w_max
# below -- fit_measurements() only varies whatever is listed in the `parameters`
# dict passed to it, so anything not listed just keeps aihwkit's built-in class
# default. ExpStepDevice's A_up/A_down/gamma_up/gamma_down/a/b shape its LTP/LTD
# step-size nonlinearity and default to a strongly asymmetric response (A_down is
# ~450x A_up) -- fitting them calibrates that nonlinearity against the actual
# measured LTP/LTD curve instead of leaving it at that generic default. Bounds are
# two orders of magnitude either side of the default (an additive +-2 range for
# `b`, since it's an offset and plausibly negative).
EXTRA_FIT_PARAMS = {
    "ExpStepDevice": {
        "A_up": (0.00081, 0.0000081, 0.081),
        "A_down": (0.36833, 0.0036833, 36.833),
        "gamma_up": (12.44625, 0.1244625, 1244.625),
        "gamma_down": (12.78785, 0.1278785, 1278.785),
        "a": (0.244, 0.00244, 24.4),
        "b": (0.2425, -2.0, 2.0),
    },
}

def fitting(path, noise=None, unit="µS", plot=True, embed_in=None, device_classes=None):
    '''Fits every device type in `device_classes` to the data in file points: flat sequence of 2*n measured physical conductances.
        `points` arg:
            points[0:n]  -- LTP sweep: n pulses applied from the device's
                             minimum conductance
            points[n:2n] -- LTD sweep: n pulses applied from the device's
                             maximum conductance

    The measured conductances are converted to uS and then normalized: the weight is
    (G - offset) / scale with offset = (G_max + G_min)/2 and scale = (G_max - G_min)/2,
    so every fitted device spans [-1, +1] regardless of the measurement unit. This keeps
    the training behaviour (learning rates, dw_min, ADC bounds) independent of the unit
    the file happened to be written in; offset and scale are returned with each candidate
    (and saved into the .fit file) so weights can always be mapped back to physical
    conductance: G = w * scale + offset (in uS).

    Fitting Parameters are set in code
    Args:
        path (str): Path to the .txt with the measurment data
        noise (ArrayLike): Array of the Noise Values
        unit (str): unit the values in `path` are written in, one of UNIT_MILTIPLIERS's
            keys. Defaults to "uS".
        plot (bool, optional): optional standalone Plot of the best (lowest-rmse) candidate. Defaults to True.
        embed_in (_type_, optional): optional embedded Plot of the best (lowest-rmse) candidate. Defaults to None.
        device_classes (dict, optional): {name: device class} to fit. Defaults to DEVICE_CLASSES (all supported types).

    Raises:
        ValueError: if number of points is not even, if all values are identical, or if
            no device could be fit (the message then lists each device type's failure reason)

    Returns:
        list[dict]: one entry per device type that could be fit, sorted by
            ascending rmse (best first). Each entry holds:
            "name" (str), "device_config" (fitted device, noise already applied),
            "offset" (float, uS), "scale" (float, uS per weight unit), "rmse" (float),
            "title" (str), "pulses_per_direction", "ltp_fit", "ltd_fit",
            "ltp_weights", "ltd_weights" (ndarray)
        list[tuple]: (device type name, reason) for every device type that could NOT
            be fit -- empty when everything fit
    '''
    try:
        points = np.loadtxt(path, delimiter=None).flatten()
    except ValueError:
        points = np.loadtxt(path, delimiter=",").flatten()
    points = np.asarray(points, dtype=float)
    if len(points) % 2 != 0:
        raise ValueError("points must hold an even number of values (n LTP + n LTD)")
    n = len(points) // 2
    points = points * UNIT_MULTIPLIERS[unit] / 1e-06
    ltp_points, ltd_points = points[:n], points[n:]
    offset = (points.max() + points.min()) / 2
    scale = (points.max() - points.min()) / 2
    if scale == 0:
        raise ValueError("all conductance values in the file are identical -- nothing to fit")
    ltp_weights = (ltp_points - offset) / scale
    ltd_weights = (ltd_points - offset) / scale
    w_min_init = -1.0
    w_max_init = 1.0

    # Bounds are in normalized weight units (device range is exactly [-1, +1]): dw_min
    # between 1e-4 (20000 states per direction) and 1.0 (2 states); w_max may drift off the
    # last measured point to accommodate soft-bounds asymptotes.
    params = {
        'dw_min': (0.01, 0.0001, 1.0),
        'up_down': (0.0, -0.99, 0.99),
        'w_max': (w_max_init, w_max_init * 0.5, w_max_init * 2),
    }
    pulse_data = (np.ones(n), -np.ones(n))
    response_data = (ltp_weights, ltd_weights)
    pulses_per_direction = np.arange(1, n + 1)

    if device_classes is None:
        device_classes = DEVICE_CLASSES

    candidates = []
    failures = []
    for name, cls in device_classes.items():
        device_config = cls(w_min=w_min_init, w_max=w_max_init)
        class_params = {**params, **EXTRA_FIT_PARAMS.get(name, {})}
        try:
            fit_res, fit_device_config, model_response = fit_measurements(
                class_params, pulse_data, response_data,
                device_config=device_config,
                suppress_device_noise=True,
                method='powell',
            )
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            continue

        ltp_fit, ltd_fit = model_response[0][:, 0], model_response[1][:, 0]
        rmse = float(np.sqrt(np.mean(np.concatenate((
            (ltp_fit - ltp_weights) ** 2, (ltd_fit - ltd_weights) ** 2,
        )))))
        title = (
            f"{name}: dw_min={fit_res.params['dw_min'].value:.4g}, "
            f"up_down={fit_res.params['up_down'].value:.4g}, "
            f"w_max={fit_res.params['w_max'].value:.4g}, rmse={rmse:.4g}"
        )

        if noise is not None:
            fit_device_config.dw_min_std = noise["dw_min_std"]
            fit_device_config.dw_min_dtod = noise["dw_min_dtod"]
            fit_device_config.w_min_dtod = noise["w_min_dtod"]
            fit_device_config.w_max_dtod = noise["w_max_dtod"]

        candidates.append({
            "name": name,
            "device_config": fit_device_config,
            "offset": offset,
            "scale": scale,
            "rmse": rmse,
            "title": title,
            "pulses_per_direction": pulses_per_direction,
            "ltp_fit": ltp_fit,
            "ltd_fit": ltd_fit,
            "ltp_weights": ltp_weights,
            "ltd_weights": ltd_weights,
        })

    if not candidates:
        reasons = "\n".join(f"  {name}: {reason}" for name, reason in failures)
        raise ValueError(f"No device could be fit to the given measurements:\n{reasons}")
    candidates.sort(key=lambda c: c["rmse"])
    best = candidates[0]

    if embed_in is not None:
        embed_in.update_fit(
            best["pulses_per_direction"], best["ltp_fit"], best["ltd_fit"], best["title"],
            ltp_weights=best["ltp_weights"], ltd_weights=best["ltd_weights"],
        )

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(pulses_per_direction, best["ltp_weights"], "o", color="blue", label="LTP (data - offset)")
        ax.plot(pulses_per_direction, best["ltd_weights"], "o", color="orange", label="LTD (data - offset)")
        ax.plot(pulses_per_direction, best["ltp_fit"], "-", color="navy", label="LTP (fit)")
        ax.plot(pulses_per_direction, best["ltd_fit"], "-", color="darkorange", label="LTD (fit)")


        ax.set_title(best["title"])
        ax.set_xlabel("Pulse number")
        ax.set_ylabel("Weight (normalized conductance units)")
        ax.legend()
        fig.tight_layout()
        plt.show(block=False)
        plt.pause(5)
        plt.close(fig)

    return candidates, failures


# The only callables a .fit file may invoke: the supported device classes, their nested
# DriftParameter, and numpy scalar wrappers (device reprs write e.g. np.float64(...)).
_FIT_FILE_CALLABLES = {**DEVICE_CLASSES, "DriftParameter": DriftParameter}
_FIT_FILE_NP_ATTRS = ("float64", "float32", "int64", "int32")


def _safe_eval(node):
    '''Evaluates the AST of a .fit file without running arbitrary code (unlike eval()):
    only literals, +/- signs, lists/tuples/dicts and calls to the whitelisted device
    classes above are allowed -- anything else raises.

    Args:
        node (ast.AST): node to evaluate

    Raises:
        ValueError: on any syntax outside the whitelist

    Returns:
        any: the evaluated value
    '''
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (int, float, bool, str)):
            return node.value
        raise ValueError(f"constant {node.value!r} not allowed in .fit files")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(element) for element in node.elts]
    if isinstance(node, ast.Dict):
        return {_safe_eval(key): _safe_eval(value) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FIT_FILE_CALLABLES:
            target = _FIT_FILE_CALLABLES[func.id]
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and func.value.id == "np" and func.attr in _FIT_FILE_NP_ATTRS):
            target = getattr(np, func.attr)
        else:
            raise ValueError(f"call not allowed in .fit files: {ast.unparse(func)}")
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError("**kwargs not allowed in .fit files")
        args = [_safe_eval(arg) for arg in node.args]
        kwargs = {keyword.arg: _safe_eval(keyword.value) for keyword in node.keywords}
        return target(*args, **kwargs)
    raise ValueError(f"syntax not allowed in .fit files: {type(node).__name__}")


def load_device(path):
    '''Loads a .fit file. Parsed through _safe_eval() -- a .fit file is data, not code,
    so nothing outside the whitelisted device constructors can run.

    Handles both formats: the current one, a dict holding the device plus the physical
    mapping ({"device": ..., "offset": <uS>, "scale": <uS per weight unit>}), and the
    legacy one, a bare device repr -- there the mapping was never saved, so offset and
    scale come back as None (weights cannot be mapped back to physical conductance).

    Args:
        path (str): path of the .fit file

    Returns:
        dict: {"device": device config, "offset": float | None, "scale": float | None}
    '''
    with open(path) as f:
        text = f.read()
    loaded = _safe_eval(ast.parse(text.strip(), mode="eval"))
    if isinstance(loaded, dict):
        return {"device": loaded["device"], "offset": loaded.get("offset"), "scale": loaded.get("scale")}
    return {"device": loaded, "offset": None, "scale": None}


def probe_device_response(device, num_pulses=50):
    '''Makes a LTP/LTD Simulation for displaying it out of a .fit file

    Note: if the device carries device-to-device variation (dw_min_dtod, w_min_dtod,
    w_max_dtod > 0), the 1x1 probe tile draws one random device instance per call --
    the preview curve then differs every time the same .fit is loaded. That is the
    variation working as configured, not data corruption.

    Args:
        device (SoftBoundsDevice): device 
        num_pulses (int, optional): number of pulses per LTD/LTP. Defaults to 50.

    Returns:
        ndarray: ltp_curve, ltd_curve
    '''
    rpu_config = SingleRPUConfig(device=device)
    rpu_config.update.desired_bl = 1
    rpu_config.update.fixed_bl = True
    w_min = device.w_min
    w_max = device.w_max
    if w_min is None:
        w_min = device.range_min
    if w_max is None:
        w_max = device.range_max

    ltp_tile = next(AnalogLinear(1, 1, rpu_config=rpu_config).analog_tiles())
    ltp_tile.set_weights(torch.full((1, 1), w_min))
    ltp_curve = compute_pulse_response(ltp_tile, np.ones(num_pulses))[:, 0, 0]

    ltd_tile = next(AnalogLinear(1, 1, rpu_config=rpu_config).analog_tiles())
    ltd_tile.set_weights(torch.full((1, 1), w_max))
    ltd_curve = compute_pulse_response(ltd_tile, -np.ones(num_pulses))[:, 0, 0]

    return ltp_curve, ltd_curve
