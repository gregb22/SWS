"""
run_RK4.py - C RK4 wrapper for the MDOF Duffing system.

This module runs the C integrator and returns:
    t_arr = np.frombuffer(_t, dtype=np.float64)[:actual_npts]
    y_arr = np.frombuffer(_y, dtype=np.float64).reshape(-1, nst)[:actual_npts]
"""

import ctypes
import os
import subprocess
import sys
import time
from typing import NamedTuple, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Nonlinear element type codes
# ---------------------------------------------------------------------------
NL_CUBIC     = 0   # params: (k,)           k * dq^3
NL_QUADRATIC = 1   # params: (k,)           k * dq * |dq|  (sign-preserving)
NL_QUINTIC   = 2   # params: (k,)           k * dq^5
NL_PIECEWISE = 3   # params: (gap, slope)   slope*(|dq|-gap)*sign(dq) if |dq|>gap
NL_HERTZ     = 4   # params: (k, exponent)  k*max(dq,0)^n, n=0 defaults to 1.5

MAX_NL_PARAMS = 4


class NonlinearElement(NamedTuple):
    """A nonlinear spring element between two DoFs (or a DoF and ground).

    dof_i   : first DoF index (0-based)
    dof_j   : second DoF index (0-based), or -1 for wall/ground
    nl_type : type code (use NL_* constants)
    params  : tuple of floats, length <= MAX_NL_PARAMS
    """
    dof_i:   int
    dof_j:   int
    nl_type: int
    params:  tuple


def build_nl_arrays(elements: List[NonlinearElement]):
    """Convert a list of NonlinearElement to flat ctypes-ready numpy arrays."""
    n = len(elements)
    if n == 0:
        zi = np.zeros(0, dtype=np.int32)
        zd = np.zeros(0, dtype=np.float64)
        return 0, zi, zi.copy(), zi.copy(), zd
    dof_i   = np.array([e.dof_i   for e in elements], dtype=np.int32)
    dof_j   = np.array([e.dof_j   for e in elements], dtype=np.int32)
    nl_type = np.array([e.nl_type for e in elements], dtype=np.int32)
    params  = np.zeros((n, MAX_NL_PARAMS), dtype=np.float64)
    for i, e in enumerate(elements):
        if len(e.params) > MAX_NL_PARAMS:
            raise ValueError(f"Element {i} has {len(e.params)} params; max is {MAX_NL_PARAMS}")
        params[i, :len(e.params)] = e.params
    return n, dof_i, dof_j, nl_type, params.ravel()


def _np_int_ptr(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.int32)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), arr


LIB_SRC = os.path.join(os.path.dirname(__file__), "rk4_lib.c")
if sys.platform == "win32":
    LIB_OUT = os.path.join(os.path.dirname(__file__), "rk4_lib.dll")
    COMPILE_CMD = ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", LIB_OUT, LIB_SRC, "-lm"]
else:
    LIB_OUT = os.path.join(os.path.dirname(__file__), "rk4_lib.so")
    COMPILE_CMD = ["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", LIB_OUT, LIB_SRC, "-lm"]


def _compile_if_needed() -> None:
    if not os.path.exists(LIB_OUT) or os.path.getmtime(LIB_SRC) > os.path.getmtime(LIB_OUT):
        print(f"Compiling {LIB_SRC} ...")
        result = subprocess.run(COMPILE_CMD, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\n{result.stderr}")
        print("Compilation successful.")


def _load_rk4_lib() -> ctypes.CDLL:
    _compile_if_needed()
    lib = ctypes.CDLL(LIB_OUT)

    _pi = ctypes.POINTER(ctypes.c_int)
    _pd = ctypes.POINTER(ctypes.c_double)
    lib.rk4_chirp.restype  = ctypes.c_int
    lib.rk4_chirp.argtypes = [
        ctypes.c_int,    # ndof
        _pd, _pd, _pd,   # M_inv, C, K
        _pd,             # fext
        ctypes.c_double, # f_amp
        ctypes.c_double, # f0
        ctypes.c_double, # f_end
        ctypes.c_double, # sweep_rate
        ctypes.c_double, # T
        ctypes.c_double, # ts
        ctypes.c_int,    # chirp_type (0=linear, 1=log, 2=constant)
        ctypes.c_int,    # num_nl
        _pi, _pi, _pi,   # nl_dof_i, nl_dof_j, nl_type
        _pd,             # nl_params
        ctypes.c_int,    # max_params
        _pd,             # y0 = [q0, qd0]
        _pd,             # t_out
        _pd,             # y_out
    ]
    return lib


def _np_ptr(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), arr


def run_rk4_chirp(
    M,
    C,
    K,
    fext,
    f_amp,
    f0,
    f1,
    T,
    ts,
    chirp_type,
    nl_elements: Optional[List[NonlinearElement]] = None,
    q0=None,
    qd0=None,
    *,
    verbose: bool = True,
):
    """Run the C RK4 solver and return (t_arr, y_arr).

    nl_elements : list of NonlinearElement describing all nonlinear connections.
                  Pass an empty list (or None) for a fully linear simulation.
    """
    if chirp_type not in (0, 1, 2):
        raise ValueError("chirp_type must be 0 (linear), 1 (log), or 2 (constant frequency)")

    M    = np.asarray(M,    dtype=np.float64)
    C    = np.asarray(C,    dtype=np.float64)
    K    = np.asarray(K,    dtype=np.float64)
    fext = np.asarray(fext, dtype=np.float64)

    ndof = len(M)
    nst  = 2 * ndof

    if q0 is None:
        q0 = np.zeros(ndof, dtype=np.float64)
    else:
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)

    if qd0 is None:
        qd0 = np.zeros(ndof, dtype=np.float64)
    else:
        qd0 = np.asarray(qd0, dtype=np.float64).reshape(-1)

    if q0.size != ndof or qd0.size != ndof:
        raise ValueError(f"q0 and qd0 must each have length {ndof}")

    y0 = np.concatenate((q0, qd0))

    if ndof == 1:
        M_inv = 1.0 / M
    else:
        M_inv = np.linalg.inv(M)

    if nl_elements is None:
        nl_elements = []

    num_nl, arr_di, arr_dj, arr_nt, arr_np = build_nl_arrays(nl_elements)

    sweep_rate = (f1 - f0) / T
    npts = int(T / ts) + 2

    t_out = np.empty(npts, dtype=np.float64)
    y_out = np.empty((npts, nst), dtype=np.float64)

    ptr_Minv, _Minv = _np_ptr(M_inv)
    ptr_C,    _C    = _np_ptr(C)
    ptr_K,    _K    = _np_ptr(K)
    ptr_fext, _fext = _np_ptr(fext)
    ptr_di,   _di   = _np_int_ptr(arr_di)
    ptr_dj,   _dj   = _np_int_ptr(arr_dj)
    ptr_nt,   _nt   = _np_int_ptr(arr_nt)
    ptr_np,   _np_  = _np_ptr(arr_np)
    ptr_y0,   _y0   = _np_ptr(y0)
    ptr_t,    _t    = _np_ptr(t_out)
    ptr_y,    _y    = _np_ptr(y_out.ravel())

    lib = _load_rk4_lib()

    t0 = time.perf_counter()
    actual_npts = lib.rk4_chirp(
        ctypes.c_int(ndof),
        ptr_Minv,
        ptr_C,
        ptr_K,
        ptr_fext,
        ctypes.c_double(float(f_amp)),
        ctypes.c_double(float(f0)),
        ctypes.c_double(float(f1)),
        ctypes.c_double(float(sweep_rate)),
        ctypes.c_double(float(T)),
        ctypes.c_double(float(ts)),
        ctypes.c_int(int(chirp_type)),
        ctypes.c_int(num_nl),
        ptr_di,
        ptr_dj,
        ptr_nt,
        ptr_np,
        ctypes.c_int(MAX_NL_PARAMS),
        ptr_y0,
        ptr_t,
        ptr_y,
    )
    t1 = time.perf_counter()

    if actual_npts <= 0:
        raise RuntimeError("rk4_chirp returned no samples")

    if verbose:
        print(f"RK4 (C, {ndof}-DOF) took {t1 - t0:.4f} s ({actual_npts} points)")

    t_arr = np.frombuffer(_t, dtype=np.float64)[:actual_npts]
    y_arr = np.frombuffer(_y, dtype=np.float64).reshape(-1, nst)[:actual_npts]

    return t_arr, y_arr
