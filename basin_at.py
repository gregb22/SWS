import numpy as np

"""Compute the basin of attraction for a dynamical system (parallel version)."""

import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from run_RK4 import run_rk4_chirp, NonlinearElement, NL_CUBIC, NL_HERTZ, NL_PIECEWISE, NL_QUADRATIC


def steady_state_amplitude_fast(M, C, K, fext, nl_elements, x0, v0, freq, forcing,
                                n_periods_total=1000,
                                n_periods_window=100):
    T = 1.0 / freq # freq in hz
    t_end = n_periods_total * T # time in seconds for n_periods_total
    ts = 1.0 / (100.0 / (2.0 * np.pi)) # sampling time in seconds
    chirp_type = 2
    t, sol = run_rk4_chirp(M, C, K, fext, forcing, freq, freq, t_end, ts, chirp_type,
                           nl_elements=nl_elements, q0=x0, qd0=v0, verbose=False)

    x = sol[:, 0] # assuming x is the first state variable

    # Keep last window only
    t_window_start = t_end - n_periods_window * T
    mask = t >= t_window_start
    x_win = x[mask]

    return np.max(np.abs(x_win))


# --- Worker function for multiprocessing -----------------------------------

def worker_task(args):
    """
    Runs one simulation for one (x0, v0) point.
    We re-import / reuse the system inside the worker to avoid pickling issues.
    """
    i, j, M, C, K, fext, nl_elements, x0, v0, freq, forcing, n_periods_total, n_periods_window = args

    # IMPORTANT: use sys_global imported at module level (picklable or not)
    # If your sys object is not picklable, you can rebuild it here instead.
    A = steady_state_amplitude_fast(M, C, K, fext, nl_elements, x0, v0, freq, forcing,
                                    n_periods_total=n_periods_total,
                                    n_periods_window=n_periods_window)
    return i, j, A


# --- Main script -----------------------------------------------------------

if __name__ == "__main__":

    # -------------- System parameters 
    M = np.array([1])
    C = np.array([0.05])
    K = np.array([1])

    #Parameters for PWL : (gap, slopes)
    nl_elements = [NonlinearElement(dof_i=0, dof_j=-1, nl_type=NL_CUBIC, params=(1.0,)), NonlinearElement(dof_i=0, dof_j=-1, nl_type=NL_QUADRATIC, params=(0.2,))]
    fext  = np.array([1.0]) # Uncomment for 1-DOF Duffing oscillator
    f_amp = 1.0 # Forcing amplitude (N)

    # ---------------------------
    # Basin computation parameters
    # ---------------------------
    om = 0.46 # rad/s
    freq = om/(2.0 * np.pi)  # Hz
    Nx0 = 300
    Nv0 = 300

    grid_x0 = np.linspace(-10, 10, Nx0)
    grid_v0 = np.linspace(-10, 10, Nv0)

    basin = np.zeros((Nx0, Nv0))

    # Simulation parameters for steady_state_amplitude_fast
    n_periods_total = 1000
    n_periods_window = 100

    # Build job list
    jobs = []
    for i, x0 in enumerate(grid_x0):
        for j, v0 in enumerate(grid_v0):
            jobs.append((i, j, M, C, K, fext, nl_elements, x0, v0, freq, f_amp, n_periods_total, n_periods_window))

    total_points = len(jobs)
    print(f"Starting basin computation: {total_points} points\n")

    # ---------------------------------------
    # Parallel execution with ProcessPoolExecutor
    # ---------------------------------------
    start_time = time.time()
    counter = 0

    max_workers = 32  # <- number of processes in parallel

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_task, job) for job in jobs]

        for fut in as_completed(futures):
            i, j, A = fut.result()
            basin[i, j] = A

            counter += 1
            elapsed = time.time() - start_time
            points_per_sec = counter / elapsed if elapsed > 0 else 0.0
            remaining = total_points - counter
            eta = remaining / points_per_sec if points_per_sec > 0 else np.inf

            x0 = grid_x0[i]
            v0 = grid_v0[j]
            if counter % 1000 == 0 or counter == total_points:
                print(
                    f"[{counter:4d}/{total_points}]  "
                    f"x0={x0:.2f}, v0={v0:.2f}   |   "
                    f"elapsed={elapsed:6.1f}s  "
                    f"eta={eta:6.1f}s  "
                    f"speed={points_per_sec:5.2f} pts/s"
                )

    # ---------------------------------------
    # Plot basin
    # ---------------------------------------

    X0, V0 = np.meshgrid(grid_x0, grid_v0, indexing='ij')

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X0, V0, basin, shading='auto', cmap='viridis')
    plt.colorbar(label='Amplitude en régime permanent')
    plt.xlabel('Position initiale $x_0$')
    plt.ylabel('Vitesse initiale $v_0$')
    plt.title(f'Bassin d\'attraction pour une excitation à {freq} Hz')
    plt.tight_layout()
    plt.show()

    total_time = time.time() - start_time
    print(f"\nTotal computation time : {total_time:.1f} seconds")