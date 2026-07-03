import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def LMS_numba(
    freq,
    amp,
    time,
    M=1,
    mu=0.2,
    varying_mu=False,
    use_warmup_init=False,
    warmup_freq_max=5.0,
    warmup_epochs=1,
):
    """
    Numba-optimized NLMS adaptive filter for harmonic estimation.
    """
    N = len(freq)
    n_weights = 2 * M + 1
    w = np.zeros(n_weights)
    Z = np.zeros((n_weights, N))
    E = np.zeros(N)

    # Pre-compute phase via trapezoidal integration
    phase = np.zeros(N)
    if N > 1:
        dt0 = time[1] - time[0]
        phase[0] = freq[0] * dt0
        for i in range(1, N):
            dt = time[i] - time[i-1]
            phase[i] = phase[i-1] + freq[i] * dt
    
    # Pre-allocate basis vector
    u = np.zeros(n_weights)
    eps = 1e-6
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    # Optional warm-up pass on low-frequency samples to get better initial weights.
    if use_warmup_init and N > 1:
        n_warmup = 0
        for i in range(N):
            if freq[i] <= warmup_freq_max:
                n_warmup += 1
            else:
                break

        if n_warmup > 1:
            for _ in range(warmup_epochs):
                for i in range(n_warmup):
                    u[0] = inv_sqrt2
                    for m in range(1, M + 1):
                        phi_m = m * phase[i]
                        u[2*m - 1] = np.sin(phi_m)
                        u[2*m] = np.cos(phi_m)

                    y = 0.0
                    for k in range(n_weights):
                        y += w[k] * u[k]
                    e = amp[i] - y

                    norm_u2 = eps
                    for k in range(n_weights):
                        norm_u2 += u[k] * u[k]

                    if varying_mu:
                        if i == 0:
                            ts = time[1] - time[0]
                        else:
                            ts = time[i] - time[i - 1]

                        theta = freq[i] * ts
                        s = np.sin(theta)
                        c = np.cos(theta)
                        den = c * c

                        if den > 1e-10:
                            mu_i = 2.0 * s * (1.0 - s) / den
                            if mu_i < 0.0:
                                mu_i = 0.0
                        else:
                            mu_i = mu
                    else:
                        mu_i = mu

                    scale = mu_i * e / norm_u2
                    for k in range(n_weights):
                        w[k] += scale * u[k]
    
    for i in range(N):
        # Build basis vector
        u[0] = inv_sqrt2  # DC component
        for m in range(1, M + 1):
            phi_m = m * phase[i]
            u[2*m - 1] = np.sin(phi_m)
            u[2*m]     = np.cos(phi_m)
        
        # Prediction and error
        y = 0.0
        for k in range(n_weights):
            y += w[k] * u[k]
        e = amp[i] - y
        E[i] = e
        
        # NLMS update
        norm_u2 = eps
        for k in range(n_weights):
            norm_u2 += u[k] * u[k]
        
        if varying_mu:
            # Abeloos varying learning rate using theta = omega * t_s.
            if i == 0:
                ts = time[1] - time[0] if N > 1 else 0.0
            else:
                ts = time[i] - time[i - 1]

            theta = freq[i] * ts
            s = np.sin(theta)
            c = np.cos(theta)
            den = c * c

            if den > 1e-10:
                mu_i = 2.0 * s * (1.0 - s) / den
                if mu_i < 0.0:
                    mu_i = 0.0
            else:
                mu_i = mu
        else:
            mu_i = mu

        scale = mu_i * e / norm_u2
        for k in range(n_weights):
            w[k] += scale * u[k]
            Z[k, i] = w[k]
            
    return Z, E

@numba.jit(nopython=True, cache=True, parallel=True)
def Sampling_tot_amplitude_numba(z, n_period=128):
    """
    Numba-optimized total amplitude sampling.
    
    Parameters:
    -----------
    z : array (2*M+1, N)
        Weight history from LMS
    n_period : int
        Number of points to sample over one period
    
    Returns:
    --------
    a : array (N,)
        Peak amplitude at each time step
    """
    Nh = (z.shape[0] - 1) // 2
    m = z.shape[1]
    a = np.zeros(m)
    
    # Pre-compute basis: sin/cos for harmonics 1..Nh over one period
    period = np.linspace(0.0, 2.0 * np.pi, n_period)
    sines = np.zeros((n_period, Nh))
    cosines = np.zeros((n_period, Nh))
    for j in range(Nh):
        harmonic = j + 1
        for p in range(n_period):
            sines[p, j] = np.sin(harmonic * period[p])
            cosines[p, j] = np.cos(harmonic * period[p])
    
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    
    # Parallel loop over columns
    for ii in numba.prange(m):
        # Reconstruct signal and find max
        max_val = 0.0
        for p in range(n_period):
            val = z[0, ii] * inv_sqrt2
            for j in range(Nh):
                val += sines[p, j] * z[2*j + 1, ii]  # sin coeffs at odd indices
                val += cosines[p, j] * z[2*j + 2, ii]  # cos coeffs at even indices
            abs_val = abs(val)
            if abs_val > max_val:
                max_val = abs_val
        a[ii] = max_val
    
    return a


def poincare_map(t, y, T):
    """
    Compute Poincaré map by sampling the state at intervals of T.
    
    Parameters:
    -----------
    t : array (N,)
        Time vector
    y : array (N, n)
        State vector (displacement and velocity)
    T : float
        Sampling period for Poincaré map
    
    Returns:
    --------
    poincare_points : array (M, n)
        State at each Poincaré section crossing
    """
    poincare_points = []
    next_time = T
    for i in range(len(t)):
        if t[i] >= next_time:
            poincare_points.append(y[i])
            next_time += T
    return np.array(poincare_points)


