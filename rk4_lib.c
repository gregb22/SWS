/*
 * rk4_lib.c — Shared library: RK4 MDOF Duffing chirp sweep
 *
 * All system parameters are passed as arguments.
 * Python pre-allocates the output arrays and passes pointers.
 *
 * Compile
 *   gcc -O3 -march=native -shared -fPIC -o rk4_lib.dll rk4_lib.c -lm
 */

#include <math.h>
#include <string.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT
#endif

#define PI 3.14159265358979323846

/* ======================================================================
 * Internal helpers (runtime ndof)
 * ====================================================================== */

static inline void matvec_n(const double *A, const double *v, double *dst, int n)
{
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += A[i*n + j] * v[j];
        dst[i] = s;
    }
}

static inline double const_frequency(double t, double freq)
{
    /* Constant frequency: phi(t) = 2*pi*f*t */
    return 2.0 * PI * freq * t;
}

static inline double chirp_phase(double t, double freq0, double sr)
{
    /* Linear chirp: phi(t) = 2*pi*(f0*t + 0.5*sr*t^2) */
    return 2.0 * PI * (freq0 * t + 0.5 * sr * t * t);
}

static inline double log_chirp_phase(double t, double freq0, double f_end, double duration)
{
    /* Log chirp: phi(t) = 2*pi*f0*(T/ln(f1/f0))*(exp(ln(f1/f0)*t/T) - 1) */
    return 2.0 * PI * freq0 * duration / log(f_end / freq0) * (exp(log(f_end / freq0) * t / duration) - 1.0);
}

/* ======================================================================
 * Nonlinear force dispatch
 * Type codes: 0=cubic, 1=quadratic(sign-preserving), 2=quintic,
 *             3=piecewise-linear(dead-band), 4=Hertz-contact(one-sided)
 * ====================================================================== */
static inline double compute_nl_force(double dq, int type, const double *p)
{
    switch (type) {
        case 0: return p[0] * dq * dq * dq;
        case 1: return p[0] * dq * dq;
        case 2: { double d2 = dq * dq; return p[0] * dq * d2 * d2; }
        case 3:
            if (dq >  p[0]) return p[1] * (dq - p[0]);
            if (dq < -p[0]) return p[1] * (dq + p[0]);
            return 0.0;
        case 4:
            if (dq <= 0.0) return 0.0;
            return p[0] * pow(dq, p[1] > 0.0 ? p[1] : 1.5);
        default: return 0.0;
    }
}

static void fun_mdof(double t, const double *y, double *dy, int ndof,
                     const double *M_inv, const double *C, const double *K,
                     const double *fext, double f_amp,
                     double freq0, double sr, int chirp_type, double f_end, double duration,
                     int num_nl, const int *nl_dof_i, const int *nl_dof_j,
                     const int *nl_type, const double *nl_params, int max_params,
                     double *rhs, double *Cqd, double *Kq, double *qdd)
{
    const double *q  = y;
    const double *qd = y + ndof;

    matvec_n(C, qd, Cqd, ndof);
    matvec_n(K, q,  Kq,  ndof);

    double phase;
    if (chirp_type == 0) {
        phase = chirp_phase(t, freq0, sr);
    } else if (chirp_type == 1) {
        phase = log_chirp_phase(t, freq0, f_end, duration);
    } else if (chirp_type == 2) {
        phase = const_frequency(t, freq0);
    } else {
        /* Fallback to linear chirp for invalid mode values. */
        phase = chirp_phase(t, freq0, sr);
    }
    double sinw = sin(phase);

    for (int i = 0; i < ndof; i++)
        rhs[i] = fext[i] * f_amp * sinw - Cqd[i] - Kq[i];

    for (int e = 0; e < num_nl; e++) {
        int di = nl_dof_i[e], dj = nl_dof_j[e];
        double dq = (dj >= 0) ? (q[di] - q[dj]) : q[di];
        double f_nl = compute_nl_force(dq, nl_type[e], nl_params + (long)e * max_params);
        rhs[di] -= f_nl;
        if (dj >= 0) rhs[dj] += f_nl;
    }

    matvec_n(M_inv, rhs, qdd, ndof);

    for (int i = 0; i < ndof; i++) {
        dy[i]        = qd[i];
        dy[ndof + i] = qdd[i];
    }
}

EXPORT int rk4_chirp(int ndof,
                     const double *M_inv, const double *C, const double *K,
                     const double *fext, double f_amp,
                     double f0, double f_end, double sweep_rate, double T, double ts,
                     int chirp_type,
                     int num_nl,
                     const int *nl_dof_i, const int *nl_dof_j,
                     const int *nl_type,
                     const double *nl_params, int max_params,
                     const double *y0,
                     double *t_out, double *y_out)
{
    int nst  = 2 * ndof;
    long n   = (long)(T / ts) + 1;
    long npts = n + 1;

    /* VLA scratch buffers on the stack (C99, safe for small ndof) */
    double rhs[ndof],  Cqd[ndof], Kq[ndof], qdd[ndof];
    double k1[nst], k2[nst], k3[nst], k4[nst], ytmp[nst];

    /* Initial condition: user-specified y0 if provided, otherwise zeros */
    t_out[0] = 0.0;
    if (y0) {
        memcpy(y_out, y0, nst * sizeof(double));
    } else {
        memset(y_out, 0, nst * sizeof(double));
    }

    double h2 = ts / 2.0;

    for (long i = 0; i < n; i++) {
        double xi        = t_out[i];
        const double *yi = &y_out[i * nst];

        /* k1 */
        fun_mdof(xi, yi, k1, ndof, M_inv, C, K, fext, f_amp, f0, sweep_rate,
                 chirp_type, f_end, T,
                 num_nl, nl_dof_i, nl_dof_j, nl_type, nl_params, max_params,
                 rhs, Cqd, Kq, qdd);

        /* k2 */
        for (int j = 0; j < nst; j++) ytmp[j] = yi[j] + h2 * k1[j];
        fun_mdof(xi + h2, ytmp, k2, ndof, M_inv, C, K, fext, f_amp, f0, sweep_rate,
                 chirp_type, f_end, T,
                 num_nl, nl_dof_i, nl_dof_j, nl_type, nl_params, max_params,
                 rhs, Cqd, Kq, qdd);

        /* k3 */
        for (int j = 0; j < nst; j++) ytmp[j] = yi[j] + h2 * k2[j];
        fun_mdof(xi + h2, ytmp, k3, ndof, M_inv, C, K, fext, f_amp, f0, sweep_rate,
                 chirp_type, f_end, T,
                 num_nl, nl_dof_i, nl_dof_j, nl_type, nl_params, max_params,
                 rhs, Cqd, Kq, qdd);

        /* k4 */
        for (int j = 0; j < nst; j++) ytmp[j] = yi[j] + ts * k3[j];
        fun_mdof(xi + ts, ytmp, k4, ndof, M_inv, C, K, fext, f_amp, f0, sweep_rate,
                 chirp_type, f_end, T,
                 num_nl, nl_dof_i, nl_dof_j, nl_type, nl_params, max_params,
                 rhs, Cqd, Kq, qdd);

        /* accumulate */
        double *yi1 = &y_out[(i + 1) * nst];
        for (int j = 0; j < nst; j++)
            yi1[j] = yi[j] + ts * (k1[j] + 2.0*k2[j] + 2.0*k3[j] + k4[j]) / 6.0;
        t_out[i + 1] = xi + ts;
    }

    return (int)npts;
}
