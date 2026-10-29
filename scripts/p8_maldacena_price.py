"""Arc-D core calculation: the price of Maldacena's ghost-removal
boundary condition, per mode, in the process currency.

Model (declared): detuned Pais-Uhlenbeck mode, omega_1, omega_2 =
omega_1 - delta, as the proxy for the dS-split graviton pair
(delta <-> the Delta=3/Delta=2 splitting; Minkowski = delta -> 0).
Ostrogradski variables q1 = x, q2 = xdot are BOTH configuration
variables (the two-slice memory), so the Neumann effect is the
multiplication operator M_sigma = exp(-q2^2/(2 sigma^2)) -- a bona
fide effect, 0 <= M <= 1; Maldacena's positive-frequency condition
is the STATE PREP (the vacuum below), not part of M.

Quantization declared (caveat 2 of the scoping report): the
DIRAC-NORMALIZABLE ground state on real configurations (the beables
currency). Gaussian ansatz Psi_0 = exp(-(a q1^2 + b q2^2)/2
- i c q1 q2) solves H Psi = E Psi exactly with
    c = omega_1 omega_2,  b = delta,  a = omega_1 omega_2 delta,
    E_0 = delta/2,
so |Psi_0|^2 has widths ~ delta^{-1/2} in BOTH variables: the
exceptional-point degeneration, quantitative. (The Bender-Mannheim
root b = omega_1+omega_2 has a < 0: normalizable only under contour
rotation -- the other leg of the left/right/silence taxonomy; not
used here.)

Closed forms tested:
 [2] survival p(sigma, delta) = [1 + 1/(2 sigma^2 delta)]^{-1/2}
     -> sigma sqrt(2 delta) as sigma -> 0.
 [3] conditioned graviton wavefunction (sigma -> 0): Gaussian of
     width 1/sqrt(a), NOT the Einstein vacuum width 1/sqrt(omega_1):
     fidelity F = 2 sqrt(a omega_1)/(a + omega_1) ~ 2 sqrt(omega_2
     delta / omega_1) -> 0 like delta^{1/2} (amplitude ~ delta^{1/4}).
 [4] price sheet: ghost-removal costs survival ~ sigma sqrt(delta)
     and delivers Einstein-fidelity ~ sqrt(delta): BOTH collapse at
     the conformal point -- the HLZ Minkowski obstruction, priced.

Grid numerics verify every closed form (H Psi residual, p, F).
ASCII only; numpy.
"""

import numpy as np

W1 = 1.0


def ground(delta):
    w2 = W1 - delta
    a = W1 * w2 * delta
    b = delta
    c = W1 * w2
    return a, b, c


def psi0_grid(delta, q1, q2):
    a, b, c = ground(delta)
    Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
    return np.exp(-0.5 * (a * Q1 ** 2 + b * Q2 ** 2)
                  - 1j * c * Q1 * Q2)


def H_apply(psi, delta, q1, q2):
    """H = -i q2 d/dq1 - (1/2) d^2/dq2^2 + (w1^2+w2^2) q2^2/2
           - w1^2 w2^2 q1^2 / 2   (Ostrogradski PU)."""
    w2 = W1 - delta
    d1 = np.gradient(psi, q1, axis=0)
    d2 = np.gradient(np.gradient(psi, q2, axis=1), q2, axis=1)
    Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
    return (-1j * Q2 * d1 - 0.5 * d2
            + 0.5 * (W1 ** 2 + w2 ** 2) * Q2 ** 2 * psi
            - 0.5 * W1 ** 2 * w2 ** 2 * Q1 ** 2 * psi)


def main():
    print("=" * 72)
    print("Arc-D: pricing Maldacena's boundary condition "
          "(detuned-PU proxy)")
    print("=" * 72)

    # ---- [1] ground state + degeneration -------------------------------
    print("\n[1] Dirac-normalizable vacuum:")
    # algebraic verification of the eigen-equations (exact; a grid
    # check at small delta under-resolves the e^{-i c q1 q2} phase
    # and fails on its own finite differences -- first run of this
    # script hit that at delta=0.1, residual 4.2: numerics artifact,
    # kept in git history)
    print("  coefficient-equation residuals (must be 0):")
    for delta in (0.5, 0.1, 0.01):
        w2 = W1 - delta
        a, b, c = ground(delta)
        r1 = abs(-c ** 2 / 2 + 0.5 * W1 ** 2 * w2 ** 2)   # q1^2
        r2 = abs(c - b ** 2 / 2 + 0.5 * (W1 ** 2 + w2 ** 2)
                 - c - (W1 ** 2 + w2 ** 2) / 2 + b ** 2 / 2)  # identity
        r2 = abs(-b ** 2 / 2 + 0.5 * (W1 ** 2 + w2 ** 2) - c)  # q2^2
        r3 = abs(a - b * c / (W1 * w2) * (W1 * w2))  # q1 q2: a = b c
        r3 = abs(a - b * c)
        print("  delta=%.2f : q1^2 %.1e ; q2^2 %.1e ; q1q2 %.1e ; "
              "E0 = delta/2 = %.3f" % (delta, r1, r2, r3, delta / 2))
    # one well-resolved grid sanity check
    delta = 0.5
    a, b, c = ground(delta)
    q1 = np.linspace(-5 / np.sqrt(a), 5 / np.sqrt(a), 1201)
    q2 = np.linspace(-5 / np.sqrt(b), 5 / np.sqrt(b), 1201)
    psi = psi0_grid(delta, q1, q2)
    Hp = H_apply(psi, delta, q1, q2)
    m = (np.abs(psi) > 0.1 * np.abs(psi).max())
    m[:100], m[-100:], m[:, :100], m[:, -100:] = 0, 0, 0, 0
    E = np.real((Hp[m] / psi[m]).mean())
    print("  grid sanity (delta=0.5, 1201^2): E0 = %.6f vs %.6f"
          % (E, delta / 2))
    print("  widths: |Psi|^2 ~ exp(-w1 w2 delta q1^2 - delta q2^2) "
          "-- both ~ delta^{-1/2}")
    print("  (the exceptional-point degeneration, quantitative)")

    # ---- [2] survival price --------------------------------------------
    print("\n[2] survival p(sigma, delta) vs closed form "
          "[1+1/(2 s^2 d)]^{-1/2}:")
    for delta in (0.5, 0.1, 0.02):
        b = delta
        for sig in (0.3, 0.05):
            L2 = 6 / np.sqrt(min(b, 1 / sig ** 2))
            q2 = np.linspace(-max(L2, 6 / np.sqrt(b)),
                             max(L2, 6 / np.sqrt(b)), 4001)
            w = np.exp(-b * q2 ** 2)
            p_num = np.trapezoid(w * np.exp(-q2 ** 2 / (2 * sig ** 2)),
                                 q2) / np.trapezoid(w, q2)
            p_cf = 1 / np.sqrt(1 + 1 / (2 * sig ** 2 * delta))
            print("  delta=%.2f sigma=%.2f : p = %.6f  closed form "
                  "%.6f  (sigma->0: sigma sqrt(2 delta) = %.4f)"
                  % (delta, sig, p_num, p_cf,
                     sig * np.sqrt(2 * delta)))

    # ---- [3] delivered state: fidelity to the Einstein vacuum ----------
    print("\n[3] conditioned graviton state vs Einstein vacuum "
          "(sigma -> 0):")
    print("  F = 2 sqrt(a w1)/(a + w1), a = w1 w2 delta:")
    for delta in (0.5, 0.1, 0.02, 0.004):
        a, _, _ = ground(delta)
        F = 2 * np.sqrt(a * W1) / (a + W1)
        print("  delta=%.3f : F = %.4f   (2 sqrt(w2 delta) = %.4f)"
              % (delta, F, 2 * np.sqrt((W1 - delta) * delta)))
    ds = np.array([0.02, 0.004, 0.0008])
    Fs = [2 * np.sqrt(ground(d)[0] * W1) / (ground(d)[0] + W1)
          for d in ds]
    slope = np.polyfit(np.log(ds), np.log(Fs), 1)[0]
    print("  scaling: F ~ delta^%.3f (predict 1/2)" % slope)
    # finite-sigma cross-check on the grid (mixed reduced state)
    delta, sig = 0.1, 0.05
    a, b, c = ground(delta)
    q1 = np.linspace(-6 / np.sqrt(a), 6 / np.sqrt(a), 301)
    q2 = np.linspace(-6 / np.sqrt(b), 6 / np.sqrt(b), 301)
    psi = psi0_grid(delta, q1, q2) * np.exp(
        -np.meshgrid(q1, q2, indexing="ij")[1] ** 2 / (4 * sig ** 2))
    rho = psi @ psi.conj().T * (q2[1] - q2[0])
    rho /= np.trapezoid(np.diag(rho).real, q1)
    phi = (W1 / np.pi) ** 0.25 * np.exp(-W1 * q1 ** 2 / 2)
    Fnum = np.real(phi @ rho @ phi) * (q1[1] - q1[0]) ** 2
    print("  grid check (delta=0.1, sigma=0.05, mixed rho): "
          "F = %.4f vs sigma->0 form %.4f"
          % (Fnum, 2 * np.sqrt(a * W1) / (a + W1)))

    # ---- [4] the price sheet -------------------------------------------
    print("\n[4] PRICE SHEET (sigma = 0.05):")
    print("  delta     p(survival)   F(Einstein delivery)")
    for delta in (0.5, 0.1, 0.02, 0.004):
        a, _, _ = ground(delta)
        p = 1 / np.sqrt(1 + 1 / (2 * 0.05 ** 2 * delta))
        F = 2 * np.sqrt(a * W1) / (a + W1)
        print("  %.3f     %.5f       %.4f" % (delta, p, F))
    print("\nverdict: ghost-removal by future boundary condition is a")
    print("survival-conditioned effect whose price DIVERGES at the")
    print("conformal point: p ~ sigma sqrt(2 delta) -> 0 AND the")
    print("delivered state's Einstein fidelity F ~ 2 sqrt(w2 delta)")
    print("-> 0. dS works because delta = O(1) there; Minkowski is")
    print("the delta -> 0 boundary: the HLZ obstruction, priced.")
    print("done.")


if __name__ == "__main__":
    main()
