"""Pricing M': the Hell-Lust-Zoupanos (HLZ, arXiv:2306.13714) alternative
future boundary conditions, as a survival-conditioned effect on the same
fourth-order dS mode used for Maldacena's M (see p8_maldacena_price.py).

IMPORTANT CITATION CORRECTION: arXiv:2306.13714 is "On the Ghost Problem
of Conformal Gravity" by A. Hell, D. Lust, G. Zoupanos (HLZ = Hell-Lust-
Zoupanos), NOT Hertzberg-Loeb-Zhang.

HLZ structure (confirmed, dS Section 3.2; eq. numbers as printed):
  (28) general tensor mode: h_k = A e^{ikn} + B e^{-ikn}
                                  + n[C e^{ikn} + D e^{-ikn}]
       -- the n-linear C,D terms are the GHOST modes (secular growth).
  (30) Neumann BC:       h_dot|_{n=0} = 0
  (59) second/novel BC:  h_ddot|_{n=0} = 2(R_ij - 1/4 g_ij R)|_0
                          -> in dS this is  h_ddot_k|_0 = k^2 h_k|_0.
  (66) together give     C = -ik A,  D = ik B
  (67) general solution  h_k = A(1-ikn)e^{ikn} + B(1+ikn)e^{-ikn}
       -- the GENERAL (2-parameter) Einstein-with-Lambda tensor mode.
  Key contrast with Maldacena: HLZ impose TWO future conditions and NO
  in-state (positive-frequency) restriction, retaining BOTH A and B.
  (The task's "~56/64/67" map to the printed (59)/(66)/(67).)

TRANSLATION to the detuned-PU proxy (same conventions as the Maldacena
script: q1 = x = h, q2 = xdot = h_dot; a = w1 w2 delta, b = delta,
c = w1 w2; ground state Psi0 = exp(-(a q1^2 + b q2^2)/2 - i c q1 q2)):

  * BC (30)  h_dot = 0        -> q2 = 0.  Soft effect P1 = exp(-q2^2/2s^2).
             THIS IS EXACTLY MALDACENA'S NEUMANN M.
  * BC (59)  x_ddot = -w1^2 x (healthy Einstein dispersion; conformal-time
             sign flips to +k^2). With x_ddot = p2 = -i d/dq2 (Ostrogradski
             velocity of q2), this is the phase-space variable
                 O2 = x_ddot + w1^2 x = p2 + w1^2 q1,
             constrained to 0.  Soft effect P2 = exp(-O2^2/2t^2).
     In normal-mode language O2 = 0 is exactly "kill the ghost coordinate
     y2": for x = y1 + y2 with y1_dd=-w1^2 y1, y2_dd=-w2^2 y2, the
     condition x_dd = -w1^2 x forces (w1^2-w2^2) y2 = 0.

So M' = M x (second condition).  KEY POINT: in the Ostrogradski PU the
momentum conjugate to Q2 = xdot is P2 = xddot, so BC(30) (fix xdot) and
BC(59) (fix xddot) constrain a CONJUGATE PAIR: [q2, O2] = [q2, p2] = i.
M' is thus a joint position+momentum (Heisenberg-limited) effect on the
velocity sector -- qualitatively distinct from M (one quadrature only).

Closed forms derived here (verified below):
  Var_Psi0(O2) = (delta/2)(w1+w2)/w2 -> delta        (VANISHES as d->0)
  p1(s,d) = [1 + 1/(2 s^2 d)]^{-1/2}   (= Maldacena)  -> s sqrt(2 d)
  p2(t,d) = [1 + Var(O2)/t^2]^{-1/2}                  -> 1  (nearly FREE)
  cov_Psi0(q2, O2) = 0, but [q2,O2]=i  =>  the survival PREFACTOR is
    ORDERING-dependent:
      Weyl-symbol effect  : p_M' = p1 * p2      -> p1  (exact; p2->1)
      product/Kraus effect: p_M' = 2 s t * p1   (Maldacena's p1 times the
                            Heisenberg cell 2 s t; this is the
                            sub-minimal-cell s*t -> 0 limit)
    BOTH give the SAME delta-scaling  p_M' ~ s sqrt(2 delta).

Punchline: M' costs the SAME leading survival power as M (~ s sqrt(delta));
BC(59) adds only an O(1) (ordering/Heisenberg) surcharge and is
asymptotically free in scale. The delta->0 collapse is NOT evaded.
Fidelity of the delivered state is checked numerically. ASCII only; numpy.
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
    return np.exp(-0.5 * (a * Q1 ** 2 + b * Q2 ** 2) - 1j * c * Q1 * Q2)


def var_O2_closed(delta):
    w2 = W1 - delta
    return 0.5 * delta * (W1 + w2) / w2


def p2_spectral(psi, q2):
    """p2 psi = -i d/dq2 psi, via FFT (exact for the e^{-i c q1 q2} phase).
    numpy convention: psi_n = (1/N) sum_k A_k e^{+i kap_n x}, so p2 = -i d/dx
    has eigenvalue +kap = +2 pi fftfreq. (This fixes the projector sign.)"""
    dq2 = q2[1] - q2[0]
    kap = 2 * np.pi * np.fft.fftfreq(q2.size, d=dq2)
    return np.fft.ifft(kap * np.fft.fft(psi, axis=1), axis=1)


# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Pricing M': HLZ (Hell-Lust-Zoupanos) two-future-condition")
    print("ghost removal, on the detuned-PU dS mode.")
    print("=" * 72)

    # ---- [1] Var(O2): the second-condition variable ---------------------
    print("\n[1] Var_Psi0(O2), O2 = x_ddot + w1^2 x = p2 + w1^2 q1")
    print("    closed form (delta/2)(w1+w2)/w2  vs grid (p2 via d/dq2):")
    for delta in (0.5, 0.1, 0.02):
        a, b, c = ground(delta)
        q1 = np.linspace(-8 / np.sqrt(a), 8 / np.sqrt(a), 401)
        L2 = max(8 / np.sqrt(b), 8.0)
        q2 = np.linspace(-L2, L2, 2048, endpoint=False)
        psi = psi0_grid(delta, q1, q2)
        Q1, _ = np.meshgrid(q1, q2, indexing="ij")
        # O2 psi = (p2 + w1^2 q1) psi, p2 spectral; <O2^2>=||O2 psi||^2/||psi||^2
        O2psi = p2_spectral(psi, q2) + W1 ** 2 * Q1 * psi
        num = np.trapezoid(np.trapezoid(np.abs(O2psi) ** 2, q2, axis=1), q1)
        den = np.trapezoid(np.trapezoid(np.abs(psi) ** 2, q2, axis=1), q1)
        print("  delta=%.2f : grid %.5f   closed %.5f   (delta=%.3f)"
              % (delta, num / den, var_O2_closed(delta), delta))
    print("  -> Var(O2) ~ delta -> 0 : the state already nearly obeys BC(59)")

    # ---- [2] covariance of the two constrained variables ----------------
    print("\n[2] cov_Psi0(q2, O2) (must be 0 -> survivals factorize):")
    for delta in (0.3, 0.05):
        a, b, c = ground(delta)
        q1 = np.linspace(-8 / np.sqrt(a), 8 / np.sqrt(a), 401)
        L2 = max(8 / np.sqrt(b), 8.0)
        q2 = np.linspace(-L2, L2, 2048, endpoint=False)
        psi = psi0_grid(delta, q1, q2)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
        O2psi = p2_spectral(psi, q2) + W1 ** 2 * Q1 * psi
        # symmetric <{q2,O2}>/2 = Re <psi| q2 O2 |psi>
        integ = np.conj(psi) * Q2 * O2psi
        num = np.trapezoid(np.trapezoid(integ, q2, axis=1), q1)
        den = np.trapezoid(np.trapezoid(np.abs(psi) ** 2, q2, axis=1), q1)
        print("  delta=%.2f : Re<q2 O2> = %.2e (imag %.2e)"
              % (delta, np.real(num / den), np.imag(num / den)))

    # ---- [3] survival price sheet ---------------------------------------
    # CRUCIAL: BC(30) fixes q2 = xdot ; BC(59) fixes O2 ~ p2 = xddot, the
    # Ostrogradski MOMENTUM conjugate to q2 ([q2,p2]=i, so [q2,O2]=i). M'
    # is therefore a JOINT position+momentum (Heisenberg-limited) effect on
    # the velocity sector -- qualitatively unlike M (a single quadrature).
    # Consequence: the O(1) survival prefactor is ORDERING-DEPENDENT.
    #   * Weyl symbol exp(-q2^2/2s^2 -O2^2/2t^2) => p_Weyl=[(1+Vx/s^2)
    #     (1+Vp/t^2)]^{-1/2} = p1*p2 EXACTLY (valid effect iff s*t>=1/2).
    #   * product/Kraus E=exp(-O2^2/4t^2)exp(-q2^2/4s^2) => grid value,
    #     which is Heisenberg-SUPPRESSED below p1*p2.
    # Both share the SAME delta-scaling p_M' ~ sigma sqrt(2 delta).

    def grid_survival(delta, sig, tau, nq1=257, nq2=1024):
        a, b, c = ground(delta)
        q1 = np.linspace(-8 / np.sqrt(a), 8 / np.sqrt(a), nq1)
        L2 = max(8 / np.sqrt(b), 8.0)
        q2 = np.linspace(-L2, L2, nq2, endpoint=False)
        dq2 = q2[1] - q2[0]
        psi = psi0_grid(delta, q1, q2)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
        e = psi * np.exp(-Q2 ** 2 / (4 * sig ** 2))            # P1 (q2=xdot)
        kap = 2 * np.pi * np.fft.fftfreq(q2.size, d=dq2)       # p2=+kap
        filt = np.exp(-(kap[None, :] + W1 ** 2 * Q1[:, :1]) ** 2
                      / (4 * tau ** 2))                         # P2 (O2=xddot)
        e2 = np.fft.ifft(np.fft.fft(e, axis=1) * filt, axis=1)
        num = np.trapezoid(np.trapezoid(np.abs(e2) ** 2, q2, axis=1), q1)
        den = np.trapezoid(np.trapezoid(np.abs(psi) ** 2, q2, axis=1), q1)
        return num / den

    print("\n[3] survival.  p1=Neumann(=Maldacena), p2=BC59.")
    print("    p_Weyl = p1*p2 (Weyl-ordered, exact).")
    print("    p_grid = product/Kraus order ; closed form 2*sigma*tau*p1")
    print("    (Maldacena's p1 times the Heisenberg cell 2*sigma*tau).")
    print("    sigma=tau=0.1 (so 2*s*t=0.02)")
    print("    delta     p1        p2        p_Weyl    p_grid    2*s*t*p1")
    sig, tau = 0.1, 0.1
    for delta in (0.5, 0.1, 0.02, 0.004):
        p1 = 1.0 / np.sqrt(1 + 1 / (2 * sig ** 2 * delta))
        p2 = 1.0 / np.sqrt(1 + var_O2_closed(delta) / tau ** 2)
        pg = grid_survival(delta, sig, tau)
        print("    %.3f     %.5f   %.5f   %.5f   %.5f   %.5f"
              % (delta, p1, p2, p1 * p2, pg, 2 * sig * tau * p1))
    # limit checks: tau->inf recovers p1 exactly; sigma->inf recovers p2
    d0 = 0.1
    p1_ref = 1 / np.sqrt(1 + 1 / (2 * sig ** 2 * d0))
    p2_ref = 1 / np.sqrt(1 + var_O2_closed(d0) / tau ** 2)
    print("  limit checks (delta=0.1): grid[tau=1e3]=%.5f vs p1=%.5f ; "
          "grid[sig=1e3]=%.5f vs p2=%.5f"
          % (grid_survival(d0, sig, 1e3), p1_ref,
             grid_survival(d0, 1e3, tau), p2_ref))
    # scaling of the (authoritative) product-ordered survival
    ds = np.array([0.1, 0.02, 0.004])
    pgs = [grid_survival(d, sig, tau) for d in ds]
    print("  p_grid ~ delta^%.3f  (Maldacena p_M ~ delta^0.5)"
          % np.polyfit(np.log(ds), np.log(pgs), 1)[0])
    print("  -> BOTH orderings scale ~sqrt(delta); the joint-conjugate")
    print("     (Heisenberg) cost is an O(1) prefactor, not a new power.")

    # ---- [4] delivered-state fidelity to the Einstein (w1) vacuum -------
    print("\n[4] fidelity of delivered state to Einstein vacuum "
          "phi=(w1/pi)^1/4 exp(-w1 q1^2/2):")
    print("    F = <phi|rho_q1|phi>.  Compare M (Neumann only) vs M' (both)")
    print("    F_M closed form = 2 sqrt(a w1)/(a+w1) ~ sqrt(delta).")
    print("    delta     F_M (Neumann)   F_M' (both)   F_M closed-form")
    for delta in (0.5, 0.1, 0.02, 0.004):
        a, b, c = ground(delta)
        q1 = np.linspace(-10 / np.sqrt(a), 10 / np.sqrt(a), 221)
        L2 = max(10 / np.sqrt(b), 10.0)
        q2 = np.linspace(-L2, L2, 1024, endpoint=False)
        dq1 = q1[1] - q1[0]
        dq2 = q2[1] - q2[0]
        psi = psi0_grid(delta, q1, q2)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
        sig, tau = 0.05, 0.05
        # ---- M : Neumann only
        eM = psi * np.exp(-Q2 ** 2 / (4 * sig ** 2))
        # ---- M': Neumann + BC59
        kap = 2 * np.pi * np.fft.fftfreq(q2.size, d=dq2)
        eMp_hat = np.fft.fft(psi * np.exp(-Q2 ** 2 / (4 * sig ** 2)), axis=1)
        filt = np.exp(-(kap[None, :] + W1 ** 2 * Q1[:, :1]) ** 2
                      / (4 * tau ** 2))
        eMp = np.fft.ifft(eMp_hat * filt, axis=1)
        phi = (W1 / np.pi) ** 0.25 * np.exp(-W1 * q1 ** 2 / 2)

        def fidelity(e):
            rho = (e @ e.conj().T) * dq2
            tr = np.trapezoid(np.real(np.diag(rho)), q1)
            rho = rho / tr
            return np.real(phi @ rho @ phi) * dq1 ** 2

        aF = 2 * np.sqrt(a * W1) / (a + W1)
        print("    %.3f     %.5f        %.5f      %.5f"
              % (delta, fidelity(eM), fidelity(eMp), aF))

    # scaling of F_M' with delta
    ds = np.array([0.1, 0.02, 0.004])
    Fmp = []
    for delta in ds:
        a, b, c = ground(delta)
        q1 = np.linspace(-10 / np.sqrt(a), 10 / np.sqrt(a), 221)
        L2 = max(10 / np.sqrt(b), 10.0)
        q2 = np.linspace(-L2, L2, 1024, endpoint=False)
        dq1 = q1[1] - q1[0]
        dq2 = q2[1] - q2[0]
        psi = psi0_grid(delta, q1, q2)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
        sig, tau = 0.05, 0.05
        kap = 2 * np.pi * np.fft.fftfreq(q2.size, d=dq2)
        eMp_hat = np.fft.fft(psi * np.exp(-Q2 ** 2 / (4 * sig ** 2)), axis=1)
        filt = np.exp(-(kap[None, :] + W1 ** 2 * Q1[:, :1]) ** 2
                      / (4 * tau ** 2))
        eMp = np.fft.ifft(eMp_hat * filt, axis=1)
        phi = (W1 / np.pi) ** 0.25 * np.exp(-W1 * q1 ** 2 / 2)
        rho = (eMp @ eMp.conj().T) * dq2
        rho = rho / np.trapezoid(np.real(np.diag(rho)), q1)
        Fmp.append(np.real(phi @ rho @ phi) * dq1 ** 2)
    print("    F_M' values %s : roughly FLAT ~0.05-0.08, does NOT track"
          % np.round(Fmp, 4))
    print("    F_M~sqrt(delta). M' targets the GENERAL (2-param A,B) Einstein")
    print("    solution (a MIXED state), so overlap with the PURE w1 vacuum")
    print("    is low by design -- not the right scorecard for M'.")

    # ---- [5] verdict ----------------------------------------------------
    print("\n[5] PRICE-SHEET COMPARISON  M (Maldacena) vs M' (HLZ):")
    print("  survival:")
    print("    p_M   = [1+1/(2 s^2 d)]^-1/2                  ~ s sqrt(2 d)")
    print("    p_M'  = p1 * p2       (Weyl order)            ~ s sqrt(2 d)")
    print("          = 2 s t * p1    (product order)         ~ s sqrt(2 d)")
    print("    -> SAME delta-power as M. The only extra cost of the 2nd HLZ")
    print("       condition is an O(1) HEISENBERG factor (it fixes xddot,")
    print("       conjugate to the xdot that Neumann fixes). It is NOT a")
    print("       new source of delta-collapse: BC(59) is scale-free at d->0")
    print("       because Var(O2)~delta already vanishes.")
    print("  delivered state:")
    print("    F_M  ~ sqrt(delta)             (pure w1 vacuum, but too broad)")
    print("    F_M' ~ flat, small             (delivers the GENERAL/mixed")
    print("                                    Einstein solution, not the")
    print("                                    pure vacuum -> low vacuum F)")
    print("  BOTTOM LINE: M' does NOT cost less than M and does NOT evade the")
    print("  conformal (delta->0) collapse -- the survival still dies as")
    print("  ~sqrt(delta). What M' buys for its extra Heisenberg surcharge is")
    print("  QUALITATIVE, not cheaper: the GENERAL 2-parameter Einstein")
    print("  solution with NO in-state (positive-frequency) prep. This is")
    print("  exactly HLZ's claim ('Neumann alone gives no general solution in")
    print("  Minkowski') re-expressed as a survival price: the same delta->0")
    print("  obstruction, now with a Heisenberg-conjugate second condition.")
    print("done.")


if __name__ == "__main__":
    main()
