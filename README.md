# Every exorcism has a price

*ghost removal as survival conditioning, from Maldacena's boundary condition to the pole sheet*

![exorcised-ghosts](exorcised-ghosts.png)

<sub>*Banner image: prompt written by Claude (Anthropic), rendered by ChatGPT (OpenAI).*</sub>

**Paper 8 of the [Indivisible Ghosts program](https://github.com/embeddetech/ghosts-program).** Jonathan Torkelson (2026).

**[Read the paper (PDF)](exorcised-ghosts-paper.pdf)** · LaTeX source in [`paper/`](paper/)

> **Status: working draft.** This manuscript is under the program's release-gate audit and is still being revised. It has not been peer reviewed. Read it accordingly; corrections land here as they are made.

## Abstract

**Draft — not for circulation.**

A higher-derivative (ghost) theory can be made ghost-free by a choice: a boundary condition, a contour, a norm. The companion papers of this series established a currency in which such choices can be priced: a quantum evolution admits an ordinary (“beable”) sub-stochastic process whenever its survival amplitude stays bounded—divisor negativity then marking indivisibility rather than forcing conditioning—and only when the amplitude grows without bound does the dynamics survive solely as a *survival-conditioned* process—which, for the coupled ghost realizations priced here, is the terminal-indivisibility class, with a strictly negative generator floor (unboundedness alone forces conditioning but not terminality: a decoupled growing mode stays divisible). This paper applies that currency to ghost *removal*. The thesis: every standard exorcism—boundary conditions that project the ghost out, and propagator prescriptions that redefine its pole—is survival conditioning in disguise, or buys its way out in another trilemma currency; for the boundary-condition removals and the Feynman route that price is computed here. Three results. (i) *Maldacena’s future boundary condition* (conformal $`\to`$ Einstein gravity) is, mode by mode, a Gaussian effect $`\mathcal{M}`$ (regularized $`E_{\sigma}`$) on the Ostrogradski velocity variable of a detuned Pais–Uhlenbeck proxy whose detuning $`\delta`$ represents the de Sitter $`\Delta=3/\Delta=2`$ splitting. Its price sheet is closed form: survival $`p=[1+1/(2\sigma^{2}\delta)]^{-1/2}\xrightarrow{\sigma\to0}\sigma\sqrt{2\delta}`$, and Einstein-vacuum fidelity of the delivered state $`F=2\sqrt{\alpha\omega_{1}}/(\alpha+\omega_{1})\simeq2\sqrt{\omega_{2}\delta}`$ (with $`\alpha=\omega_{1}\omega_{2}\delta`$, in units $`\omega_{1}=1`$). Both collapse at the conformal (Minkowski, $`\delta\to0`$) point—the known flat-space obstruction, now priced; the convention-free content is the survival collapse and the delivered state’s absolute $`\delta^{-1/2}`$ width divergence, the fidelity reading the latter against a fixed Einstein-vacuum reference. (ii) The *Hell–Lüst–Zoupanos two-condition variant* $`\mathcal{M}'`$ fixes a canonically conjugate pair ($`\dot h`$ and the conjugate combination $`\ddot h+\omega_{1}^{2}h`$, $`[q_{2},O_{2}]=\mathrm{i}`$), so $`\mathcal{M}'`$ is Heisenberg-limited: it pays the same leading $`\sigma\sqrt{2\delta}`$ survival power plus an $`O(1)`$ ordering/phase-space-cell surcharge, and what the surcharge buys is not a cheaper exorcism but a more general one—the full two-parameter Einstein solution with no in-state restriction. The $`\delta\to0`$ collapse is not evaded. (iii) *The pole sheet is readable in the divisor*: by Bochner’s theorem a second-sheet resonance (positive spectral measure) has $`|A(t)|\le1`$ and survives as an *unconditioned* sub-stochastic process (its divisor negativity transient when the amplitude is eventually monotone decreasing), while a first-sheet ghost pair (a discrete complex-conjugate pole pair off the real axis) has unbounded $`|A(t)|`$ in states overlapping the growing leg, so conditioning is *mandatory*; for the dimer realization it is terminal, with the generator floor of the first companion paper. Buoninfante’s multi-particle masking law $`\sinh^{2}(\Gamma t/2)`$ is exactly the time dependence of the cross-population of the terminal-class (conditioning-mandatory) dimer (with $`\widetilde{k}=\Gamma/2`$), up to the normalization $`(\kappa/\widetilde{k})^{2}`$, and his ghost-prescription taxonomy (Feynman ghost / anti-Feynman / fakeon) maps onto the trilemma currencies of the series: sample space, causal order, silence. Unstable particles survive unconditioned; ghosts survive only conditioned—and every way of pretending otherwise pays, measurably, in one currency or another.

## Reproducing the results

Each script is standalone Python (NumPy/SciPy) and prints its controls inline. The paper's *Code availability* section maps scripts to results.

```
pip install numpy scipy
python scripts/p8_buoninfante_map.py
```

| Script |
|---|
| [`p8_buoninfante_map.py`](scripts/p8_buoninfante_map.py) |
| [`p8_hlz_mprime.py`](scripts/p8_hlz_mprime.py) |
| [`p8_maldacena_price.py`](scripts/p8_maldacena_price.py) |

## Audit trail and provenance

This manuscript is an experiment in AI-conducted research: the work was carried out by large language models (Claude, Anthropic), directed and reviewed by the author. The paper's provenance statement says who did what. Nothing here should be taken on trust.

- The audit process: [`AUDIT.md`](AUDIT.md)
- Pass-by-pass audit ledger: [`histories/paper8-history.md`](histories/paper8-history.md)
- The full research notebook, the originating conversation, and every companion paper: the [program hub](https://github.com/embeddetech/ghosts-program)

## License

MIT — see [`LICENSE`](LICENSE).
