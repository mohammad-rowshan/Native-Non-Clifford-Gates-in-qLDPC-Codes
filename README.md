# Native Non-Clifford Gates in qLDPC Codes

Reproducibility code for the paper

> **Native Non-Clifford Gates in qLDPC Codes: Conditions, Synthesis, and Scaling Limits**
> Mohammad Rowshan (UNSW Sydney & UTS:QSI)

A single self-contained Python script, [`repro.py`](repro.py), that recomputes
every numerical quantity, counterexample, and asserted claim in the
manuscript from scratch over GF(2). There are no data files and no
precomputed tables: each number printed by the script is derived on the fly,
and every claim the paper makes "the script checks that ..." is backed by an
`assert` that fails loudly if the property does not hold.

## What the script establishes

The paper studies when a triple `(x, y, z)` of X-type logical representatives
of a CSS code can host a native, constant-depth logical `CCZ` gate. A triple
is **CCZ-admissible** when the three pairwise inner products vanish and the
triple overlap `tau(x,y,z)` is odd. The script verifies the following, in the
order the paper presents them.

- **Stability criterion (Proposition 1).** `tau` is invariant under
  simultaneous stabilizer changes of all three representatives iff three
  conditions (S1), (S2), (S3) hold, where (S3) forces the X-stabilizer code
  `C_X` to be totally triorthogonal. The coded criterion is checked against
  brute-force enumeration over all `(u, v, w)` in `C_X^3` on 1000 random
  instances.
- **Seven-term synthesis (Lemma 3) and its uniqueness (Lemma 4).** For a
  saturated triple, a diagonal circuit of `Z`, `CZ`, `CCZ` gates realizes the
  cubic phase. Its gate multiset equals the support of the algebraic normal
  form of the phase, so it is the unique reduced ancilla-free realization; the
  script confirms the synthesized gate list matches the ANF on every basis
  state of the combined support.
- **Depth–distance barrier (Corollary 1).** The exact per-qubit incidences of
  the reduced circuit are computed by four closed-form region formulas and
  checked against the ANF, giving `L >= Delta >= 3p >= d`.
- **Code-space preservation (Lemma 7).** A diagonal circuit preserves the code
  space iff its phase is constant on every coset of `C_X` inside `C_Z^perp`;
  the script evaluates this coset-phase test directly.
- **Negative example (Section VI-A).** On hypergraph-product codes built from
  the `[7,4,3]` Hamming code and path parity-check matrices, exhaustive search
  finds saturated **admissible** triples that fail all of (S1), (S2), (S3),
  and whose diagonal circuits do **not** preserve the code space. The
  `[[27,4,3]]` instance is confirmed by exact minimum-distance computation
  over both stabilizer sectors.
- **Positive example (Section VI-B).** An explicit `[[12,3,1]]` code carries a
  saturated triple satisfying admissibility and all of (S1), (S2), (S3), with
  the code space provably preserved. Its distance is small, and
  **Proposition 6** explains why this is forced: every stabilizer meets each
  region in even size, so `d <= 3p`.

## Requirements

The script uses only `numpy` and the standard library
(`itertools`).

## Usage

```bash
python repro.py
```

The script prints a sequence of clearly labeled blocks, one per result above,
and exits with a nonzero status only if any internal `assert` fails. A
successful run takes a few seconds on a laptop; the most expensive step is the
exact minimum-distance computation for the `[[27,4,3]]` code, which enumerates
all `2^14` elements of one stabilizer sector.

Expected tail of the output (abridged):

```
====================================================================
Section VI-B: a code satisfying admissibility, saturation, (S1)-(S3)
====================================================================
  [[12,3,1]]  dX=3 dZ=1  (S1,S2,S3)=(True,True,True)  code space preserved=True
  synthesis: 111 gates, Delta=37=1+3(3)+3(3)^2, phase mismatches=0
  Proposition (stability bounds distance): every stabilizer meets each region evenly; d=1 <= 3p=9
```

## How the code is organized

Everything lives in one file so it can be read top to bottom alongside the
paper.

| Component | Functions | Paper reference |
|---|---|---|
| GF(2) linear algebra | `rref`, `rank2`, `nullspace`, `row_reduce_basis`, `span`, `kron2` | Section II conventions |
| Overlap forms | `tau`, `ip`, `regions`, `admissible` | Definition 1 |
| Stability tests | `schur_span`, `one_coordinate_tests`, `stability_criterion` | Lemma 1, Proposition 1 |
| Brute-force checks | `brute_tau_stable`, `brute_orth_stable` | Proposition 1 (validation) |
| Circuit synthesis | `synthesize`, `anf_support` | Lemma 3, Lemma 4, Corollary 1 |
| Code-space test | `preserves_code_space` | Lemma 7 |
| Code construction | `hgp`, `logical_basis`, `min_distance`, `path_check`, `search` | Section VI |

Two implementation notes that matter if you extend the code. The Schur span
`D` is built from products of a **basis** of `C_X` rather than of all its
elements, which is valid by bilinearity of the Schur product over GF(2) and
reduces the work from `4^r` to `r^2` products for a rank-`r` stabilizer space.
The code follows one fixed CSS convention throughout, `C_X = rowspace(H_X)` and
`C_Z = rowspace(H_Z)` with `H_X H_Z^T = 0`; mixing conventions will silently
break the orthogonality assertions.

## Citation

If you use this code, please cite the accompanying paper. A BibTeX entry will
be added here once the preprint is public.

## License

Released under the MIT License. See [`LICENSE`](LICENSE) for details.
