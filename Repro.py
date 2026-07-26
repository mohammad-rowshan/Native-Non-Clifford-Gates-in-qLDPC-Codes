"""
Verification script for the paper to apear in the proceedings of the IEEE QCE 2026, titled
"Native Non-Clifford Gates in qLDPC Codes: Conditions, Synthesis, and Scaling Limits".

Reproduces every computational quantity reported in Section VI and in the
stability discussion of Section II:

  * hypergraph-product CSS codes over GF(2), with H_X H_Z^T = 0 asserted;
  * exact minimum distance of the [[27,4,3]] instance, both sectors;
  * exhaustive search over triples of the selected canonical
    representatives of the X-logical classes;
  * the one-coordinate Schur tests (S1) and the full stability
    criterion (S1)-(S3);
  * the coset-phase test that decides code-space preservation for a
    diagonal circuit;
  * the seven-term Zhegalkin identity and the diagonal circuit it
    synthesizes, checked against the cubic phase on every basis state.

Conventions, fixed once and used throughout:
  C_X = rowspace(H_X), C_Z = rowspace(H_Z), with H_X H_Z^T = 0 over GF(2).
  L_X = C_Z^perp / C_X,  L_Z = C_X^perp / C_Z,  k = n - rank H_X - rank H_Z.

HGP of H1 (m1 x n1) and H2 (m2 x n2), n = n1*n2 + m1*m2:
  H_X = [ H1 (x) I_n2 | I_m1 (x) H2^T ]
  H_Z = [ I_n1 (x) H2 | H1^T (x) I_m2 ]
so that
  H_X H_Z^T = (H1 (x) I_n2)(I_n1 (x) H2^T) + (I_m1 (x) H2^T)(H1 (x) I_m2)
            = H1 (x) H2^T + H1 (x) H2^T = 0.
Note the transpose: H1 (x) H2 has the wrong shape for this product.
"""

import itertools
import numpy as np

# ----------------------------------------------------------------------
# GF(2) linear algebra
# ----------------------------------------------------------------------

def rref(M):
    """Row-reduced echelon form over GF(2). Returns (R, pivot_columns)."""
    R = M.copy() % 2
    rows, cols = R.shape
    piv, r = [], 0
    for c in range(cols):
        sel = next((i for i in range(r, rows) if R[i, c]), None)
        if sel is None:
            continue
        R[[r, sel]] = R[[sel, r]]
        for i in range(rows):
            if i != r and R[i, c]:
                R[i] = (R[i] + R[r]) % 2
        piv.append(c)
        r += 1
        if r == rows:
            break
    return R, piv


def rank2(M):
    return 0 if M.size == 0 else len(rref(M)[1])


def nullspace(M):
    """Basis (as rows) of {v : M v = 0} over GF(2)."""
    rows, cols = M.shape
    R, piv = rref(M)
    basis = []
    for f in (c for c in range(cols) if c not in piv):
        v = np.zeros(cols, dtype=np.uint8)
        v[f] = 1
        for i, p in enumerate(piv):
            if R[i, f]:
                v[p] = 1
        basis.append(v)
    return np.array(basis, np.uint8) if basis else np.zeros((0, cols), np.uint8)


def row_reduce_basis(rows, n):
    """Independent subset of `rows` spanning the same GF(2) space."""
    keep, stack = [], np.zeros((0, n), np.uint8)
    for v in rows:
        if not v.any():
            continue
        test = np.vstack([stack, v[None, :]])
        if rank2(test) > rank2(stack):
            keep.append(v)
            stack = test
    return np.array(keep, np.uint8) if keep else np.zeros((0, n), np.uint8)


def span(M):
    """
    All 2^rank(M) distinct elements of the GF(2) rowspace of M.

    The rows are reduced to an independent basis first; iterating over
    dependent rows would emit each element more than once.
    """
    n = M.shape[1]
    G = row_reduce_basis(list(M), n)
    out = [np.zeros(n, np.uint8)]
    for row in G:
        out += [(v + row) % 2 for v in out]
    return out


def kron2(A, B):
    return np.kron(A, B).astype(np.uint8) % 2


# ----------------------------------------------------------------------
# Overlap forms and the simultaneous-stability criterion
# ----------------------------------------------------------------------

def tau(x, y, z):
    """Triple overlap: parity of |supp(x) & supp(y) & supp(z)|."""
    return int((x & y & z).sum() % 2)


def ip(u, v):
    return int((u & v).sum() % 2)


def schur_span(HX):
    """
    Basis of D = span{ u (*) v : u, v in C_X }.

    The Schur (bitwise AND) product is bilinear over GF(2): if
    u = sum_i a_i g_i and v = sum_j b_j g_j then
    (u (*) v)_k = (sum_i a_i g_{i,k})(sum_j b_j g_{j,k})
                = sum_{i,j} a_i b_j (g_i (*) g_j)_k  (mod 2),
    so D is spanned by the products of BASIS vectors alone.  For a rank-r
    stabilizer space this is r^2 candidate products rather than 4^r, which
    is what makes this function cheap (81 products at r = 9, not 262144).
    Note C_X is a subspace of D, since g (*) g = g.
    """
    n = HX.shape[1]
    G = row_reduce_basis(list(HX), n)
    prods = [(G[i] & G[j]) for i in range(len(G)) for j in range(i, len(G))]
    return row_reduce_basis(prods, n)


def perp(vecs, basisM):
    """True iff every v in vecs is orthogonal to every row of basisM."""
    return all(not ((basisM @ v) % 2).any() for v in vecs)


def one_coordinate_tests(x, y, z, HX):
    """
    (S1) only: the three Schur products x(*)y, x(*)z, y(*)z lie in C_X^perp.

    These are the one-coordinate conditions, controlling variation of ONE
    representative with the other two held fixed.  They are necessary but
    NOT sufficient for tau to be a function of the three logical classes;
    see stability_criterion below.
    """
    return perp([(x & y), (x & z), (y & z)], HX)


def stability_criterion(x, y, z, HX, D):
    """
    Full simultaneous-stability criterion.  Returns (S1, S2, S3):
      (S1) x(*)y, x(*)z, y(*)z in C_X^perp
      (S2) x, y, z in D^perp
      (S3) C_X in D^perp   (trilinear form vanishes on all of C_X^3)
    tau is invariant under x->x+u, y->y+v, z->z+w for all u,v,w in C_X
    iff all three hold.
    """
    s1 = one_coordinate_tests(x, y, z, HX)
    s2 = perp([x, y, z], D)
    s3 = perp(list(HX), D)
    return s1, s2, s3


def preserves_code_space(x, y, z, HX, HZ):
    """
    Code-space preservation.  A diagonal U with phase
    Q(c) = l_x(c) l_y(c) l_z(c) maps the
    code space of CSS(C_X, C_Z) into itself iff Q is constant on every coset
    c + C_X with c in ker H_Z = C_Z^perp, because a logical basis state is the
    uniform superposition over such a coset.

    Returns (preserved, n_bad_cosets, n_cosets, witness), where witness is
    some u in C_X with Q(u) != Q(0) if one exists, else None.  The witness
    alone certifies failure, since 0 lies in C_Z^perp.
    """
    n = HX.shape[1]
    CX = span(HX)
    Q = lambda c: (ip(x, c) * ip(y, c) * ip(z, c)) % 2
    q0 = Q(np.zeros(n, np.uint8))
    witness = next((u for u in CX if Q(u) != q0), None)

    # One representative per coset of C_X inside C_Z^perp.  logical_basis
    # returns exactly a basis of the quotient C_Z^perp / C_X, so spanning it
    # enumerates the 2^k cosets directly instead of sieving all of C_Z^perp.
    reps = span(logical_basis(HX, HZ))
    bad = sum(1 for c in reps if len({Q((c + u) % 2) for u in CX}) > 1)
    return bad == 0, bad, len(reps), witness


def brute_tau_stable(x, y, z, CX):
    """Direct check over every (u,v,w) in C_X^3.  Exponential; small cases only."""
    t0 = tau(x, y, z)
    return all(tau((x + u) % 2, (y + v) % 2, (z + w) % 2) == t0
               for u in CX for v in CX for w in CX)


def brute_orth_stable(x, y, z, CX):
    """Direct check that the three pairwise inner products are representative-stable."""
    for u in CX:
        for v in CX:
            if ip((x + u) % 2, (y + v) % 2) != ip(x, y): return False
            if ip((x + u) % 2, (z + v) % 2) != ip(x, z): return False
            if ip((y + u) % 2, (z + v) % 2) != ip(y, z): return False
    return True


# ----------------------------------------------------------------------
# Region decomposition and the seven-term synthesis
# ----------------------------------------------------------------------

def regions(x, y, z):
    """Sizes of the seven support regions of a triple."""
    X, Y, Z = x.astype(bool), y.astype(bool), z.astype(bool)
    return dict(
        Px=int((X & ~Y & ~Z).sum()), Py=int((~X & Y & ~Z).sum()),
        Pz=int((~X & ~Y & Z).sum()),
        Pxy=int((X & Y & ~Z).sum()), Pxz=int((X & ~Y & Z).sum()),
        Pyz=int((~X & Y & Z).sum()), Pxyz=int((X & Y & Z).sum()),
        S=int((X | Y | Z).sum()),
    )


def admissible(x, y, z):
    """Pairwise orthogonal with odd triple overlap."""
    return (ip(x, y) == 0 and ip(x, z) == 0 and ip(y, z) == 0
            and tau(x, y, z) == 1)


def synthesize(x, y, z):
    """
    Build the diagonal circuit from the seven-term identity
        l_x l_y l_z = abe + ace + bce + ae + be + ce + e
    with a = P_xy, b = P_xz, c = P_yz, e = P_xyz, and verify it against the
    cubic phase on every basis state of S_t.  Requires saturation.
    Returns (gate count, max per-qubit incidence, mismatch count).
    """
    X, Y, Z = x.astype(bool), y.astype(bool), z.astype(bool)
    assert not (X & ~Y & ~Z).any() and not (~X & Y & ~Z).any() \
        and not (~X & ~Y & Z).any(), "triple is not saturated"
    Rxy = np.flatnonzero(X & Y & ~Z)
    Rxz = np.flatnonzero(X & ~Y & Z)
    Ryz = np.flatnonzero(~X & Y & Z)
    Re = np.flatnonzero(X & Y & Z)
    S = np.flatnonzero(X | Y | Z)

    gates = []                                    # CCZ: abe, ace, bce
    for P, Q, R in ((Rxy, Rxz, Re), (Rxy, Ryz, Re), (Rxz, Ryz, Re)):
        gates += [(i, j, k) for i in P for j in Q for k in R]
    for P, R in ((Rxy, Re), (Rxz, Re), (Ryz, Re)):   # CZ: ae, be, ce
        gates += [(i, k) for i in P for k in R]
    gates += [(k,) for k in Re]                      # Z: e

    incidence = max(sum(q in g for g in gates) for q in S)
    monomials = {frozenset(g) for g in gates}
    assert len(monomials) == len(gates), "synthesis emitted a repeated gate"
    assert monomials == anf_support(x, y, z, S), \
        "synthesized gate list must equal the ANF support (uniqueness lemma)"
    bad = 0
    for bits in itertools.product([0, 1], repeat=len(S)):
        c = np.zeros(len(x), np.uint8)
        c[S] = bits
        lhs = (int(x @ c % 2) * int(y @ c % 2) * int(z @ c % 2)) % 2
        rhs = sum(int(np.prod([c[i] for i in g])) for g in gates) % 2
        bad += (lhs != rhs)
    return len(gates), incidence, bad


# ----------------------------------------------------------------------
# Hypergraph-product construction
# ----------------------------------------------------------------------

def anf_support(x, y, z, S):
    """
    Monomials of the algebraic normal form of Q(c) = l_x(c) l_y(c) l_z(c),
    restricted to the coordinates S, as a set of frozensets.

    Computed by the Mobius (Zhegalkin) transform of the truth table.  By
    uniqueness of the ANF, any ancilla-free circuit of Z / CZ / CCZ gates
    realizing Q on every basis state must, after cancelling repeated gates,
    consist of exactly these monomials -- so this set is the unique reduced
    gate list, and the per-qubit incidence it gives is exact in that model.
    """
    S = list(S)
    m = len(S)
    coeff = {}
    for bits in itertools.product([0, 1], repeat=m):
        c = np.zeros(len(x), np.uint8)
        for b, i in zip(bits, S):
            if b:
                c[i] = 1
        coeff[bits] = (ip(x, c) * ip(y, c) * ip(z, c)) % 2
    for j in range(m):
        for bits in list(coeff):
            if bits[j]:
                lower = list(bits)
                lower[j] = 0
                coeff[bits] = (coeff[bits] + coeff[tuple(lower)]) % 2
    return {frozenset(S[j] for j in range(m) if bits[j])
            for bits, v in coeff.items() if v}


def hgp(H1, H2):
    m1, n1 = H1.shape
    m2, n2 = H2.shape
    HX = np.hstack([kron2(H1, np.eye(n2, dtype=np.uint8)),
                    kron2(np.eye(m1, dtype=np.uint8), H2.T)]) % 2
    HZ = np.hstack([kron2(np.eye(n1, dtype=np.uint8), H2),
                    kron2(H1.T, np.eye(m2, dtype=np.uint8))]) % 2
    assert not ((HX @ HZ.T) % 2).any(), "CSS orthogonality failed"
    return HX, HZ


def logical_basis(Hstab, Hother):
    """Representatives of a basis of ker(Hother) / rowspace(Hstab)."""
    reps, stack = [], Hstab.copy()
    for v in nullspace(Hother):
        test = np.vstack([stack, v[None, :]])
        if rank2(test) > rank2(stack):
            reps.append(v)
            stack = test
    return np.array(reps, np.uint8)


def min_distance(reps, stabspan, n):
    """Exact minimum weight over all nonzero classes and all coset shifts."""
    best = n + 1
    for bits in itertools.product([0, 1], repeat=len(reps)):
        if not any(bits):
            continue
        v = np.zeros(n, np.uint8)
        for b, r in zip(bits, reps):
            if b:
                v = (v + r) % 2
        for s in stabspan:
            best = min(best, int(((v + s) % 2).sum()))
    return best


# H_1: parity-check matrix of the [7,4,3] Hamming code, Eq. (18) of the paper.
HAMMING = np.array([[1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)


def path_check(nu):
    """(nu-1) x nu matrix with rows e_i + e_{i+1}; kernel = {0...0, 1...1}."""
    H = np.zeros((nu - 1, nu), dtype=np.uint8)
    for i in range(nu - 1):
        H[i, i] = H[i, i + 1] = 1
    return H


def search(HX, HZ, label, report_hits=1):
    """
    Exhaustive search over triples of CANONICAL representatives of the
    X-logical classes: one representative per class, formed by summing the
    selected quotient-basis vectors.  Admissibility is representative-
    dependent, so this does not search all C_X-shifts of every class.
    """
    n = HX.shape[1]
    rX, rZ = rank2(HX), rank2(HZ)
    L = logical_basis(HX, HZ)
    D = schur_span(HX)
    classes = []
    for bits in itertools.product([0, 1], repeat=len(L)):
        if not any(bits):
            continue
        v = np.zeros(n, np.uint8)
        for b, row in zip(bits, L):
            if b:
                v = (v + row) % 2
        classes.append((bits, v))

    hits, checked = [], 0
    for (b1, x), (b2, y), (b3, z) in itertools.combinations(classes, 3):
        if rank2(np.array([b1, b2, b3], np.uint8)) < 3:
            continue                      # classes not independent in L_X
        checked += 1
        if admissible(x, y, z):
            hits.append(((b1, b2, b3), (x, y, z), regions(x, y, z),
                         stability_criterion(x, y, z, HX, D)))

    print(f"--- {label} ---")
    print(f"  n={n}  rank H_X={rX}  rank H_Z={rZ}  k={n - rX - rZ}")
    print(f"  H_X row wt {HX.sum(1).min()}-{HX.sum(1).max()}, "
          f"col wt <= {HX.sum(0).max()};  "
          f"H_Z row wt {HZ.sum(1).min()}-{HZ.sum(1).max()}, "
          f"col wt <= {HZ.sum(0).max()}")
    print(f"  logical-X basis weights: {[int(v.sum()) for v in L]}")
    print(f"  dim C_X={rX}, dim D={rank2(D)}")
    print(f"  independent triples checked: {checked}")
    print(f"  CCZ-admissible triples found: {len(hits)}")
    for coeffs, _, reg, (s1, s2, s3) in hits[:report_hits]:
        print(f"    coeffs={coeffs}")
        print(f"    regions={reg}")
        print(f"    stability: (S1)={s1}  (S2)={s2}  (S3)={s3}")
    return dict(n=n, k=n - rX - rZ, checked=checked, found=len(hits),
                hits=hits, HX=HX, HZ=HZ, L=L, D=D)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 68)
    print("SECTION VI: hypergraph-product instances, Table III")
    print("=" * 68)
    results = {}
    for nu in (3, 4, 5):
        H2 = path_check(nu)
        ker_wt = int(nullspace(H2).sum())
        HX, HZ = hgp(HAMMING, H2)
        results[nu] = search(HX, HZ,
                             f"HGP(Hamming[7,4], path_{nu}), ker H2 weight {ker_wt}",
                             report_hits=(1 if nu == 3 else 0))

    print("\nTable III summary:")
    print("  nu  [[n,k]]      d         indep.  admissible")
    for nu, r in results.items():
        # nu=3 is computed exactly below; for nu=4,5 the lowest-weight
        # representative found gives an upper bound, which is the value
        # quoted with a '<=' in Table III.
        wmin = min(int(v.sum()) for v in r["L"])
        dstr = "3 (exact)" if nu == 3 else f"<= {wmin}"
        print(f"  {nu}   [[{r['n']},{r['k']}]]  {dstr:9s} "
              f"{r['checked']:6d}  {r['found']:d}")

    # ---------------- exact distance of the nu=3 instance ----------------
    print("\n" + "=" * 68)
    print("Exact minimum distance of the nu=3 instance (both sectors)")
    print("=" * 68)
    r3 = results[3]
    HX, HZ, n = r3["HX"], r3["HZ"], r3["n"]
    LX = r3["L"]
    LZ = logical_basis(HZ, HX)
    dX = min_distance(LX, span(HX), n)       # 2^9 shifts per class
    dZ = min_distance(LZ, span(HZ), n)       # 2^14 shifts per class
    print(f"  d_X = {dX} (over 2^{rank2(HX)} elements of C_X)")
    print(f"  d_Z = {dZ} (over 2^{rank2(HZ)} elements of C_Z)")
    print(f"  => [[{n},{r3['k']},{min(dX, dZ)}]]")
    print(f"  logical-X basis supports (chi_1..chi_4): "
          f"{[np.flatnonzero(v).tolist() for v in LX]}")

    # ---------------- stability of the nu=3 admissible triple ------------
    print("\n" + "=" * 68)
    print("Stability of the nu=3 admissible triple (Section VI-A)")
    print("=" * 68)
    (_, (x, y, z), reg, (s1, s2, s3)) = r3["hits"][0]
    D = r3["D"]
    print(f"  regions: {reg}")
    print(f"  (S1) one-coordinate Schur tests: {s1}")
    for nm, (a, b) in zip(("x*y", "x*z", "y*z"), ((x, y), (x, z), (y, z))):
        print(f"        H_X({nm}) = 0 ? {not ((HX @ (a & b)) % 2).any()}")
    print(f"  (S2) x,y,z in D^perp: {s2}")
    for nm, v in zip("xyz", (x, y, z)):
        print(f"        H_D {nm} = 0 ? {not ((D @ v) % 2).any()}")
    print(f"  (S3) C_X in D^perp: {s3}")
    gens = [not ((D @ g) % 2).any() for g in HX]
    print(f"        generators of C_X in D^perp: {sum(gens)}/{len(gens)}")
    assert all(not a and not b and not c
               for _, _, _, (a, b, c) in r3["hits"]), \
        "manuscript claims all four triples fail (S1), (S2) and (S3)"
    # the prose makes the stronger per-product and per-vector claims, not just
    # the aggregate booleans, so assert those directly
    for _, (xa, ya, za), _, _ in r3["hits"]:
        assert all(((HX @ (u & v)) % 2).any()
                   for u, v in ((xa, ya), (xa, za), (ya, za))), \
            "each of the three Schur products must fail C_X^perp membership"
        assert all(((D @ v) % 2).any() for v in (xa, ya, za)), \
            "each of x, y, z must fail D^perp membership"
    assert all(((D @ g) % 2).any() for g in HX), \
        "every generator of C_X must fail D^perp membership"
    print(f"  all {len(r3['hits'])} admissible triples fail (S1), (S2) and (S3),")
    print(f"  each of the three Schur products and each of x,y,z failing "
          f"separately, and all {HX.shape[0]} generators of C_X failing (S3)")

    # ---------------- seven-term identity and synthesis ------------------
    print("\n" + "=" * 68)
    print("Seven-term identity and diagonal circuit synthesis")
    print("=" * 68)
    ok = all(((a + b + e) % 2) * ((a + c + e) % 2) * ((b + c + e) % 2) % 2
             == (a*b*e + a*c*e + b*c*e + a*e + b*e + c*e + e) % 2
             for a, b, c, e in itertools.product([0, 1], repeat=4))
    print(f"  identity holds on all 16 assignments of (a,b,c,e): {ok}")

    x4 = np.array([0, 1, 1, 1], np.uint8)
    y4 = np.array([1, 0, 1, 1], np.uint8)
    z4 = np.array([1, 1, 0, 1], np.uint8)
    # Example 1's logical action rests on {x,y,z,r} being a self-dual basis
    r4 = np.array([1, 1, 1, 0], np.uint8)
    B4 = [x4, y4, z4, r4]
    gram = [[ip(u, v) for v in B4] for u in B4]
    assert gram == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], gram
    assert rank2(np.array(B4, np.uint8)) == 4
    mism = 0
    for bits in itertools.product([0, 1], repeat=4):
        c = np.zeros(4, np.uint8)
        for b, v in zip(bits, B4):
            if b:
                c = (c + v) % 2
        mism += ((ip(x4, c) * ip(y4, c) * ip(z4, c)) % 2
                 != bits[0] * bits[1] * bits[2])
    assert mism == 0
    print(f"  Example 1: {{x,y,z,r}} self-dual (Gram = I), phase = CCZ on the "
          f"first three basis coords, mismatches {mism}")

    g, inc, bad = synthesize(x4, y4, z4)
    print(f"  Example 1 (p_0=1): {g} gates, max incidence {inc}, "
          f"phase mismatches {bad};  1+3p+3p^2 = {1+3+3}")
    g, inc, bad = synthesize(x, y, z)
    print(f"  Section VI-A triple (p_0=3): {g} gates, max incidence {inc}, "
          f"phase mismatches {bad};  1+3p+3p^2 = {1+9+27}")

    # ---------------- exact region incidences and L >= Delta >= 3p >= d ---
    print("\n" + "=" * 68)
    print("Exact region incidences in the reduced ANF gate multiset")
    print("=" * 68)

    def region_triple(a, b, c, e):
        """Saturated triple with |P_xy|=a, |P_xz|=b, |P_yz|=c, |P_xyz|=e."""
        nn = a + b + c + e
        xx = np.zeros(nn, np.uint8)
        yy = np.zeros(nn, np.uint8)
        zz = np.zeros(nn, np.uint8)
        i = 0
        Rxy = list(range(i, i + a)); i += a
        Rxz = list(range(i, i + b)); i += b
        Ryz = list(range(i, i + c)); i += c
        Re = list(range(i, i + e))
        for j in Rxy: xx[j] = yy[j] = 1
        for j in Rxz: xx[j] = zz[j] = 1
        for j in Ryz: yy[j] = zz[j] = 1
        for j in Re:  xx[j] = yy[j] = zz[j] = 1
        return xx, yy, zz, Rxy, Rxz, Ryz, Re

    def predicted(a, b, c, e):
        return dict(xy=e * (b + c + 1), xz=e * (a + c + 1),
                    yz=e * (a + b + 1),
                    xyz=a*b + a*c + b*c + a + b + c + 1)

    print("  (a,b,c,e)      predicted [xy,xz,yz,xyz]   3p   Delta>=3p  wt<=3p")
    for (a, b, c, e) in [(1,1,1,1), (3,3,3,3), (1,1,3,1),
                         (3,1,1,5), (1,3,5,1), (5,3,1,3), (7,1,1,1)]:
        xx, yy, zz, Rxy, Rxz, Ryz, Re = region_triple(a, b, c, e)
        SS = np.flatnonzero(xx | yy | zz)
        anf = anf_support(xx, yy, zz, SS)
        act = dict(xy=sum(1 for m in anf if Rxy[0] in m),
                   xz=sum(1 for m in anf if Rxz[0] in m),
                   yz=sum(1 for m in anf if Ryz[0] in m),
                   xyz=sum(1 for m in anf if Re[0] in m))
        pr = predicted(a, b, c, e)
        assert act == pr, (a, b, c, e, act, pr)
        pmax = max(a, b, c, e)
        Delta = max(act.values())
        wmax = max(int(xx.sum()), int(yy.sum()), int(zz.sum()))
        assert Delta >= 3 * pmax and wmax <= 3 * pmax
        print(f"  {str((a,b,c,e)):>13}  "
              f"{str([pr['xy'], pr['xz'], pr['yz'], pr['xyz']]):>22}  "
              f"{3*pmax:>3}   {str(Delta >= 3*pmax):>9}  {str(wmax <= 3*pmax):>6}")

    # Delta >= 3p over all odd region sizes up to 9
    bad = [(a, b, c, e)
           for a in range(1, 10, 2) for b in range(1, 10, 2)
           for c in range(1, 10, 2) for e in range(1, 10, 2)
           if max(predicted(a, b, c, e).values()) < 3 * max(a, b, c, e)]
    assert not bad
    print(f"  Delta >= 3p on all {5**4} odd (a,b,c,e) with entries <= 9: "
          f"{len(bad)} counterexamples")

    # ---------------- code-space preservation ----------------------------
    print("\n" + "=" * 68)
    print("Code-space preservation: does each circuit preserve the code space?")
    print("=" * 68)
    any_preserved = False
    for i, (_, (xa, ya, za), _, _) in enumerate(r3["hits"], start=1):
        pres, bad, tot, wit = preserves_code_space(xa, ya, za, HX, HZ)
        wstr = "yes" if wit is not None else "no"
        print(f"  triple {i}: preserves code space = {pres}; "
              f"non-constant on {bad}/{tot} cosets; Q(u)!=Q(0) witness: {wstr}")
        any_preserved |= pres
    assert not any_preserved, "manuscript claims (H3) fails for all four"
    print("  => (H3) fails for all four admissible triples")

    # ---------------- positive example: [[12,3,1]] satisfying all conditions
    print("\n" + "=" * 68)
    print("Section VI-B: a code satisfying admissibility, saturation, (S1)-(S3)")
    print("=" * 68)
    np12 = 12
    Rg = [list(range(0, 3)), list(range(3, 6)), list(range(6, 9)), list(range(9, 12))]
    xp = np.zeros(np12, np.uint8); yp = np.zeros(np12, np.uint8); zp = np.zeros(np12, np.uint8)
    for i in Rg[0]: xp[i] = yp[i] = 1
    for i in Rg[1]: xp[i] = zp[i] = 1
    for i in Rg[2]: yp[i] = zp[i] = 1
    for i in Rg[3]: xp[i] = yp[i] = zp[i] = 1
    gp = np.zeros(np12, np.uint8)
    for i in [0, 1, 3, 4, 6, 7, 9, 10]: gp[i] = 1
    HXp = gp[None, :]
    Dp = schur_span(HXp)
    s1p, s2p, s3p = stability_criterion(xp, yp, zp, HXp, Dp)
    assert admissible(xp, yp, zp) and s1p and s2p and s3p
    # region-parity lemma: g meets each region evenly
    for R in Rg:
        assert int(gp[np.array(R)].sum()) % 2 == 0
    HZp = row_reduce_basis(list(nullspace(np.vstack([HXp, xp, yp, zp]))), np12)
    assert not ((HXp @ HZp.T) % 2).any()
    kp = np12 - rank2(HXp) - rank2(HZp)
    LXp = logical_basis(HXp, HZp); LZp = logical_basis(HZp, HXp)
    def _md(reps, stab, n):
        best = n + 1
        for bits in itertools.product([0, 1], repeat=len(reps)):
            if not any(bits): continue
            v = np.zeros(n, np.uint8)
            for b, r in zip(bits, reps):
                if b: v = (v + r) % 2
            for sv in stab: best = min(best, int(((v + sv) % 2).sum()))
        return best
    dXp = _md(LXp, span(HXp), np12); dZp = _md(LZp, span(HZp), np12)
    presp = preserves_code_space(xp, yp, zp, HXp, HZp)[0]
    gp_ct, incp, mismp = synthesize(xp, yp, zp)
    assert presp and mismp == 0 and incp == 37
    print(f"  [[{np12},{kp},{min(dXp, dZp)}]]  dX={dXp} dZ={dZp}  "
          f"(S1,S2,S3)=({s1p},{s2p},{s3p})  code space preserved={presp}")
    print(f"  synthesis: {gp_ct} gates, Delta={incp}=1+3(3)+3(3)^2, phase mismatches={mismp}")
    print(f"  Proposition (stability bounds distance): every stabilizer meets "
          f"each region evenly; d={min(dXp, dZp)} <= 3p=9")

    # ---------------- claims made in Sections II-III ---------------------
    print("\n" + "=" * 68)
    print("Section II: (S1) holds and (S3) holds, yet (S2) fails")
    print("=" * 68)
    # C_Z = {0}, C_X = span{(1,1,1,1)}; x,y,z as in Example 1.
    HXc = np.array([[1, 1, 1, 1]], np.uint8)
    Dc = schur_span(HXc)
    g = np.array([1, 1, 1, 1], np.uint8)
    s1c, s2c, s3c = stability_criterion(x4, y4, z4, HXc, Dc)
    print(f"  admissible={admissible(x4, y4, z4)}  "
          f"classes independent mod C_X="
          f"{rank2(np.vstack([HXc, x4, y4, z4])) == 4}")
    print(f"  (S1)={s1c}  (S2)={s2c}  (S3)={s3c}   [dim D={rank2(Dc)}, wt(g)={int(g.sum())}]")
    print(f"  <x,g>={ip(x4, g)} <y,g>={ip(y4, g)} <z,g>={ip(z4, g)}")
    print(f"  tau(x,y,z)={tau(x4, y4, z4)}  "
          f"tau(x,y+g,z+g)={tau(x4, (y4 + g) % 2, (z4 + g) % 2)}")
    assert s1c and s3c and not s2c and tau(x4, (y4+g) % 2, (z4+g) % 2) != tau(x4, y4, z4)

    print("\n" + "=" * 68)
    print("Section III: both hypotheses of the conjugation proposition bind")
    print("=" * 68)
    # saturation alone does not force odd weights
    e2 = np.array([1, 1], np.uint8)
    r2 = regions(e2, e2, e2)
    print(f"  x=y=z=(1,1): saturated={r2['Px'] == r2['Py'] == r2['Pz'] == 0}, "
          f"wt={int(e2.sum())} (even), admissible={admissible(e2, e2, e2)}")
    # admissibility alone (unsaturated) also permits even weight
    xu = np.array([0, 1, 1, 1, 1], np.uint8)
    yu = np.array([1, 0, 1, 1, 0], np.uint8)
    zu = np.array([1, 1, 0, 1, 0], np.uint8)
    ru = regions(xu, yu, zu)
    mism = 0
    for bits in itertools.product([0, 1], repeat=5):
        c = np.array(bits, np.uint8)
        Qc = (ip(xu, c) * ip(yu, c) * ip(zu, c)) % 2
        Qcx = (ip(xu, (c + xu) % 2) * ip(yu, (c + xu) % 2)
               * ip(zu, (c + xu) % 2)) % 2
        mism += ((Qc + Qcx) % 2 != (ip(yu, c) * ip(zu, c)) % 2)
    print(f"  x=(0,1,1,1,1),y=(1,0,1,1,0),z=(1,1,0,1,0): "
          f"admissible={admissible(xu, yu, zu)}, "
          f"saturated={ru['Px'] == ru['Py'] == ru['Pz'] == 0}, "
          f"wt(x)={int(xu.sum())} (even); conjugation identity fails on "
          f"{mism}/32 basis states")
    assert admissible(xu, yu, zu) and ru['Px'] == 1 and int(xu.sum()) % 2 == 0 and mism > 0

    # pairwise orthogonality is needed for the conjugation identity itself
    Zop = np.array([[1, 0], [0, -1]])
    Xop = np.array([[0, 1], [1, 0]])
    print(f"  x=y=z=(1): synthesis gives U=Z; Z X Z^dag = "
          f"{(Zop @ Xop @ Zop.conj().T).tolist()}, "
          f"formula would give X.D_yz = {(Xop @ Zop).tolist()}")
    assert not np.array_equal(Zop @ Xop @ Zop.conj().T, Xop @ Zop)

    # incidence equality at even p_0 for a non-admissible saturated triple
    n8 = 8
    xa = np.zeros(n8, np.uint8); ya = np.zeros(n8, np.uint8); za = np.zeros(n8, np.uint8)
    for i in (0, 1): xa[i] = ya[i] = 1          # P_xy
    for i in (2, 3): xa[i] = za[i] = 1          # P_xz
    for i in (4, 5): ya[i] = za[i] = 1          # P_yz
    for i in (6, 7): xa[i] = ya[i] = za[i] = 1  # P_xyz
    ga, inca, bada = synthesize(xa, ya, za)
    print(f"  even p_0=2 saturated triple: admissible={admissible(xa, ya, za)}, "
          f"{ga} gates, max incidence {inca} = 1+3(2)+3(4) = {1 + 6 + 12}, "
          f"mismatches {bada}")
    assert inca == 19 and bada == 0

    print("\n" + "=" * 68)
    print("Section III-C: monomial count without saturation")
    print("=" * 68)
    from collections import Counter
    # seven idempotent region variables: bx,by,bz (private), a,b,c (pairwise), e (triple)
    lx = [frozenset([0]), frozenset([3]), frozenset([4]), frozenset([6])]
    ly = [frozenset([1]), frozenset([3]), frozenset([5]), frozenset([6])]
    lz = [frozenset([2]), frozenset([4]), frozenset([5]), frozenset([6])]
    cnt = Counter()
    for u in lx:
        for v in ly:
            for w in lz:
                cnt[u | v | w] += 1
    surv = [m for m, k in cnt.items() if k % 2]
    sat_only = [m for m in surv if not (m & {0, 1, 2})]
    print(f"  surviving monomials: {len(surv)}; involving no private region: "
          f"{len(sat_only)}; extra beyond the saturated case: "
          f"{len(surv) - len(sat_only)}")
    print(f"  bx.by.bz survives: {frozenset([0, 1, 2]) in surv}")
    assert len(sat_only) == 7 and len(surv) - len(sat_only) == 31

    # ---------------- criterion vs brute force --------------------------
    print("\n" + "=" * 68)
    print("Implementation sanity check: stability criterion vs brute force")
    print("(the proposition is established analytically; these cases only")
    print(" check that the coded criterion matches direct enumeration)")
    print("=" * 68)
    rng = np.random.default_rng(7)

    agree = pos = trials = 0
    while trials < 300:                       # unrestricted random CSS codes
        nn = 8
        HXr = rng.integers(0, 2, size=(2, nn), dtype=np.uint8)
        if not HXr.any():
            continue
        CXr, Dr = span(HXr), schur_span(HXr)
        xr, yr, zr = (rng.integers(0, 2, size=nn, dtype=np.uint8) for _ in range(3))
        s = stability_criterion(xr, yr, zr, HXr, Dr)
        pred = all(s)
        agree += (pred == brute_tau_stable(xr, yr, zr, CXr))
        pos += pred
        trials += 1
    print(f"  300 unrestricted random cases: {agree}/300 agree, "
          f"{pos} satisfied (S1)-(S3)")

    agree = pos = trials = 0
    while trials < 400:                       # structured C_X, low-weight rows
        nn = 7
        HXr = np.zeros((2, nn), np.uint8)
        idx = rng.permutation(nn)
        HXr[0, idx[:2]] = 1
        HXr[1, idx[2:4]] = 1
        CXr, Dr = span(HXr), schur_span(HXr)
        xr, yr, zr = (rng.integers(0, 2, size=nn, dtype=np.uint8) for _ in range(3))
        s = stability_criterion(xr, yr, zr, HXr, Dr)
        pred = all(s)
        agree += (pred == brute_tau_stable(xr, yr, zr, CXr))
        pos += pred
        trials += 1
    print(f"  400 structured random cases:   {agree}/400 agree, "
          f"{pos} satisfied (S1)-(S3)")

    agree = trials = 0
    while trials < 300:                       # pairwise-orthogonality stability
        nn = 8
        HXr = rng.integers(0, 2, size=(2, nn), dtype=np.uint8)
        if not HXr.any():
            continue
        CXr = span(HXr)
        xr, yr, zr = (rng.integers(0, 2, size=nn, dtype=np.uint8) for _ in range(3))
        pred = perp([xr, yr, zr], HXr) and all(ip(u, v) == 0
                                               for u in CXr for v in CXr)
        agree += (pred == brute_orth_stable(xr, yr, zr, CXr))
        trials += 1
    print(f"  300 pairwise-orthogonality cases: {agree}/300 agree")
