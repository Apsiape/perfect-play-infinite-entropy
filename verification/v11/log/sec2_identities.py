"""Item 3: numerical second engine for the algebraic steps of Theorem (20).

Representation: canonical Catalan rep  (+)  a NON-canonical "loop" sector
(left-infinite word ...ABAB with suffix v, identified via base.AB = base; P=0
there).  Both satisfy (18) exactly on vectors away from the length cutoff.
States: K (coefficient matrix, Omega = sum K_ij |i>|j>) = canonical height cutoff
plus random noise of size s over both sectors and mismatched pairs.

Checks, for each sample:
 (a) recurrence defect |e_k| <= d2 + d2^2 + 2 d4 + d4^2 + 16 d5^2 <= 20 delta;
     a_k <= (k+1) 2^-k + 80 delta;
 (b) exact identity (31) in convention (b): T_H = (1/2) a(x)a + b(x)b with a = A on
     heights <= H-2, b = B on heights 2..H, Omega_H = D_{<=H} Omega:
        (I - T_H) Omega_H = xi_0 + xi_H + r_H,
     xi_0 = (P(x)P) Omega, xi_H = (E_H (x) E_H) Omega, ||r_H|| <= delta;
 (c) height matching: ||(I - D_inf) Omega||^2 <= 4^{k+1} d5^2 + a_{k+1}  (any k);
 (d) the mirror defect of the pure canonical cutoff equals q_H/2 (sharpness).
"""
import numpy as np, itertools, json, math
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

LMAX = 12          # operator truncation (word length)
LSUP = 9           # support of states (word length)
rng = np.random.default_rng(20261103)


def canonical_words(L):
    out = [("", 0, 0)]  # (word, height, nA)
    frontier = [("", 0, 0)]
    for _ in range(L):
        new = []
        for w, h, n in frontier:
            new.append((w + "A", h + 1, n + 1))
            if h >= 2:
                new.append((w + "B", h - 1, n))
        out += new; frontier = new
    return out


def loop_words(L):
    # v legal from absolute height 1; reduced: not starting with "AB"
    out = [("", 1)]
    frontier = [("", 1)]
    for _ in range(L):
        new = []
        for v, h in frontier:
            for c in "AB":
                if c == "B" and h < 2:
                    continue
                nv = v + c
                if nv.startswith("AB"):
                    continue  # identified with a shorter word (only v="A"+"B" -> "")
                new.append((nv, h + 1 if c == "A" else h - 1))
        out += new; frontier = new
    return out


can = canonical_words(LMAX)
lp = loop_words(LMAX)
basis = [("c", w) for w, h, n in can] + [("l", v) for v, h in lp]
height = np.array([h for w, h, n in can] + [h for v, h in lp])
length = np.array([len(w) for w, h, n in can] + [len(v) for v, h in lp])
nA = np.array([n for w, h, n in can] + [-1 for v, h in lp])
index = {b: i for i, b in enumerate(basis)}
d = len(basis)
print("basis size", d, "(canonical", len(can), ", loop", len(lp), ")")


def build(letter):
    rows, cols = [], []
    for i, (s, w) in enumerate(basis):
        h = height[i]
        if letter == "B" and h < 2:
            continue
        if s == "c":
            t = ("c", w + letter)
        else:
            nv = w + letter
            if nv == "AB":
                nv = ""
            t = ("l", nv)
        if t in index:
            rows.append(index[t]); cols.append(i)
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(d, d))


A = build("A"); B = build("B")
C = sp.diags(np.where(True, 2.0 ** (-height.astype(float)), 0))
P = sp.diags((height == 0).astype(float)); Q = sp.diags((height == 1).astype(float))
inner = length <= LMAX - 2
# relation check on inner vectors
I = sp.identity(d)
def relerr(M):
    M = M.tocsc()[:, np.where(inner)[0]]
    return abs(M).max() if M.nnz else 0.0
print("relation errors on inner vectors:",
      relerr(A.T @ A - I), relerr(A @ A.T + B @ B.T - (I - P)), relerr(B.T @ B - (I - P - Q)),
      relerr(C @ A - 0.5 * A @ C), relerr(C @ B - 2 * B @ C))

Fdiag = lambda k: sp.diags((height >= k).astype(float))
Ediag = lambda h: sp.diags((height == h).astype(float))
sup = np.where(length <= LSUP)[0]


def defects(K):
    d1 = 0.5 * (A @ K) - K @ A          # (A_A/2) Om - A_B^* Om   ;  (I(x)Y)Om <-> K Y^T, Y=A^*: K A
    d2 = A.T @ K - 0.5 * (K @ A.T)      # A_A^* Om - (A_B/2) Om   ;  Y = A: K A^T
    d3 = B @ K - K @ B                  # B_A Om - B_B^* Om
    d4 = B.T @ K - K @ B.T              # B_A^* Om - B_B Om
    d5 = C @ K - K @ C
    return [spnorm(x) for x in (d1, d2, d3, d4, d5)], (d1, d2, d3, d4, d5)


def cutoff_state(H, nmax):
    K = sp.lil_matrix((d, d))
    for i, (s, w) in enumerate(basis):
        if s == "c" and length[i] <= LSUP:
            # all prefixes height <= H
            hh, ok = 0, True
            for c in w:
                hh += 1 if c == "A" else -1
                ok &= hh <= H
            if ok:
                K[i, i] = 2.0 ** (-nA[i])
    return K.tocsr()


results = []
for H in (2, 3, 4, 5):
    Kc = cutoff_state(H, None)
    for s in (0.0, 1e-4, 1e-3, 1e-2, 5e-2):
        K = Kc.copy()
        if s > 0:
            idx = rng.choice(sup, size=(400, 2))
            noise = sp.csr_matrix((rng.normal(size=400), (idx[:, 0], idx[:, 1])), shape=(d, d))
            K = K / spnorm(K) + s * noise / spnorm(noise)
        K = (K / spnorm(K)).tocsr()
        dn, dv = defects(K)
        Delta = sum(x * x for x in dn); delta = math.sqrt(Delta)
        # (a) recurrence
        a = [spnorm(Fdiag(k) @ K) ** 2 for k in range(0, 14)]
        e = [a[k + 1] - a[k] + a[k - 1] / 4 for k in range(1, 13)]
        ebound = dn[1] + dn[1] ** 2 + 2 * dn[3] + dn[3] ** 2 + 16 * dn[4] ** 2
        a_ok = all(a[k] <= (k + 1) * 2.0 ** -k + 80 * delta + 1e-12 for k in range(14))
        # (b) identity (31), convention (b)
        D = lambda X, hmax: sum(Ediag(h) @ X @ Ediag(h) for h in range(0, hmax + 1))
        OmH = D(K, H)
        a_r = A @ sp.diags((height <= H - 2).astype(float))
        b_r = B @ sp.diags(((height >= 2) & (height <= H)).astype(float))
        TH = lambda X: 0.5 * (a_r @ X @ a_r.T) + b_r @ X @ b_r.T
        lhs = OmH - TH(OmH)
        xi0 = P @ K @ P
        xiH = Ediag(H) @ K @ Ediag(H)
        rH = lhs - xi0 - xiH
        # predicted r_H = - D_{<=H-1}(A d2' + B d4') with d2' = (A_B/2)Om - A_A^*Om = -d2, d4' = B_B Om - B_A^* Om = -d4
        r_full = A @ (-dv[1]) + B @ (-dv[3])
        rpred = -D(r_full, H - 1)
        id_err = spnorm(rH - rpred)
        # (c) height matching
        mism = spnorm(K - D(K, 20)) ** 2
        hm_ok = all(mism <= 4 ** (k + 1) * dn[4] ** 2 + a[k + 1] + 1e-12 for k in range(0, 12))
        row = dict(H=H, noise=s, Delta=Delta, delta=delta, max_e=max(abs(x) for x in e), e_bound=ebound,
                   e_ok=max(abs(x) for x in e) <= ebound + 1e-12, ebound_le_20delta=ebound <= 20 * delta + 1e-15,
                   a_k_bound_ok=a_ok, identity31_err=id_err, rH=spnorm(rH), rH_le_delta=spnorm(rH) <= delta + 1e-12,
                   mismatch=mism, heightmatch_ok=hm_ok)
        results.append(row)
        print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})

# (d) sharpness Delta = q/2 for the pure canonical cutoff restricted to n <= nmax (prefix- and B-closed)
print("--- sharpness: Delta(S) vs q_S/2 for finite prefix-closed B-closed S ---")
for H in (1, 2, 3, 4, 5):
    diag = cutoff_state(H, None).diagonal()
    for m in (2, 3, 4):
        keep = (diag > 0) & (nA <= m)
        vals = np.where(keep, diag, 0.0)
        Z = np.sum(vals ** 2)
        bd = np.sum(np.where(keep & ((height == H) | (nA == m)), vals ** 2, 0.0))
        q = bd / Z
        Km = sp.diags(vals / math.sqrt(Z)).tocsr()
        dn, _ = defects(Km)
        print(f"H={H} nA<={m}: Delta={sum(x*x for x in dn):.12f}  q/2={q/2:.12f}  B-defects={dn[2]:.1e},{dn[3]:.1e} C={dn[4]:.1e}")
json.dump(results, open("sec2_identities.json", "w"), indent=1, default=float)
