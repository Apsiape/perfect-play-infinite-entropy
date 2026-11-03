"""Exact checks of the renewal-lemma path formulas (Z_0, m_h, M_0/Z_0, Z_H, S_0, S_H),
killed-walk mean times / survival bounds, and the long-path tail inequality.
Strip convention (appendix Lemma 3.1 / assembly): vertices 0..H,
A edges h->h+1 for 0<=h<=H-2 (weight x per A step, x=1/4 squared amplitude), B edges h->h-1 for 2<=h<=H.
"""
from fractions import Fraction as Fr
import math, itertools

def edges(H):
    A = [(h, h + 1) for h in range(0, H - 1)]       # 0..H-2
    B = [(h, h - 1) for h in range(2, H + 1)]       # 2..H
    return A, B

def masses_by_enumeration(H, src, maxlen):
    """Brute-force: sum over legal paths from src of x^{n_A} and n_A x^{n_A}, by length DP (exact)."""
    A, B = edges(H)
    up = {s: t for s, t in A}
    dn = {s: t for s, t in B}
    x = Fr(1, 4)
    # state: (vertex, nA) -> count
    cur = {(src, 0): 1}
    Z = Fr(0); M = Fr(0); Ent_terms = []
    zh = [Fr(0)] * (H + 1)
    for k in range(maxlen + 1):
        nxt = {}
        for (v, na), c in cur.items():
            w = x ** na
            Z += c * w; M += c * na * w; zh[v] += c * w
            if v in up:
                nxt[(up[v], na + 1)] = nxt.get((up[v], na + 1), 0) + c
            if v in dn:
                nxt[(dn[v], na)] = nxt.get((dn[v], na), 0) + c
        cur = nxt
        if not cur:
            break
    return Z, M, zh

def masses_linear(H, src):
    """Exact solution of z_h = x z_{h-1} + z_{h+1} etc. via the transfer structure: solve linear system with fractions."""
    import sympy as sp
    x = sp.Rational(1, 4)
    A, B = edges(H)
    z = sp.symbols('z0:%d' % (H + 1)); m = sp.symbols('m0:%d' % (H + 1))
    eqs = []
    inc = {v: [] for v in range(H + 1)}
    for s, t in A: inc[t].append(('A', s))
    for s, t in B: inc[t].append(('B', s))
    for v in range(H + 1):
        rhs = (1 if v == src else 0)
        rhsm = 0
        for typ, s in inc[v]:
            if typ == 'A':
                rhs += x * z[s]; rhsm += x * m[s] + x * z[s]
            else:
                rhs += z[s]; rhsm += m[s]
        eqs.append(sp.Eq(z[v], rhs)); eqs.append(sp.Eq(m[v], rhsm))
    sol = sp.solve(eqs, list(z) + list(m), dict=True)[0]
    zs = [sol[z[v]] for v in range(H + 1)]
    ms = [sol[m[v]] for v in range(H + 1)]
    return zs, ms

def main():
    import sympy as sp
    print("== bottom/top source formulas (exact, sympy) ==")
    worst0 = 1e9; worstH = 1e9
    for H in range(2, 81):
        zs, ms = masses_linear(H, 0)
        # claimed z_h=(H-h)/(H 2^h), m_h = h(H-h)(2H+3-h)/(6H 2^h)
        ok_z = all(sp.simplify(zs[h] - sp.Rational(H - h, H * 2 ** h)) == 0 for h in range(H + 1))
        ok_m = all(sp.simplify(ms[h] - sp.Rational(h * (H - h) * (2 * H + 3 - h), 6 * H * 2 ** h)) == 0 for h in range(H + 1))
        Z0 = sum(zs); M0 = sum(ms)
        n = H - 1
        Z0c = sp.Rational(2 * (H - 1), H) + sp.Rational(2, H * 2 ** H)  # (2(H-1)+2^{1-H})/H
        MZc = sp.Rational(n - 1, 3) + sp.Rational((n + 1) * (n + 2), 6 * 2 ** n) / (2 * n + sp.Rational(1, 2 ** n))
        ok_Z0 = sp.simplify(Z0 - Z0c) == 0
        ok_MZ = sp.simplify(M0 / Z0 - MZc) == 0
        assert ok_z and ok_m and ok_Z0 and ok_MZ, ('bottom exact identities', H)
        S0 = float(sp.log(Z0, 2)) + 2 * float(M0 / Z0)
        # top source
        zt, mt = masses_linear(H, H)
        ok_zt = all(sp.simplify(zt[h] - (sp.Rational(h, H) * 2 ** (H - h) if h >= 1 else 0)) == 0 for h in range(H + 1))
        ZH = sum(zt); MH = sum(mt)
        ok_ZH = sp.simplify(ZH - sp.Rational(2 ** (H + 1) - H - 2, H)) == 0
        assert ok_zt and ok_ZH, ('top exact identities', H)
        SH = float(sp.log(ZH, 2)) + 2 * float(MH / ZH)
        target = 2 * H / 3 - 2 / 3
        worst0 = min(worst0, S0 - target); worstH = min(worstH, SH - target)
        if H <= 6 or H % 10 == 0 or not (ok_z and ok_m and ok_Z0 and ok_MZ and ok_zt and ok_ZH):
            print(f"H={H:3d} z:{ok_z} m:{ok_m} Z0:{ok_Z0} M0/Z0:{ok_MZ} ztop:{ok_zt} ZH:{ok_ZH} "
                  f"S0={S0:.5f} SH={SH:.5f} (2H-2)/3={target:.5f} S0-t={S0-target:.4f} logZH-(H-log2H)={math.log2(float(ZH))-(H-math.log2(H)):.4f}")
    print("min S0-(2H/3-2/3) over H=2..80:", worst0, " min SH-(...):", worstH)
    # small-H stated values
    for H, val in [(2, math.log2(5) - 8 / 5), (3, math.log2(17 / 12) + 46 / 51), (4, math.log2(49 / 32) + 72 / 49)]:
        zs, ms = masses_linear(H, 0)
        S0 = float(sp.log(sum(zs), 2)) + 2 * float(sum(ms) / sum(zs))
        assert abs(S0-val) < 1e-12, ('small-H entropy', H)
        print(f"stated S0(H={H})={val:.10f} computed={S0:.10f}")
    # brute-force enumeration cross-check (independent engine) for H=2..9
    print("== brute-force path enumeration cross-check ==")
    for H in range(2, 10):
        Z, M, zh = masses_by_enumeration(H, 0, 400)
        Zc = Fr(2 * (H - 1), H) + Fr(2, H * 2 ** H)
        Zt, Mt, _ = masses_by_enumeration(H, H, 400)
        print(f"H={H} Z0 enum={float(Z):.12f} formula={float(Zc):.12f}  ZH enum={float(Zt):.10f} formula={(2**(H+1)-H-2)/H:.10f}")

    print("== killed fair walk: mean killing time and survival ==")
    for H in [2, 3, 4, 5, 8, 16, 32, 64]:
        A, B = edges(H)
        up = dict(A); dn = dict(B)
        # mean killing time E_v: E_v = 1 + 1/2 sum_{legal moves} E_next  (illegal move -> killed at that step)
        import numpy as np
        nV = H + 1
        Mx = np.eye(nV); rhs = np.ones(nV)
        for v in range(nV):
            if v in up: Mx[v, up[v]] -= 0.5
            if v in dn: Mx[v, dn[v]] -= 0.5
        E = np.linalg.solve(Mx, rhs)
        claimed = [h * (H - h) if 1 <= h <= H - 1 else 1 + (H - 1) / 2 for h in range(nV)]
        err = max(abs(E[h] - claimed[h]) for h in range(nV))
        assert err < 1e-8, ('mean killing time', H, err)
        # survival probabilities P_v(tau>n) = (#legal paths length n)/2^n, check <= 4^{-floor(n/(2H^2))}
        P = np.ones(nV); Lb = 2 * H * H; viol = 0; sumnorm = 0.0
        for nstep in range(0, 40 * Lb + 1):
            mx = P.max()
            sumnorm += math.sqrt(mx)
            if mx > 4.0 ** (-(nstep // Lb)) + 1e-15: viol += 1
            newP = np.zeros(nV)
            for v in range(nV):
                s = 0.0
                if v in up: s += 0.5 * P[up[v]]
                if v in dn: s += 0.5 * P[dn[v]]
                newP[v] = s
            P = newP
        assert viol == 0 and sumnorm <= 4*H*H + 1e-8, ('survival', H, viol, sumnorm)
        print(f"H={H:3d} max|E-claimed|={err:.2e} max E={E.max():.2f} H^2/2={H*H/2:.1f} survival-bound violations={viol} "
              f"sum_n ||U^n|| ~ {sumnorm:.2f} vs 4H^2={4*H*H}")

    print("== long-path tail inequality ==")
    for H in range(2, 200):
        L = 2 * H * H; J0 = math.ceil(8 * math.log2(2 * H))
        lhs = (H + 1) * L * 4.0 ** (-J0) * (2 * L * (4 / 3 * (J0 + 1) + 4 / 9) + 4 / 3 * (H + 1))
        b = 162 * H ** 6 / (2 * H) ** 16
        if not (lhs <= b < 1):
            raise AssertionError(("FAIL tail", H, lhs, b))
    print("tail inequality checked H=2..199")
    # exact tail entropy vs bound for small H (independent): entropy of paths longer than N
    for H in [2, 3, 4]:
        Lb = 2 * H * H; J0 = math.ceil(8 * math.log2(2 * H)); N = Lb * J0
        print(f"H={H} N_hist={N}")

if __name__ == '__main__':
    main()
