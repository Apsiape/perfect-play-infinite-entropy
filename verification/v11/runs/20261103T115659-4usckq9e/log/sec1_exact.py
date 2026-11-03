"""Item 1: exact rational checks of the boundary-class formulas.

Engine 1 (generating functions): exact Fraction linear solves for path sums
in the height-H cutoff W_H, at x = 1/4, plus the x d/dx moment.
Engine 2 (DP over (A-count n, height h)): exact Fraction partial sums,
with the tail certified by the geometric decay rate; and, for small H,
brute-force word enumeration.

Checks m_h, Z_H, b_H, q_H, M_H, E(S_H), and the top-source path sum Z_{H,H}.
"""
from fractions import Fraction as F
import math, json, sys

X = F(1, 4)


def solve(Mat, rhs):
    """Exact Gaussian elimination on Fractions."""
    n = len(rhs)
    A = [row[:] + [rhs[i]] for i, row in enumerate(Mat)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        piv = A[c][c]
        A[c] = [v / piv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)]


def band_sums(H, lo=1):
    """Path sums from each start height h in [lo..H], walk confined to [lo..H]
    (A up allowed if h<H; B down allowed if h>lo).  Returns dicts
    Z[h] = sum x^{#A}, Y[h] = sum #A x^{#A}, Bd[h] = sum over paths ending at H,
    and G[h][j] = sum over paths ending at j."""
    hs = list(range(lo, H + 1))
    n = len(hs)
    idx = {h: i for i, h in enumerate(hs)}
    # (I - K) Z = 1 with K_{h,h+1} = x, K_{h,h-1} = 1
    Mat = [[F(0)] * n for _ in range(n)]
    for h in hs:
        i = idx[h]
        Mat[i][i] += 1
        if h < H:
            Mat[i][idx[h + 1]] -= X
        if h > lo:
            Mat[i][idx[h - 1]] -= 1
    Z = solve(Mat, [F(1)] * n)
    # Y = x*K_up (Z + Y) + K_down Y  ->  (I-K) Y = x * Z_{h+1}
    rhs = [X * Z[idx[h + 1]] if h < H else F(0) for h in hs]
    Y = solve(Mat, rhs)
    G = {}
    for j in hs:
        e = [F(1) if h == j else F(0) for h in hs]
        G[j] = solve(Mat, e)
    Zd = {h: Z[idx[h]] for h in hs}
    Yd = {h: Y[idx[h]] for h in hs}
    Gd = {h: {j: G[j][idx[h]] for j in hs} for h in hs}
    Bd = {h: Gd[h][H] for h in hs}
    return Zd, Yd, Bd, Gd


def cutoff_quantities(H):
    """W_H from the root: empty word, then A to height 1, then band [1..H]."""
    Zb, Yb, Bdb, Gb = band_sums(H)
    Z = 1 + X * Zb[1]
    M = X * (Zb[1] + Yb[1])
    b = X * Bdb[1]
    m = {0: F(1)}
    for h in range(1, H + 1):
        m[h] = X * Gb[1][h]
    return Z, M, b, m, Zb[H]


def dp_partial(H, nmax):
    """Engine 2: DP over A-count n.  c[n][h] = # words in W_H with n A's at height h."""
    Z = F(0); M = F(0); b = F(0)
    m = [F(0)] * (H + 1)
    cur = [0] * (H + 1)  # entries at count n (after an A step), before B chains
    cur[0] = 1
    for n in range(nmax + 1):
        # B-chains: c[h] = entry[h] + c[h+1] for h>=1 (B from h+1 to h needs h+1>=2)
        c = [0] * (H + 2)
        for h in range(H, -1, -1):
            c[h] = cur[h] + (c[h + 1] if h >= 1 else 0)
        w = F(1, 4 ** n)
        for h in range(H + 1):
            Z += c[h] * w
            M += n * c[h] * w
            m[h] += c[h] * w
        b += c[H] * w
        nxt = [0] * (H + 1)
        for h in range(H):
            nxt[h + 1] = c[h]
        cur = nxt
    return Z, M, b, m


def brute(H, maxlen):
    """Engine 3: enumerate words of length <= maxlen in W_H."""
    Z = F(0); M = F(0); b = F(0)
    stack = [(0, 0, 0)]  # (height, nA, length)
    while stack:
        h, n, L = stack.pop()
        w = F(1, 4 ** n)
        Z += w; M += n * w
        if h == H:
            b += w
        if L < maxlen:
            if h < H:
                stack.append((h + 1, n + 1, L + 1))
            if h >= 2:
                stack.append((h - 1, n, L + 1))
    return Z, M, b


def E_of(Z, M):
    return math.log2(Z) + 2 * float(M / Z)


def main():
    out = {"rows": []}
    ok = True
    for H in range(1, 31):
        Z, M, b, m, ZHH = cutoff_quantities(H)
        q = b / Z
        claims = {
            "Z": F(2 * H * 2 ** H + 1, (H + 1) * 2 ** H),
            "b": F(1, (H + 1) * 2 ** H),
            "q": F(1, H * 2 ** (H + 1) + 1),
            "M": F(2 * H * (H - 1), 3 * (H + 1)) + F(H * (H + 5), 6 * (H + 1) * 2 ** H),
        }
        mclaim_ok = all(m[h] == F(H + 1 - h, (H + 1) * 2 ** h) for h in range(H + 1))
        checks = {k: (v == {"Z": Z, "b": b, "q": q, "M": M}[k]) for k, v in claims.items()}
        ZHH_claim = F(2 ** (H + 1) - H - 2, H)
        ZHH_mine = F(2 * (2 ** (H + 1) - H - 2), H + 1)
        E = E_of(Z, M)
        row = dict(H=H, E=E, E_minus_2H_over_3=E - 2 * H / 3, m_h_ok=mclaim_ok,
                   Z_ok=checks["Z"], b_ok=checks["b"], q_ok=checks["q"], M_ok=checks["M"],
                   Z_HH=str(ZHH), Z_HH_claimed=str(ZHH_claim), Z_HH_matches_claim=(ZHH == ZHH_claim),
                   Z_HH_matches_2x2pow_over_Hp1=(ZHH == ZHH_mine))
        ok &= mclaim_ok and all(checks.values())
        out["rows"].append(row)
        print(f"H={H:2d} m_h:{mclaim_ok} Z:{checks['Z']} b:{checks['b']} q:{checks['q']} M:{checks['M']} "
              f"E={E:.6f} E-2H/3={E-2*H/3:.6f}  Z_HH={float(ZHH):.4f} claim={float(ZHH_claim):.4f} "
              f"mine-formula:{ZHH == ZHH_mine}")
    # Engine 2 and 3 cross-checks
    print("--- engine 2 (DP partial sums) and engine 3 (brute force) ---")
    for H in [1, 2, 3, 4, 5, 8]:
        Z, M, b, m, _ = cutoff_quantities(H)
        nmax = 60 if H <= 5 else 120
        Zp, Mp, bp, mp = dp_partial(H, nmax)
        lam = math.cos(math.pi / (H + 1)) ** 2
        print(f"H={H} DP(nmax={nmax}): |Z-Zp|={float(Z-Zp):.3e} |M-Mp|={float(M-Mp):.3e} |b-bp|={float(b-bp):.3e}"
              f"  (lambda^nmax={lam**nmax:.2e}); m-diff max={max(float(m[h]-mp[h]) for h in range(H+1)):.2e}")
        if H <= 4:
            L = 22
            Zb, Mb, bb = brute(H, L)
            print(f"      brute(len<={L}): Z-Zb={float(Z-Zb):.3e} M-Mb={float(M-Mb):.3e} b-bb={float(b-bb):.3e}")
    # Z_{H,H}: length-DP (independent of the linear solve) for two conventions
    print("--- top-source path sum: (a) paths from H confined to [1..H]; (b) source at H, first step B, then [1..H-1] ---")
    for H in [1, 2, 3, 4, 5, 6, 10]:
        def lendp(start, top, Lmax=3000):
            v = {start: F(1)}; s = F(1)
            for _ in range(Lmax):
                nv = {}
                for h, a in v.items():
                    if h < top: nv[h+1] = nv.get(h+1, F(0)) + a * X
                    if h >= 2: nv[h-1] = nv.get(h-1, F(0)) + a
                v = nv; s += sum(v.values())
            return s
        a = lendp(H, H)
        bconv = 1 + (lendp(H-1, H-1) if H >= 2 else 0)
        print(f"H={H}: (a)={float(a):.6f} [2(2^(H+1)-H-2)/(H+1)={float(F(2*(2**(H+1)-H-2),H+1)):.6f}]  "
              f"(b)={float(bconv):.6f} [claimed (2^(H+1)-H-2)/H={float(F(2**(H+1)-H-2,H)):.6f}]")
    out["all_formulas_ok"] = ok
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "sec1_exact.json", "w"), indent=1)
    print("ALL boundary-class formulas exact for H=1..30:", ok)


if __name__ == "__main__":
    main()
