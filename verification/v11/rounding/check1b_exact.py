"""Exact Fraction re-check of z_h, m_h, Z_0, M_0/Z_0, top z_h, Z_H for H=2..80 (no sympy)."""
from fractions import Fraction as Fr
import math

def solve(Amat, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(Amat)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]

def masses(H, src):
    x = Fr(1, 4)
    nV = H + 1
    inc = {v: [] for v in range(nV)}
    for h in range(0, H - 1): inc[h + 1].append(('A', h))
    for h in range(2, H + 1): inc[h - 1].append(('B', h))
    # z_v - sum = [v==src]
    Am = [[Fr(0)] * nV for _ in range(nV)]
    b = [Fr(1 if v == src else 0) for v in range(nV)]
    for v in range(nV):
        Am[v][v] += 1
        for t, s in inc[v]:
            Am[v][s] -= (x if t == 'A' else 1)
    z = solve(Am, b)
    bm = [Fr(0)] * nV
    for v in range(nV):
        for t, s in inc[v]:
            if t == 'A': bm[v] += x * z[s]
    m = solve(Am, bm)
    return z, m

bad = []
for H in range(2, 81):
    z, m = masses(H, 0)
    okz = all(z[h] == Fr(H - h, H * 2 ** h) for h in range(H + 1))
    okm = all(m[h] == Fr(h * (H - h) * (2 * H + 3 - h), 6 * H * 2 ** h) for h in range(H + 1))
    Z0 = sum(z); M0 = sum(m); n = H - 1
    okZ = Z0 == Fr(2 * (H - 1), H) + Fr(2, H * 2 ** H)
    okMZ = M0 / Z0 == Fr(n - 1, 3) + Fr((n + 1) * (n + 2), 6 * 2 ** n) / (2 * n + Fr(1, 2 ** n))
    zt, mt = masses(H, H)
    okzt = all(zt[h] == (Fr(h, H) * 2 ** (H - h) if h >= 1 else 0) for h in range(H + 1))
    okZH = sum(zt) == Fr(2 ** (H + 1) - H - 2, H)
    S0 = math.log2(Z0) + 2 * float(M0 / Z0)
    if not (okz and okm and okZ and okMZ and okzt and okZH) or S0 < 2 * H / 3 - 2 / 3:
        bad.append((H, okz, okm, okZ, okMZ, okzt, okZH, S0))
print("exact formula failures H=2..80:", bad)

assert not bad, bad
