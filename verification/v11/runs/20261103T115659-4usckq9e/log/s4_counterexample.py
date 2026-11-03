"""Item 2: the S_4 counterexample, built explicitly and evaluated exactly.

S' = W_4, modified by
  (i) pruning: every height-3 word whose A-count is >= N loses its A-edge
      (the whole W_4 subtree below wA is removed; w becomes an A-boundary word);
      optionally also a prescribed number k of the height-3 words of count N-1
      (each such word is pruned together with the identical rule below it);
  (ii) extension: one height-4 word u0 with n0 A's (n0 < N-1) gets its A-child
      u0A (height 5) and the full ceiling-5 subtree below u0A.
S' is prefix-closed and B-closed.  Z, M, b are exact rationals (generating
functions in the band [1..3] and [1..5]); E = log2 Z + 2 M/Z is evaluated with
mpmath at 60 digits.  A second engine (direct DP over (count, height) with the
pruning rule applied literally, truncated at large count with a certified
geometric tail) cross-checks Z, M, b.
"""
from fractions import Fraction as F
import mpmath as mp
import itertools, json, sys
mp.mp.dps = 60
X = F(1, 4)


def solve(Mat, rhs):
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


def band(H):
    hs = list(range(1, H + 1)); n = len(hs)
    Mat = [[F(0)] * n for _ in range(n)]
    for i, h in enumerate(hs):
        Mat[i][i] += 1
        if h < H: Mat[i][i + 1] -= X
        if h > 1: Mat[i][i - 1] -= 1
    Z = solve(Mat, [F(1)] * n)
    Y = solve(Mat, [X * Z[i + 1] if hs[i] < H else F(0) for i in range(n)])
    Bd = solve(Mat, [F(1) if h == H else F(0) for h in hs])
    return ({h: Z[i] for i, h in enumerate(hs)}, {h: Y[i] for i, h in enumerate(hs)},
            {h: Bd[i] for i, h in enumerate(hs)})


Z3, Y3, B3 = band(3)
Z5, Y5, B5 = band(5)


def W4_counts(nmax):
    """c[n][h] for W_4 (unrestricted), n = 0..nmax."""
    out = []
    cur = [1, 0, 0, 0, 0]
    for n in range(nmax + 1):
        c = [0] * 6
        for h in range(4, -1, -1):
            c[h] = cur[h] + (c[h + 1] if h >= 1 else 0)
        out.append(c[:5])
        cur = [0] + c[:4]
    return out


def modified(N, n0=4, k_partial=0, extend=True):
    """Exact (Z, M, b, P) for S'.  k_partial: number of count-(N-1) height-3 words
    additionally pruned (each pruned word's subtree below wA removed; the word
    becomes boundary).  P = total pruned weight."""
    c = W4_counts(N)
    Z = M = b = F(0)
    for n in range(N):
        w = F(1, 4 ** n)
        Z += sum(c[n]) * w
        M += n * sum(c[n]) * w
        b += c[n][4] * w
    # entries at count N via an A step from count N-1 at height h -> h+1
    e = {h + 1: c[N - 1][h] for h in range(4)}
    wN = F(1, 4 ** N)
    P = F(0)
    for he, cnt in e.items():
        if cnt == 0: continue
        if he <= 3:
            Zr, Yr, Br = Z3[he], Y3[he], B3[he]
            Pr = B3[he]
        else:  # height-4 word at count N: itself (boundary), then B to 3, then band [1..3]
            Zr, Yr, Br = 1 + Z3[3], Y3[3], 1 + B3[3]
            Pr = B3[3]
        Z += cnt * wN * Zr
        M += cnt * wN * (N * Zr + Yr)
        b += cnt * wN * Br
        P += cnt * wN * Pr
    # partial pruning at count N-1 (height-3 words of count N-1): each such word w
    # loses its A-child wA (height 4, count N) and the subtree below it.
    if k_partial:
        assert k_partial <= c[N - 1][3]
        wA = F(1, 4 ** N)
        # subtree below a height-4 count-N word in the (count>=N) regime:
        Zr, Yr, Br = 1 + Z3[3], Y3[3], 1 + B3[3]
        Z -= k_partial * wA * Zr
        M -= k_partial * wA * (N * Zr + Yr)
        b -= k_partial * wA * Br
        P -= k_partial * wA * B3[3]          # pruned count>=N height-3 words that disappear
        b += k_partial * F(1, 4 ** (N - 1))   # w becomes boundary
        P += k_partial * F(1, 4 ** (N - 1))
    if extend:
        w0 = F(1, 4 ** n0)
        b -= w0                         # u0 no longer an A-boundary word
        Z += w0 / 4 * Z5[5]
        M += w0 / 4 * ((n0 + 1) * Z5[5] + Y5[5])
        b += w0 / 4 * B5[5]
    return Z, M, b, P


def E(Z, M):
    return mp.log(mp.mpf(Z.numerator) / Z.denominator, 2) + 2 * mp.mpf((M / Z).numerator) / (M / Z).denominator


def engine2(N, n0, k_partial, nmax=400):
    """Direct DP over (count n, height h, state) with the literal pruning rule.
    States: 'main' (ordinary W_4 words), 'ext' (inside the ceiling-5 extension).
    Words are tracked by multiplicity only; the k_partial pruned count-(N-1)
    height-3 words are removed by multiplicity.  Tail beyond nmax bounded by
    the decay (cos^2(pi/6))^n of the ceiling-5 part."""
    Z = M = b = F(0)
    # cur[(state, h)] = number of words entering count n at height h (after an A step)
    cur = {("main", 0): 1}
    ext_started = False
    for n in range(nmax + 1):
        # B chains at fixed count
        cnt = {}
        for st in ("main", "ext"):
            top = 5 if st == "ext" else 4
            run = 0
            for h in range(top, -1, -1):
                run = cur.get((st, h), 0) + (run if h >= 1 else 0)
                cnt[(st, h)] = run
            # B from height 1 is illegal: run resets below h=1 handled by (h>=1) guard
        w = F(1, 4 ** n)
        nxt = {}
        for (st, h), m in cnt.items():
            if m == 0: continue
            Z += m * w; M += n * m * w
            # A-edge allowed?
            if st == "main":
                if h == 4:
                    allowedA = 0
                elif h == 3 and n >= N:
                    allowedA = 0
                elif h == 3 and n == N - 1:
                    allowedA = m - k_partial
                else:
                    allowedA = m
                if n == n0 and h == 4:
                    # one of these words is u0: its A-child starts the extension
                    nxt[("ext", 5)] = nxt.get(("ext", 5), 0) + 1
                    boundary = m - 1
                else:
                    boundary = m - allowedA
                b += boundary * w
                if allowedA:
                    nxt[("main", h + 1)] = nxt.get(("main", h + 1), 0) + allowedA
            else:
                if h == 5:
                    b += m * w
                else:
                    nxt[("ext", h + 1)] = nxt.get(("ext", h + 1), 0) + m
        cur = nxt
    return Z, M, b


def main():
    Z4 = F(129, 80); b4 = F(1, 80); M4 = F(67, 40)
    E4 = E(Z4, M4)
    print("W_4: Z=", Z4, " b=", b4, " q=", b4 / Z4, " M=", M4, " E=", mp.nstr(E4, 20))
    # sanity: no pruning, no extension reproduces W_4
    Zs, Ms, bs, _ = modified(60, extend=False)
    print("check N=60 no-ext:", float(Zs - Z4), float(Ms - M4), float(bs - b4))
    # the two budget rates
    lam = mp.cos(mp.pi / 5) ** 2
    prune_rate = 1 - X * B3[3] * 0  # placeholder (not used)
    print("relative sums: R_{4,4}=", band(4)[0][4], " b_{4,4}=", band(4)[2][4],
          " R_{5,5}=", Z5[5], " b_{5,5}=", B5[5])
    print("prune rate  3/5 + 13/(5*129) =", F(3, 5) + F(13, 5 * 129),
          "; 1 - b44/4 + R44/(4*129) =", 1 - band(4)[2][4] / 4 + band(4)[0][4] / (4 * 129))
    print("extend rate 7/12 + 19/(4*129) =", F(7, 12) + F(19, 4 * 129),
          "; 1 - b55/4 + R55/(4*129) =", 1 - B5[5] / 4 + Z5[5] / (4 * 129))
    print("lambda=cos^2(pi/5)=", mp.nstr(lam, 15), " (3+sqrt5)/8=", mp.nstr((3 + mp.sqrt(5)) / 8, 15))
    val = F(13, 5) * lam * mp.log(4) / (-mp.log(lam))
    print("(13/5) lambda log4/(-log lambda) =", mp.nstr(val, 12), " vs 19/4 = 4.75")
    rows = []
    best = None
    for n0 in range(4, 13):
        for N in range(n0 + 2, n0 + 30):
            Zm, Mm, bm, P = modified(N, n0)
            q = bm / Zm
            if q <= F(1, 129):
                # use the remaining budget with partial pruning at count N-1
                c = W4_counts(N)
                # budget 129 b - Z is linear in k: solve for the largest admissible k
                Z0, M0, b0, _ = modified(N, n0, 0)
                Z1, M1, b1, _ = modified(N, n0, 1)
                f0 = 129 * b0 - Z0; sl = (129 * b1 - Z1) - f0
                kbest = min(c[N - 1][3], int((-f0) // sl)) if sl > 0 else c[N - 1][3]
                Zk, Mk, bk, Pk = modified(N, n0, kbest)
                Ek = E(Zk, Mk)
                Ef = E(Zm, Mm)
                row = dict(n0=n0, N=N, k_partial=kbest, q_le=(bk / Zk <= F(1, 129)),
                           q_minus_q4=float(bk / Zk - F(1, 129)), dE_full_rule=float(Ef - E4),
                           dE_with_partial=float(Ek - E4), pruned_weight_over_w0=float(Pk * 4 ** n0))
                rows.append(row)
                print(row)
                if best is None or Ek - E4 < best[0]:
                    best = (Ek - E4, n0, N, kbest, Zk, Mk, bk)
                break
    dE, n0, N, k, Zk, Mk, bk = best
    print("\nBEST:", dict(n0=n0, N=N, k=k, dE=float(dE)))
    print("  Z'=", Zk, "\n  M'=", Mk, "\n  b'=", bk, "\n  q'=", bk / Zk, " <= 1/129:", bk / Zk <= F(1, 129))
    print("  E'=", mp.nstr(E(Zk, Mk), 25), "\n  E4=", mp.nstr(E4, 25))
    # engine 2 cross-check of the best set
    Z2, M2, b2 = engine2(N, n0, k, nmax=300)
    print("engine2 (DP to count 300) diffs:", float(Zk - Z2), float(Mk - M2), float(bk - b2),
          " (tail <~ 300*cos^2(pi/6)^300 =", 300 * float(mp.cos(mp.pi / 6) ** 600), ")")
    json.dump(dict(rows=rows, best=dict(n0=n0, N=N, k=k, dE=float(dE), Z=str(Zk), M=str(Mk), b=str(bk),
                                        q=str(bk / Zk)), E4=float(E4)),
              open("s4_counterexample.json", "w"), indent=1)


if __name__ == "__main__":
    main()
