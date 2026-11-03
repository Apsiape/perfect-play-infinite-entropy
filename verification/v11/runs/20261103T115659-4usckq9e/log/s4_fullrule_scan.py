"""Full-rule (no partial pruning) scan of the S_4 construction for larger n0:
checks the leading-order criterion (13/5) lambda log4/(-log lambda) > 19/4 asymptotically."""
from s4_counterexample import modified, E, F
import mpmath as mp, json
Z4 = F(129, 80); M4 = F(67, 40); E4 = E(Z4, M4)
rows = []
for n0 in range(4, 31):
    for N in range(n0 + 2, n0 + 80):
        Zm, Mm, bm, P = modified(N, n0)
        if bm / Zm <= F(1, 129):
            dE = E(Zm, Mm) - E4
            w0 = F(1, 4 ** n0)
            # normalized: dE / (w0 * n0)
            rows.append(dict(n0=n0, N=N, P_over_w0=float(P / w0), dE=float(dE),
                             dE_over_w0n0=float(dE / (mp.mpf(1) / 4 ** n0) / n0), q_strict=bool(bm / Zm < F(1, 129))))
            print(rows[-1]); break
json.dump(rows, open("s4_fullrule_scan.json", "w"), indent=1)
