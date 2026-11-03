"""Rule-only S_4 witness: n0 = 7 (u0 = AAAABABABA), N = 18, no partial pruning. Two engines."""
from s4_counterexample import modified, engine2, E, F
import mpmath as mp, json
mp.mp.dps = 60
Z4 = F(129, 80); M4 = F(67, 40); E4 = E(Z4, M4)
for (n0, N, k) in ((7, 18, 0), (6, 15, 17016), (4, 9, 136)):
    Z, M, b, P = modified(N, n0, k)
    Z2, M2, b2 = engine2(N, n0, k, nmax=300)
    Ep = E(Z, M)
    print(dict(n0=n0, N=N, k=k, Z=str(Z), M=str(M), b=str(b), q=str(b / Z), q_lt_1_129=b / Z < F(1, 129),
               E_new=mp.nstr(Ep, 20), E4=mp.nstr(E4, 20), dE=mp.nstr(Ep - E4, 10),
               engine2_maxdiff=float(max(abs(Z - Z2), abs(M - M2), abs(b - b2)))))
