# Exact free-algebra check of the out5 Section 3 identity, arbitrary effects:
#   F = S T + k sum_i (E_i + F_i - 2 E_i F_i)
#     = (S^2+T^2)/2 + sum_i D_i^2 + 1/2 sum_{i<j} (a_j D_i - a_i D_j)^2 + k sum_i (E_i - E_i^2 + F_i - F_i^2)
# with S = a0 + sum a_i E_i, T = a0 + sum a_i F_i, D_i = E_i - F_i, k = 1 + 1/2 sum a_i^2.
# Only cross-party commutation is used (Alice letters commute with Bob letters);
# no idempotency, no same-party commutation.  Normal form: (Alice word, Bob word).
import random
from fractions import Fraction as Fr
from collections import defaultdict

def mul(p, q):
    r = defaultdict(Fr)
    for (a1, b1), c1 in p.items():
        for (a2, b2), c2 in q.items():
            r[(a1 + a2, b1 + b2)] += c1 * c2
    return {k: v for k, v in r.items() if v}
def add(*ps):
    r = defaultdict(Fr)
    for p in ps:
        for k, v in p.items(): r[k] += v
    return {k: v for k, v in r.items() if v}
def sc(c, p): return {k: c * v for k, v in p.items() if c * v}
ONE = {((), ()): Fr(1)}
def E(i): return {((i,), ()): Fr(1)}
def F(i): return {((), (i,)): Fr(1)}

random.seed(20261102)
ok = True
for trial in range(6):
    m = random.randint(2, 5)
    a0 = Fr(random.randint(-9, 9), random.choice([1, 2, 4, 8, 16, 32]))
    al = [Fr(random.randint(-9, 9), random.choice([1, 2, 4, 8, 16, 32])) for _ in range(m)]
    k = 1 + sum(x * x for x in al) / 2
    S = add(sc(a0, ONE), *[sc(al[i], E(i)) for i in range(m)])
    T = add(sc(a0, ONE), *[sc(al[i], F(i)) for i in range(m)])
    D = [add(E(i), sc(-1, F(i))) for i in range(m)]
    lhs = add(mul(S, T), *[sc(k, add(E(i), F(i), sc(-2, mul(E(i), F(i))))) for i in range(m)])
    rhs = add(sc(Fr(1, 2), add(mul(S, S), mul(T, T))), *[mul(D[i], D[i]) for i in range(m)])
    for i in range(m):
        for j in range(i + 1, m):
            L = add(sc(al[j], D[i]), sc(-al[i], D[j]))
            rhs = add(rhs, sc(Fr(1, 2), mul(L, L)))
    for i in range(m):
        rhs = add(rhs, sc(k, add(E(i), sc(-1, mul(E(i), E(i))), F(i), sc(-1, mul(F(i), F(i))))))
    diff = add(lhs, sc(-1, rhs))
    print('trial', trial, 'm', m, 'identity exact:', not diff)
    ok &= not diff
print('ALL OK' if ok else 'FAIL')
