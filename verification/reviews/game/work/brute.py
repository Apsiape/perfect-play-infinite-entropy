"""Brute-force exact cross-check of the row gadget at cutoff N (default 0), for rows with d=16.
Builds the explicit D_t on C^16 (x) H_old (spectator qubits factor out exactly: D_t, R_k, and the
rowzero event act as identity on qubits 4..7 and the spectator marginal of Phi_256 is maximally mixed),
checks that {D_t} is a PVM commuting with the density, and evaluates with the reviewer's literal
predicate (gamespec.my_forbidden) every answer pair of every graph test and of the row-zero test.
usage: python -I brute.py N rowname [rowname ...]
"""
import sys, json, math, collections, time
import numpy as np
from fractions import Fraction as Fr
from pathlib import Path
HERE = Path(__file__).parent
N = int(sys.argv[1]); FLIP = [int(a.split('=')[1]) for a in sys.argv[2:] if a.startswith('--flip=')]
names = [a for a in sys.argv[2:] if not a.startswith('--')]
# reuse the reviewer's own old-model projection builder (top part of old_eval.py, no evaluation)
src = (HERE / 'old_eval.py').read_text()
src = src[:src.index('key = M.key')]
sys.argv = ['old_eval.py', str(N), '--noproj']
g = {'__file__': str(HERE / 'old_eval.py'), '__name__': 'builder'}
exec(compile(src, 'old_eval_builder', 'exec'), g)
F, np, M, proj, DEN, Q, G, core = g['F'], g['np'], g['M'], g['proj'], g['DEN'], g['Q'], g['G'], g['core']
import gamespec as GS
n = M.n
qint = np.array([int(M.kappa[c]) * 4**(N - int(M.nA[w])) for f in range(3) for c in range(9) for w in range(M.NW)], dtype=np.int64)
hden = 75 * M.ZN * 4**N
sv = []
for f in range(3):
    for c in range(9):
        for w in range(M.NW):
            base = {1: F.ONE, 2: F.SQRT2, 4: 2 * F.ONE}[int(M.kappa[c])]
            sv.append(base * 2**(N - int(M.nA[w])))
sv = np.array(sv).T
gad = GS.row_gadgets(G, Q); ix = GS.rule_index(G)
rows = core['projection_linear_relations']
byname = {rows[gg['row']]['name']: dr for dr, gg in gad.items()}
d = 16; nl = 4
def chi(t, i): return -1 if bin(t & i).count('1') % 2 else 1
# scalar low-qubit (first 4 qubits) matrices, exact rationals scaled by 32 (R_k entries 1/2, |+><+| entries 1/16)
def Rk(j, k):
    v0 = np.zeros(d, dtype=np.int64); v0[0] = 1
    vj = np.zeros(d, dtype=np.int64); vj[j] = 1
    if k == 0: return 16 * np.outer(v0 + vj, v0 + vj)        # 32 * |0+j><0+j|/2
    if k == 1: return 16 * np.outer(v0 - vj, v0 - vj)
    S = np.zeros((d, d), dtype=np.int64); S[0, 0] = S[j, j] = 1
    return 32 * (np.eye(d, dtype=np.int64) - S)
def pauliX(j):
    P = np.zeros((d, d), dtype=np.int64)
    for x in range(d): P[x ^ j, x] = 1
    return P
def pauliZ(l):
    return np.diag([chi(l, x) for x in range(d)]).astype(np.int64)
def check_Rk(j, kernel):
    """closed form R_k vs the joint spectral projections of X^j, Z^{l_k} named in the question meta"""
    Id = np.eye(d, dtype=np.int64)
    Z0 = Id * 1
    for l in kernel: Z0 = Z0 @ (Id + pauliZ(l))          # 2^(n-1) * prod (I+Z^l)/2
    Rp = (Id + pauliX(j)) @ Z0; Rm = (Id - pauliX(j)) @ Z0   # 2^n * R_pm
    return (np.array_equal(Rp * 2, Rk(j, 0)) and np.array_equal(Rm * 2, Rk(j, 1))
            and np.array_equal(32 * Id - 2 * Rp - 2 * Rm, Rk(j, 2)))
sys.path.insert(0, str(HERE))
def kronF(S, B):
    """scalar integer matrix S (d x d) (x) field matrix B (8,n,n) -> (8, d n, d n)"""
    return np.einsum('ab,kij->kaibj', S, B).reshape(8, d * n, d * n)
Kw = F.emul(np.tile(sv, (1, d))[:, :, None], np.tile(sv, (1, d))[:, None, :])  # k_i k_j * hden (old part), scalar part 1/d via Phi_d
res = {}
t0 = time.time()
for nm in names:
    dr = byname[nm]; gg = gad[dr]
    if gg['d'] != d: print(nm, 'skipped, d =', gg['d']); continue
    Vs = [None] * d
    I = np.zeros((8, n, n), dtype=np.int64); I[0] = DEN * np.eye(n, dtype=np.int64)
    Vs[0] = I
    for j, s in gg['slots'].items():
        if s['var'] is None: Vs[j] = s['sgn'] * I
        else:
            E = proj(s['var']); V = 2 * E; V[0] -= DEN * np.eye(n, dtype=np.int64); Vs[j] = s['sgn'] * V
    Vtrue = list(Vs)
    for jf in FLIP: Vs[jf] = -Vs[jf]   # negative control: prover uses a wrong column sign
    # noncommutativity witness among the row's projections
    vars_ = sorted({s['var'] for s in gg['slots'].values() if s['var'] is not None})
    ncpairs = 0
    for x in range(len(vars_)):
        for y in range(x + 1, len(vars_)):
            Px, Py = proj(vars_[x]), proj(vars_[y])
            if not np.array_equal(F.fmatmul(Px, Py), F.fmatmul(Py, Px)): ncpairs += 1
    VV = [[F.fmatmul(Vs[i], Vs[k]) for k in range(d)] for i in range(d)]   # den DEN^2
    Dt = []
    for t in range(d):
        A = np.zeros((8, d * n, d * n), dtype=np.int64)
        for i in range(d):
            for k in range(d):
                A[:, i*n:(i+1)*n, k*n:(k+1)*n] = chi(t, i) * chi(t, k) * VV[i][k]
        Dt.append(A)   # D_t = A / (d DEN^2)
    dD = d * DEN * DEN
    # PVM checks (exact)
    tot = sum(Dt); Iful = np.zeros_like(tot); Iful[0] = dD * np.eye(d * n, dtype=np.int64)
    pvm = bool(np.array_equal(tot, Iful))
    idem = all(np.array_equal(F.fmatmul(X, X), dD * X) for X in Dt)
    herm = all(np.array_equal(X, F.fadj(X)) for X in Dt)
    orth = 'implied (Hermitian idempotents summing to I are mutually orthogonal)'
    keyful = np.tile(M.key, d)
    comm = all(not np.any(X[:, keyful[:, None] != keyful[None, :]]) for X in Dt)
    # literal graph tests: Bob's row_joint effects E^e (x) R_k (Alice form)
    def prob(Aop, aden, Bop, bden):
        s = F.emul(F.emul(Kw, Aop), np.swapaxes(Bop, 1, 2)).sum(axis=(1, 2))
        assert not s[1:].any()
        return Fr(int(s[0]), aden * bden * d) / hden    # 1/d from the maximally mixed scalar marginal
    gtot = Fr(0); gpairs = 0
    for j, s in gg['slots'].items():
        jr = s['jr']; rules = ix[dr, jr]
        pq = Q['A'][s['pr']]; assert pq['j'] == j and check_Rk(j, pq['kernel'])
        Ev = proj(s['var']) if s['var'] is not None else I
        nE = -Ev.copy(); nE[0] += DEN * np.eye(n, dtype=np.int64)
        for b, (e, k) in enumerate(Q['B'][jr]['answers']):
            Bop = kronF(Rk(j, k), Ev if e == 1 else nE) if (s['var'] is not None or e == 1) else np.zeros((8, d*n, d*n), dtype=np.int64)
            for t in range(d):
                if GS.my_forbidden(Q, rules, dr, t, jr, b):
                    gtot += prob(Dt[t], dD, Bop, 32 * DEN); gpairs += 1
    # literal row-zero test against all 256 global-X answers: G_b = P^low_{b mod 16} (x) P^high, p = p_low/16
    gX, mask = gg['rowzero']; rules = ix[dr, gX]
    Hd = np.array([[chi(z, x) for z in range(d)] for x in range(d)], dtype=np.int64)  # columns = X-eigenvectors
    plow = {}
    for z in range(d):
        P = np.outer(Hd[:, z], Hd[:, z]) * 2   # 32 * |x_z><x_z| (entries +-1/16)
        plow[z] = prob(Dt[0], dD, kronF(P, I), 32 * DEN)
    rz = Fr(0)
    for b in range(256):
        for t in range(d):
            if GS.my_forbidden(Q, rules, dr, t, gX, b):
                assert t == 0
                rz += plow[b % d] / 16
    # D_t sync literally: sum_{t != t'} Tr(K D_t K D_t')
    sync = Fr(0)
    for a in range(d):
        for b in range(d):
            if a != b: sync += prob(Dt[a], dD, Dt[b], dD)
    res[nm] = dict(flip=FLIP, noncommuting_variable_pairs=ncpairs, pvm_sum=pvm, idempotent=idem, hermitian=herm, orthogonal=orth, commutes_h=comm,
                   graph_forbidden_pairs=gpairs, graph_loss=str(gtot), rowzero_loss=str(rz), D_sync_loss=str(sync))
    print(nm, res[nm], round(time.time() - t0, 1), flush=True)
json.dump(res, open(HERE / (f'brute-N{N}.json' if not FLIP else f'brute-N{N}-flip{FLIP[0]}.json'), 'w'), indent=1)
