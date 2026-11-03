"""Sharp numerical test of the banding chain (B1)-(B19) and the row-repair lemma (Lemma 3.1) on perturbed
truncated canonical Catalan representations, with environment. Every claimed inequality is checked with
eta := eta_meas = max(delta, e, sqrt(sum of the nine selected band-mismatch squares)), i.e. exactly the facts the
derivation uses, so a missing term or wrong constant in a derivation would show up as a violation.
Run: python -I check2_banding.py
"""
import numpy as np, itertools, math, sys

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)

def catalan_rep(ell):
    words = [()]
    frontier = [()]
    ht = {(): 0}
    for k in range(ell):
        nf = []
        for w in frontier:
            h = ht[w]
            for c in 'AB':
                if c == 'B' and h < 2: continue
                w2 = w + (c,)
                ht[w2] = h + (1 if c == 'A' else -1)
                nf.append(w2)
        words += nf; frontier = nf
    idx = {w: i for i, w in enumerate(words)}
    n = len(words)
    A = np.zeros((n, n)); B = np.zeros((n, n))
    for w, i in idx.items():
        if len(w) < ell:
            A[idx[w + ('A',)], i] = 1
            if ht[w] >= 2: B[idx[w + ('B',)], i] = 1
    C = np.diag([2.0 ** (-ht[w]) for w in words])
    P = np.diag([1.0 if ht[w] == 0 else 0 for w in words])
    Q = np.diag([1.0 if ht[w] == 1 else 0 for w in words])
    nA = np.array([w.count('A') for w in words])
    return dict(A=A, B=B, C=C, P=P, Q=Q, nA=nA, n=n, ht=np.array([ht[w] for w in words]))

def herm(X): return (X + X.conj().T) / 2

def perturb_ops(rep, eps):
    n = rep['n']
    def g(): return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / math.sqrt(n)
    A = rep['A'] + eps * g(); A /= max(1, np.linalg.norm(A, 2))
    B = rep['B'] + eps * g(); B /= max(1, np.linalg.norm(B, 2))
    C = herm(rep['C'] + eps * g()); w, V = np.linalg.eigh(C); C = (V * np.clip(w, 0, 1)) @ V.conj().T
    def proj(P0):
        Hh = herm(g()); w, V = np.linalg.eigh(Hh); U = (V * np.exp(1j * eps * w)) @ V.conj().T
        return U @ P0 @ U.conj().T
    return dict(A=A, B=B, C=C, P=proj(rep['P']), Q=proj(rep['Q']))

def state(rep, eps, dE=2):
    n = rep['n']
    K = np.zeros((n, n, dE), complex)
    amp = 2.0 ** (-rep['nA'])
    for i in range(n): K[i, i, 0] = amp[i]
    K += eps * (rng.standard_normal(K.shape) + 1j * rng.standard_normal(K.shape)) / n
    return K / np.linalg.norm(K)

# operator actions on the 3-tensor K[i(L), j(R), e]
def L(X, K): return np.einsum('ab,bje->aje', X, K)
def R(Y, K): return np.einsum('ab,ibe->iae', Y, K)
def nrm(K): return np.linalg.norm(K)
def ip(K1, K2): return np.vdot(K1, K2)

def bands(Cm, t, N):
    w, V = np.linalg.eigh(Cm)
    def ind(lo, hi):
        m = (w >= lo) & (w < hi)
        return (V[:, m] @ V[:, m].conj().T)
    E = [ind(t, np.inf)] + [ind(t * 2.0 ** (-h), t * 2.0 ** (-(h - 1))) for h in range(1, N)] + [ind(-np.inf, t * 2.0 ** (-(N - 1)))]
    return E, w, V

def band_of_scaled(Cm, k, t, N):
    return bands(k * Cm, t, N)[0]

def run(ell=9, eps_op=1e-3, eps_st=1e-3, N=5, t=None, verbose=False):
    rep = catalan_rep(ell)
    n = rep['n']
    OL = perturb_ops(rep, eps_op); OR = perturb_ops(rep, eps_op)
    K = state(rep, eps_st)
    I = np.eye(n)
    ops = {'L': OL, 'R': OR}
    act = {'L': L, 'R': R}
    # residuals
    res = {}
    for s in 'LR':
        o = ops[s]; f = act[s]
        A, B, C, P, Q = o['A'], o['B'], o['C'], o['P'], o['Q']
        Rl = {
            'AsA': A.conj().T @ A - I, 'rng': A @ A.conj().T + B @ B.conj().T + P - I,
            'BsB': B.conj().T @ B + P + Q - I, 'CP': C @ P - P, 'CQ': C @ Q - Q / 2,
            'CA': C @ A - A @ C / 2, 'CB': C @ B - 2 * B @ C, 'PQ': P @ Q}
        for k_, X in Rl.items(): res[(s, k_)] = nrm(f(X, K))
    AL, AR, BL, BR, CL, CR = OL['A'], OR['A'], OL['B'], OR['B'], OL['C'], OR['C']
    mir = {
        'm1': nrm(L(AL / 2, K) - R(AR.conj().T, K)), 'm2': nrm(L(AL.conj().T, K) - R(AR / 2, K)),
        'm3': nrm(L(BL, K) - R(BR.conj().T, K)), 'm4': nrm(L(BL.conj().T, K) - R(BR, K)),
        'm5': nrm(L(CL, K) - R(CR, K))}
    delta = max(list(res.values()) + list(mir.values()))
    viol = []
    def chk(name, lhs, rhs):
        if lhs > rhs * (1 + 1e-9) + 1e-13:
            viol.append((name, lhs, rhs))
    # (B1)
    chk('B1P', nrm(L(OL['P'], K) - R(OR['P'], K)), 6 * delta)
    chk('B1Q', nrm(L(OL['Q'], K) - R(OR['Q'], K)), 10 * delta)
    # (B2)
    kT = {'A': .5, 'As': 2., 'B': 2., 'Bs': .5}; cT = {'A': 1.5, 'As': 3.5, 'B': 3., 'Bs': 2.5}
    def Top(o, T):
        return {'A': o['A'], 'As': o['A'].conj().T, 'B': o['B'], 'Bs': o['B'].conj().T}[T]
    TK = {}
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'
        for T in kT:
            v = act[s](Top(ops[s], T), K); TK[(s, T)] = v
            lhs = nrm(act[s](ops[s]['C'], v) - kT[T] * act[sb](ops[sb]['C'], v))
            chk('B2' + s + T, lhs, cT[T] * delta)
    # bands
    if t is None: t = rng.uniform(2 / 3, 3 / 4)
    E = {s: bands(ops[s]['C'], t, N)[0] for s in 'LR'}
    def mism(Xs, Ys, v, Xpart, Ypart):
        # sum_h ||(Pi_h(X)-Pi_h(Y)) v||^2 for X on party Xpart, Y on Ypart
        return sum(nrm(act[Xpart](a, v) - act[Ypart](b, v)) ** 2 for a, b in zip(Xs, Ys))
    nine = [mism(E['L'], E['R'], K, 'L', 'R')]
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'
        for T in kT:
            Ys = band_of_scaled(ops[sb]['C'], kT[T], t, N)
            nine.append(mism(E[s], Ys, TK[(s, T)], s, sb))
    # matched vector
    Phi_h = [L(E['L'][h], R(E['R'][h], K)) for h in range(N + 1)]
    Phi = sum(Phi_h)
    e = nrm(K - Phi); q = nrm(Phi)
    eta = max(delta, e, math.sqrt(sum(nine)))
    chk('B5', abs((1 - q ** 2) - 0.5 * nine[0]), 1e-10)
    # (B6)
    for s in 'LR':
        o = ops[s]; f = act[s]
        chk('B6P' + s, nrm(f((I - E[s][0]) @ o['P'], K)), 4 * delta)
        chk('B6Q' + s, nrm(f((I - E[s][1]) @ o['Q'], K)), 8 * delta)
    # (B7) (B8)
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'; o = ops[s]; f = act[s]
        lhsP = math.sqrt(sum(nrm(f(o['P'], Phi_h[h])) ** 2 for h in range(1, N + 1)))
        lhsQ = math.sqrt(sum(nrm(f(o['Q'], Phi_h[h])) ** 2 for h in range(N + 1) if h != 1))
        chk('B7P' + s, lhsP, 10 * delta + e); chk('B7Q' + s, lhsQ, 18 * delta + e)
        chk('B7Peta' + s, lhsP, 11 * eta); chk('B7Qeta' + s, lhsQ, 19 * eta)
        for nm, X, nb in [('RA', o['A'].conj().T @ o['A'] - I, 1), ('RB', o['B'].conj().T @ o['B'] + o['P'] + o['Q'] - I, 2),
                          ('Rr', o['A'] @ o['A'].conj().T + o['B'] @ o['B'].conj().T + o['P'] - I, 2)]:
            agg = math.sqrt(sum(nrm(f(X, Phi_h[h])) ** 2 for h in range(N + 1)))
            chk('B8eq' + nm + s, abs(agg - nrm(f(X, Phi))), 1e-10)
            chk('B8' + nm + s, agg, delta + nb * e)
            chk('opnorm' + nm + s, np.linalg.norm(X, 2), nb)
    # (B9) leakage, (B10)
    for s in 'LR':
        o = ops[s]; f = act[s]; Es = E[s]
        A_, B_ = o['A'], o['B']
        l1 = math.sqrt(sum(nrm(f((I - Es[h + 1]) @ A_, Phi_h[h])) ** 2 for h in range(0, N - 1)))
        l2 = math.sqrt(sum(nrm(f((I - Es[h - 1]) @ B_, Phi_h[h])) ** 2 for h in range(1, N)))
        l3 = math.sqrt(sum(nrm(f((I - Es[h - 1]) @ A_.conj().T, Phi_h[h])) ** 2 for h in range(1, N)))
        l4 = math.sqrt(sum(nrm(f((I - Es[h + 1]) @ B_.conj().T, Phi_h[h])) ** 2 for h in range(0, N - 1)))
        for nm, l in zip(['A', 'B', 'As', 'Bs'], [l1, l2, l3, l4]):
            chk('B9' + nm + s, l, 2 * eta)
        chk('B10B' + s, nrm(f(B_, Phi_h[0])), 9 * delta + e)
        chk('B10As' + s, nrm(f(A_.conj().T, Phi_h[0])), 10.5 * delta + e)
    # (B11) and edge mirrors, rows v=1..N-2
    agg2 = 0.0
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'; o = ops[s]; f = act[s]; Es = E[s]
        A_, B_, P_, Q_ = o['A'], o['B'], o['P'], o['Q']
        sa = math.sqrt(sum(nrm(f(Es[v - 1] @ A_.conj().T @ Es[v] @ A_ @ Es[v - 1] - Es[v - 1], Phi_h[v - 1])) ** 2 for v in range(1, N - 1)))
        sbb = math.sqrt(sum(nrm(f(Es[v + 1] @ B_.conj().T @ Es[v] @ B_ @ Es[v + 1] - Es[v + 1], Phi_h[v + 1])) ** 2 for v in range(1, N - 1)))
        rr = math.sqrt(sum(nrm(f(Es[v] @ A_ @ Es[v - 1] @ A_.conj().T @ Es[v] + Es[v] @ B_ @ Es[v + 1] @ B_.conj().T @ Es[v] - Es[v], Phi_h[v])) ** 2 for v in range(1, N - 1)))
        chk('B11a' + s, sa, 4 * eta); chk('B11b' + s, sbb, 35 * eta); chk('B11r' + s, rr, 18 * eta)
        # direct mirrors of the lemma: m^s_A = x_A^s Phi_{v-1} - 2 (x_A^sb)^* Phi_v ; m^s_B = x_B^s Phi_{v+1} - (x_B^sb)^* Phi_v
        ob = ops[sb]; fb = act[sb]; Eb = E[sb]
        mA = math.sqrt(sum(nrm(f(Es[v] @ A_ @ Es[v - 1], Phi_h[v - 1]) - 2 * fb((Eb[v] @ ob['A'] @ Eb[v - 1]).conj().T, Phi_h[v])) ** 2 for v in range(1, N - 1)))
        mB = math.sqrt(sum(nrm(f(Es[v] @ B_ @ Es[v + 1], Phi_h[v + 1]) - fb((Eb[v] @ ob['B'] @ Eb[v + 1]).conj().T, Phi_h[v])) ** 2 for v in range(1, N - 1)))
        chk('mirA' + s, mA, 5 * eta); chk('mirB' + s, mB, 3 * eta)
        agg2 += sa ** 2 + sbb ** 2 + rr ** 2 + mA ** 2 + mB ** 2
    chk('B12agg', math.sqrt(agg2), 57 * eta)
    # (B13)-(B15)
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'; o = ops[s]; f = act[s]; Es = E[s]; ob = ops[sb]; fb = act[sb]; Eb = E[sb]
        chk('B13a' + s, nrm(f(o['P'] - I, Phi_h[0])), 32 * eta)
        chk('B13b' + s, nrm(f(o['P'] - Es[0], Phi)), 34 * eta)
        chk('B14' + s, nrm(f(o['B'].conj().T, Phi_h[0])) ** 2, 35 * eta)
        x0 = Es[0] @ o['B'] @ Es[1]
        chk('B15a' + s, nrm(f(x0, Phi_h[1])), 9 * math.sqrt(eta))
    # (B16)-(B19) tails: F_j = sum_{h>j} E_h
    Fs = {s: [sum(E[s][h] for h in range(j + 1, N + 1)) for j in range(N)] for s in 'LR'}
    a = [nrm(L(Fs['L'][j], Phi)) ** 2 for j in range(N)]
    for j in range(0, N - 1):
        FjL = Fs['L'][j]; Fjm1 = I if j == 0 else Fs['L'][j - 1]
        lhs1 = nrm(L(AL.conj().T @ FjL, Phi) - 0.5 * R(AR, L(Fjm1, Phi)))
        chk('B16a', lhs1, 3.5 * eta)
        if j + 1 <= N - 1:
            lhs2 = nrm(L(BL.conj().T @ FjL, Phi) - R(BR, L(Fs['L'][j + 1], Phi)))
            chk('B16b', lhs2, 5 * eta)
    for j in range(1, N - 1):
        chk('B17', abs(a[j] - a[j - 1] / 4 - a[j + 1]), 87 / 4 * eta + 603 * eta ** 2)
    for k in range(N):
        chk('B18', a[k], (k + 1) * 2.0 ** (-k) + 2560 * eta)
        chk('B19', nrm(L(Fs['L'][k], K)) ** 2, (k + 1) * 2.0 ** (-k) + 2561 * eta)
    # ---- row repair lemma on the strip 0..H, H=N-1, rows 1..H-1, normalized phi ----
    Hs = N - 1
    Psi = sum(Phi_h[h] for h in range(Hs + 1)); qq = nrm(Psi)
    ph = [Phi_h[h] / qq for h in range(Hs + 1)]
    # band bases
    basis = {}
    for s in 'LR':
        basis[s] = []
        for h in range(Hs + 1):
            w, V = np.linalg.eigh(E[s][h]); basis[s].append(V[:, w > .5])
    # edges per party: dict e -> (src, tgt, matrix tgt x src in band bases)
    def edgesof(s):
        o = ops[s]; Bs = basis[s]; ed = {}
        for v in range(1, Hs):
            ed[('A', v)] = (v - 1, v, Bs[v].conj().T @ o['A'] @ Bs[v - 1], 2.0)
            ed[('B', v)] = (v + 1, v, Bs[v].conj().T @ o['B'] @ Bs[v + 1], 1.0)
        return ed
    ED = {s: edgesof(s) for s in 'LR'}
    # coordinates of phi_h in band bases: tensor c[h] with shape (dL_h, dR_h, dE)
    coord = [np.einsum('ia,jb,ije->abe', basis['L'][h].conj(), basis['R'][h].conj(), ph[h]) for h in range(Hs + 1)]
    def actL(X, c): return np.einsum('ab,bje->aje', X, c)
    def actR(Y, c): return np.einsum('ab,ibe->iae', Y, c)
    actb = {'L': actL, 'R': actR}
    d2 = 0.0
    for s in 'LR':
        sb = 'R' if s == 'L' else 'L'; f = actb[s]; fb = actb[sb]
        for key, (src, tgt, x, lam) in ED[s].items():
            d2 += nrm(f(x.conj().T @ x - np.eye(x.shape[1]), coord[src])) ** 2
            xb = ED[sb][key][2]
            d2 += nrm(f(x, coord[src]) - lam * fb(xb.conj().T, coord[tgt])) ** 2
        for v in range(1, Hs):
            xa = ED[s][('A', v)][2]; xb_ = ED[s][('B', v)][2]
            r = xa @ xa.conj().T + xb_ @ xb_.conj().T - np.eye(xa.shape[0])
            d2 += nrm(f(r, coord[v])) ** 2
    d = math.sqrt(d2)
    Kc = 2 * (12 + 13 * math.sqrt(2))
    dir2 = 0.0; adj2 = 0.0; rg_err = 0.0
    for s in 'LR':
        f = actL
        cs = coord if s == 'L' else [c_.transpose(1, 0, 2) for c_ in coord]
        for v in range(1, Hs):
            xa = ED[s][('A', v)][2]; xb_ = ED[s][('B', v)][2]
            Rv = np.hstack([xa, xb_]); na_ = xa.shape[1]
            W, S, Vh = np.linalg.svd(Rv, full_matrices=False)
            keep = S > 1e-12; W = W[:, keep]; S = S[keep]; V = Vh[keep].conj().T
            # direct: zeta = j_e phi_src
            for blk, src, x in [(0, v - 1, xa), (1, v + 1, xb_)]:
                c = cs[src]
                z = np.zeros((Rv.shape[1],) + c.shape[1:], complex)
                if blk == 0: z[:na_] = c
                else: z[na_:] = c
                Vz = np.einsum('ab,bje->aje', V.conj().T, z)
                err2 = nrm(np.einsum('a,aje->aje', 1 - S, Vz)) ** 2 + (nrm(z) ** 2 - nrm(Vz) ** 2)
                dir2 += max(err2, 0)
                # cross-Gram identity check
                if blk == 0: other = xb_; own = xa
                else: other = xa; own = xb_
                lhs = nrm(f(other.conj().T @ own, c)) ** 2
                rv = Rv @ Rv.conj().T - np.eye(Rv.shape[0]); g = own.conj().T @ own
                rhs = ip(f(own, c), f(rv @ own, c)).real + ip(c, f(g @ (np.eye(g.shape[0]) - g), c)).real
                rg_err = max(rg_err, abs(lhs - rhs))
            c = cs[v]
            Wc = np.einsum('ab,bje->aje', W.conj().T, c)
            adj2 += nrm(np.einsum('a,aje->aje', 1 - S, Wc)) ** 2 + max(nrm(c) ** 2 - nrm(Wc) ** 2, 0)
    if d <= 1:
        chk('rowdirect', math.sqrt(dir2), math.sqrt(Kc * d))
        chk('rowadj', math.sqrt(adj2), d)
    chk('crossGram', rg_err, 1e-9)
    out = dict(n=n, delta=delta, e=e, eta=eta, d=d, dir=math.sqrt(dir2), adj=math.sqrt(adj2), sqrtKd=math.sqrt(Kc * d), t=t,
               viol=viol, a=a, maxratio_dir=math.sqrt(dir2) / max(math.sqrt(d), 1e-300))
    return out

if __name__ == '__main__':
    allv = []
    worst_dir = 0
    for trial in range(int(sys.argv[2]) if len(sys.argv) > 2 else 12):
        ell = [8, 9, 10][trial % 3]
        eo = [1e-4, 1e-3, 1e-2, 3e-2][trial % 4]
        es = [1e-4, 1e-3, 1e-2][trial % 3]
        N = [4, 5, 6][trial % 3]
        o = run(ell=ell, eps_op=eo, eps_st=es, N=N)
        worst_dir = max(worst_dir, o['maxratio_dir'])
        print(f"trial {trial} n={o['n']} N={N} delta={o['delta']:.2e} e={o['e']:.2e} eta={o['eta']:.2e} d={o['d']:.2e} "
              f"rowdir={o['dir']:.2e} (sqrt(Kd)={o['sqrtKd']:.2e}) rowadj={o['adj']:.2e} viol={o['viol'][:4]}", flush=True)
        allv += o['viol']
    print("TOTAL violations:", len(allv), " worst rowdir/sqrt(d):", worst_dir, " (sqrt(K*)=", math.sqrt(2*(12+13*math.sqrt(2))), ")")

if __name__ == '__main__':
    assert not allv, allv
