# Third pass (fixes pass 2): base point with beta_L = beta_R = I (WLOG by local unitaries on J; all
# quantities are unitarily invariant), so the low-weight flag AND magic directions are aligned and
# corruption on them is genuinely cheap in eta while O(1) in operator norm.
import json, sys
import numpy as np
src = open('julia_mirror_stress.py').read().split('results = []')[0]
ns = {}; exec(src, ns)
analyse, rc, contract = ns['analyse'], ns['rc'], ns['contract']
SQ2 = np.sqrt(2)
rng = np.random.default_rng(13)
def model(n, decay):
    f = decay**(np.arange(n)/2.0); F = np.diag(f).astype(complex); Dm = SQ2*F
    X = np.zeros((2*n, 2*n), complex); X[n:, :n] = np.eye(n); X[:n, n:] = np.eye(n)
    return F, Dm, X, X.copy(), f
def garbage(k):
    G = rc(k); G = G + G.conj().T; return G/np.linalg.norm(G, 2)
rows = []
for decay in [0.5, 0.25, 1/16, 1/256, 1e-6, 1e-12]:
    for n in [12, 32]:
        for frac in [0.25, 0.5, 0.75]:
            for trial in range(12):
                F, Dm, X, Y, f = model(n, decay)
                k0 = int(n*(1 - frac)); low = np.r_[k0:n, n + k0:2*n]
                def corrupt(Z):
                    Z = Z.copy(); Z[low, :] = 0; Z[:, low] = 0
                    Z[np.ix_(low, low)] = garbage(len(low)); return (Z + Z.conj().T)/2
                tiny = 10.0**rng.uniform(-12, -5)
                Xs = [contract(corrupt(X) + tiny*rc(2*n)) for _ in range(2)]
                Ys = [contract(corrupt(Y) + tiny*rc(2*n)) for _ in range(2)]
                # state perturbation too (unnormalised components, weights preserved in scale)
                Fp = F + tiny*rc(n)*f[None, :]; Dp = Dm + tiny*rc(n)*f[None, :]
                r = analyse(Fp, Dp, Xs, Ys)
                rows.append(dict(decay=decay, n=n, frac=frac, eta=r['eta'],
                                 opnorm_err=float(max(np.linalg.norm(Xs[0] - X, 2), np.linalg.norm(Ys[0] - Y, 2))),
                                 sqrt_low_mass=float(np.sqrt(np.sum(f[k0:]**2))),
                                 bridge_over_eta=r['bridge']/r['eta'], mirrors_over_eta=(r['m1'] + r['m2'])/r['eta']))
def adv(decay, n, iters=800, cap=1e-4):
    F, Dm, X, Y, f = model(n, decay)
    wts = np.r_[1/f, 1/f]; wts = np.minimum(wts, 1e8)
    scale = np.outer(1/np.sqrt(wts), 1/np.sqrt(wts))  # proposals ~ cheap where weight small
    ps = [np.zeros((2*n, 2*n), complex) for _ in range(4)]
    def sc(ps):
        Xs = [contract(X + p + p.conj().T) for p in ps[:2]]; Ys = [contract(Y + p + p.conj().T) for p in ps[2:]]
        r = analyse(F, Dm, Xs, Ys)
        if r['eta'] > cap or r['eta'] == 0: return -1, r
        return max(r['bridge'], r['m1'] + r['m2'])/r['eta'], r
    best, br = -1, None; step = 1e-6
    for it in range(iters):
        cand = [p + step*rc(2*n)*(1/np.sqrt(np.outer(np.r_[f, f], np.r_[f, f])).clip(max=1e4))*1e-2 for p in ps]
        s, rr = sc(cand)
        if s > best: best, ps, br = s, cand, rr; step *= 1.2
        else: step *= 0.98
    return dict(decay=decay, n=n, best_ratio=best, eta=None if br is None else br['eta'])
advs = [adv(d, 10) for d in [0.5, 1/16, 1e-6]]
small = [r for r in rows if r['eta'] < 1e-3]
summ = dict(trials=len(rows), eta_range=[min(r['eta'] for r in rows), max(r['eta'] for r in rows)],
            opnorm_err_range=[min(r['opnorm_err'] for r in rows), max(r['opnorm_err'] for r in rows)],
            worst_bridge_over_eta=max(r['bridge_over_eta'] for r in rows),
            worst_mirrors_over_eta=max(r['mirrors_over_eta'] for r in rows),
            small_eta_count=len(small),
            small_eta_worst=dict(bridge=max(r['bridge_over_eta'] for r in small), mirrors=max(r['mirrors_over_eta'] for r in small),
                                 min_opnorm_err=min(r['opnorm_err'] for r in small)),
            eta_over_sqrt_low_mass_range=[min(r['eta']/r['sqrt_low_mass'] for r in rows), max(r['eta']/r['sqrt_low_mass'] for r in rows)],
            constrained_adversarial=advs)
json.dump(dict(summary=summ, rows=rows), open('julia_mirror_stress3.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1))
