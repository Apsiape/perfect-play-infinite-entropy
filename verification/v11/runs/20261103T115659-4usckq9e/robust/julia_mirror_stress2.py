# Second pass: the small-eta / large-operator-error regime (the regime where dense-range cancellation
# fails).  X_x, Y_y equal the exact sign on the high-weight sector and an ARBITRARY self-adjoint
# contraction on the low-weight sector (direct sum => still contractions, no global rescaling).
import json, sys
import numpy as np
sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('jm', 'julia_mirror_stress.py')
src = open('julia_mirror_stress.py').read().split('results = []')[0]
ns = {}; exec(src, ns)
analyse, exact_model, rc, contract = ns['analyse'], ns['exact_model'], ns['rc'], ns['contract']
rng = np.random.default_rng(11)

def garbage_sa_contraction(k):
    G = rc(k); G = G + G.conj().T
    return G/np.linalg.norm(G, 2)

rows = []
for decay in [0.5, 0.25, 1/16, 1/256, 1e-6, 1e-12]:
    for n in [12, 32]:
        for frac in [0.25, 0.5, 0.75]:
            for trial in range(15):
                F, Dm, X, Y, f = exact_model(n, decay)
                k0 = int(n*(1 - frac))
                idx_low = np.r_[k0:n, n + k0:2*n]
                def corrupt(Z):
                    Z = Z.copy()
                    Z[np.ix_(idx_low, range(2*n))] = 0; Z[np.ix_(range(2*n), idx_low)] = 0
                    Z[np.ix_(idx_low, idx_low)] = garbage_sa_contraction(len(idx_low))
                    return (Z + Z.conj().T)/2
                Xs = [corrupt(X), corrupt(X)]; Ys = [corrupt(Y), corrupt(Y)]
                # also a tiny generic perturbation (keeps contraction by rescale ~1)
                tiny = 10.0**rng.uniform(-12, -4)
                Xs = [contract(Z + tiny*rc(2*n)) for Z in Xs]; Ys = [contract(Z + tiny*rc(2*n)) for Z in Ys]
                opnorm_err = max(np.linalg.norm(Xs[0] - X, 2), np.linalg.norm(Ys[0] - Y, 2))
                r = analyse(F, Dm, Xs, Ys)
                rows.append(dict(decay=decay, n=n, frac=frac, eta=r['eta'], opnorm_err=opnorm_err,
                                 bridge_over_eta=r['bridge']/r['eta'],
                                 mirrors_over_eta=(r['m1'] + r['m2'])/r['eta'],
                                 low_weight_mass=float(np.sum(f[k0:]**2)*3/np.sum(f**2*3))))

# constrained adversarial search: keep eta <= 1e-3, maximise ratio
def adv(decay, n, iters=600, cap=1e-3):
    F, Dm, X, Y, f = exact_model(n, decay)
    ps = [rc(2*n)*1e-5 for _ in range(4)]
    def build(ps):
        return [contract(X + p + p.conj().T) for p in ps[:2]], [contract(Y + p + p.conj().T) for p in ps[2:]]
    def sc(ps):
        Xs, Ys = build(ps); r = analyse(F, Dm, Xs, Ys)
        if r['eta'] > cap: return -1, r
        return max(r['bridge'], r['m1'] + r['m2'])/r['eta'], r
    best, br = sc(ps); step = 1e-5
    for it in range(iters):
        # weight proposals toward low-weight coordinates (where operator errors are cheap)
        wts = np.r_[1/np.maximum(f, 1e-300), 1/np.maximum(f, 1e-300)]
        wts = np.minimum(wts/wts.min(), 1e6)
        cand = [p + step*(rc(2*n)*np.sqrt(np.outer(wts, wts))/1e3) for p in ps]
        s, rr = sc(cand)
        if s > best: best, ps, br = s, cand, rr; step *= 1.2
        else: step *= 0.98
    return dict(decay=decay, n=n, best_ratio=best, eta=br['eta'])
advs = [adv(d, 10) for d in [0.5, 1/16, 1e-6]]
summ = dict(trials=len(rows),
            eta_range=[min(r['eta'] for r in rows), max(r['eta'] for r in rows)],
            opnorm_err_range=[min(r['opnorm_err'] for r in rows), max(r['opnorm_err'] for r in rows)],
            worst_bridge_over_eta=max(r['bridge_over_eta'] for r in rows),
            worst_mirrors_over_eta=max(r['mirrors_over_eta'] for r in rows),
            worst_small_eta=dict(
                bridge=max(r['bridge_over_eta'] for r in rows if r['eta'] < 1e-6) if any(r['eta'] < 1e-6 for r in rows) else None,
                mirrors=max(r['mirrors_over_eta'] for r in rows if r['eta'] < 1e-6) if any(r['eta'] < 1e-6 for r in rows) else None,
                count=int(sum(r["eta"] < 1e-6 for r in rows)),
                max_opnorm_err=max([r['opnorm_err'] for r in rows if r['eta'] < 1e-6] or [0])),
            constrained_adversarial=advs)
json.dump(dict(summary=summ, rows=rows), open('julia_mirror_stress2.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1))
