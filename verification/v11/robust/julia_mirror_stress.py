# Stress test of chat s Sec 3: Julia-unitary repair and the calibrated mirrors.
# Claim: with eta = 2 max_xy sqrt(p_bad,xy) (so every term of (7) is <= eta), the exact transports
# w_L, w_R (Julia unitaries of the off-diagonal blocks beta_L, beta_R of X_0, Y_0) satisfy
#   ||Omega_c0c0 - sqrt2 (w_L (x) w_R) Omega_ff|| <= 7 sqrt2 eta                       (bridge vector)
#   ||(w_L - lam w_R^*) Omega|| + ||(w_L^* - lam^{-1} w_R) Omega|| <= K eta, lam = 1/sqrt2 (mirrors)
# uniformly in dimension and Schmidt spectrum. We attack with geometric spectra spanning up to
# 10^-120 and perturbations that are O(1) in operator norm on the low-weight sectors.
import json, sys
import numpy as np

rng = np.random.default_rng(7)
SQ2 = np.sqrt(2.0)

def haar(n):
    Z = (rng.standard_normal((n, n)) + 1j*rng.standard_normal((n, n)))/np.sqrt(2)
    Q, R = np.linalg.qr(Z); d = np.diag(R); return Q*(d/np.abs(d))
def rc(n, m=None):
    m = n if m is None else m
    return (rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m)))/np.sqrt(2)
def sqrtm_psd(A):
    vals, vecs = np.linalg.eigh((A + A.conj().T)/2)
    return (vecs*np.sqrt(np.clip(vals, 0, None))) @ vecs.conj().T
def contract(X):
    X = (X + X.conj().T)/2
    n = np.linalg.norm(X, 2)
    return X/max(1.0, n)

def julia(beta):
    """Halmos/Julia unitary of a contraction beta: f -> m, as an operator on f+m+m'+f' (blocks 0..3)."""
    n = beta.shape[0]
    w = np.zeros((4*n, 4*n), complex)
    w[n:2*n, 0:n] = beta
    w[n:2*n, 2*n:3*n] = sqrtm_psd(np.eye(n) - beta @ beta.conj().T)
    w[3*n:4*n, 0:n] = sqrtm_psd(np.eye(n) - beta.conj().T @ beta)
    w[3*n:4*n, 2*n:3*n] = -beta.conj().T
    return w

def analyse(F, Dm, Xs, Ys, Kfm=None, Kmf=None):
    n = F.shape[0]
    H = np.zeros((2*n, 2*n), complex); H[:n, :n] = F; H[n:, n:] = Dm/SQ2
    J = np.diag([1]*n + [-1]*n).astype(complex)
    W = {(0, 0): H, (0, 1): H, (1, 0): H, (1, 1): J @ H}
    sig = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): -1}
    four_p = []
    for (x, y), Wxy in W.items():
        X = Xs[x]; Y = Ys[y]
        t1 = np.linalg.norm(X @ Wxy - sig[(x, y)]*Wxy @ Y.T)**2
        t2 = np.linalg.norm(sqrtm_psd(np.eye(2*n) - X @ X) @ Wxy)**2
        t3 = np.linalg.norm(Wxy @ sqrtm_psd(np.eye(2*n) - Y.T @ Y.T))**2
        four_p.append(t1 + t2 + t3)          # = 4 p_bad(x,y) by identity (7)
    eta = np.sqrt(max(four_p))
    bL = Xs[0][n:, :n]; bR = Ys[0][n:, :n]
    wL = julia(bL); wR = julia(bR)
    # state coefficient on the enlarged spaces (zero on ancillas)
    K = np.zeros((4*n, 4*n), complex); K[:n, :n] = F; K[n:2*n, n:2*n] = Dm
    if Kfm is not None: K[:n, n:2*n] = Kfm
    if Kmf is not None: K[n:2*n, :n] = Kmf
    Kff = np.zeros_like(K); Kff[:n, :n] = K[:n, :n]
    # c0c0 component on the enlarged magic range m + f'
    Pm = np.zeros((4*n, 4*n)); Pm[n:2*n, n:2*n] = np.eye(n); Pm[3*n:, 3*n:] = np.eye(n)
    bridge = np.linalg.norm(Pm @ K @ Pm.T - SQ2 * wL @ Kff @ wR.T)
    lam = 1/SQ2
    m1 = np.linalg.norm(wL @ K - lam * K @ wR.conj().T.T)     # (w_L - lam w_R^*) Omega
    m2 = np.linalg.norm(wL.conj().T @ K - (1/lam) * K @ wR.T)  # (w_L^* - lam^{-1} w_R) Omega
    # exact transport check
    fh = np.zeros((4*n, 4*n)); fh[:n, :n] = np.eye(n); fh[2*n:3*n, 2*n:3*n] = np.eye(n)
    exact = max(np.abs(wL.conj().T @ wL - fh).max(), np.abs(wL @ wL.conj().T - Pm).max())
    return dict(eta=eta, bridge=bridge, m1=m1, m2=m2, exact_err=exact,
                mass=np.linalg.norm(K)**2)

def exact_model(n, decay):
    f = decay**(np.arange(n)/2.0)          # Schmidt weights f_k^2 = decay^k
    F = np.diag(f).astype(complex)
    bL = haar(n); bR = haar(n)
    Dm = SQ2 * bL @ F @ bR.T
    X = np.zeros((2*n, 2*n), complex); X[n:, :n] = bL; X[:n, n:] = bL.conj().T
    Y = np.zeros((2*n, 2*n), complex); Y[n:, :n] = bR; Y[:n, n:] = bR.conj().T
    return F, Dm, X, Y, f

def low_weight_projector(f, frac):
    """projector (on f+m, 2n) onto flag+magic directions whose weight is in the lowest frac."""
    n = len(f); k0 = int(n*(1-frac))
    P = np.zeros(2*n); P[k0:n] = 1; P[n+k0:] = 1
    return np.diag(P)

results = []
worst = dict(bridge=0, mirrors=0)
for decay in [0.5, 0.25, 1/16, 1/256, 1e-6]:
    for n in [8, 24, 48]:
        for trial in range(60):
            F, Dm, X, Y, f = exact_model(n, decay)
            kind = trial % 5
            amp = 10.0**rng.uniform(-6, -1)
            Xs = [X.copy(), X.copy()]; Ys = [Y.copy(), Y.copy()]
            Fp, Dp = F.copy(), Dm.copy(); Kfm = Kmf = None
            if kind == 0:   # small generic perturbations everywhere
                Xs = [contract(X + amp*rc(2*n)) for _ in range(2)]
                Ys = [contract(Y + amp*rc(2*n)) for _ in range(2)]
            elif kind == 1:  # O(1) operator-norm garbage on the low-weight half (Alice and Bob)
                P = low_weight_projector(f, 0.5)
                for i in range(2):
                    G = rc(2*n); G = G + G.conj().T
                    Xs[i] = contract(X + P @ G @ P + amp*rc(2*n))
                    G = rc(2*n); G = G + G.conj().T
                    Ys[i] = contract(Y + P @ G @ P + amp*rc(2*n))
            elif kind == 2:  # replace beta on low-weight rows/columns by an arbitrary contraction
                k0 = n//2
                for i in range(2):
                    Xi = X.copy(); Bp = Xi[n:, :n].copy(); Bp[k0:, :] = 0.3*rc(n - k0, n); Bp[:, k0:] = 0.3*rc(n, n - k0)
                    Xi[n:, :n] = Bp; Xi[:n, n:] = Bp.conj().T; Xs[i] = contract(Xi + amp*rc(2*n))
                    Yi = Y.copy(); Bq = Yi[n:, :n].copy(); Bq[k0:, :] = 0.3*rc(n - k0, n)
                    Yi[n:, :n] = Bq; Yi[:n, n:] = Bq.conj().T; Ys[i] = contract(Yi + amp*rc(2*n))
            elif kind == 3:  # nonzero diagonal blocks (X_0 not off-diagonal) + state perturbation
                for i in range(2):
                    Dg = np.zeros((2*n, 2*n), complex); Dg[:n, :n] = amp*rc(n); Dg[n:, n:] = amp*rc(n)
                    Xs[i] = contract(X + Dg + Dg.conj().T)
                    Ys[i] = contract(Y + amp*rc(2*n))
                Fp = F + amp*np.diag(f)*rc(n)
                Dp = Dm + amp*rc(n)*np.sqrt(decay**(np.arange(n)/2.0))[None, :]
                Kfm = amp*rc(n)*f[None, :]; Kmf = amp*rc(n)*f[:, None]
            else:            # X_0 != X_1 and Y_0 != Y_1: independent perturbations of each
                Xs = [contract(X + amp*(i + 1)*rc(2*n)) for i in range(2)]
                Ys = [contract(Y + amp*(2 - i)*rc(2*n)) for i in range(2)]
                Ys[1] = contract(Ys[1] + 0.5*low_weight_projector(f, 0.3) @ (lambda G: G + G.conj().T)(rc(2*n)) @ low_weight_projector(f, 0.3))
            r = analyse(Fp, Dp, Xs, Ys, Kfm, Kmf)
            rb = r['bridge']/r['eta']; rm = (r['m1'] + r['m2'])/r['eta']
            worst['bridge'] = max(worst['bridge'], rb); worst['mirrors'] = max(worst['mirrors'], rm)
            results.append(dict(decay=decay, n=n, kind=kind, eta=r['eta'], bridge_over_eta=rb,
                                mirrors_over_eta=rm, exact_err=r['exact_err'], min_weight=float(f[-1]**2)))

# adversarial local search: maximize (m1+m2)/eta and bridge/eta from the worst random seeds
def adv_search(decay, n, iters=400):
    F, Dm, X, Y, f = exact_model(n, decay)
    params = [rc(2*n)*1e-3 for _ in range(4)]
    def build(ps):
        Xs = [contract(X + (p + p.conj().T)) for p in ps[:2]]
        Ys = [contract(Y + (p + p.conj().T)) for p in ps[2:]]
        return Xs, Ys
    def score(ps):
        Xs, Ys = build(ps); r = analyse(F, Dm, Xs, Ys)
        return max(r['bridge'], r['m1'] + r['m2'])/r['eta'], r
    best, br = score(params)
    step = 1e-3
    for it in range(iters):
        cand = [p + step*rc(2*n)*(rng.uniform() < 0.5) for p in params]
        s, rr = score(cand)
        if s > best:
            best, params, br = s, cand, rr; step *= 1.3
        else:
            step *= 0.97
    return best, br

adv = []
for decay in [0.5, 1/16, 1e-6]:
    b, r = adv_search(decay, 10)
    adv.append(dict(decay=decay, n=10, best_ratio=b, eta=r['eta']))

summary = dict(
    trials=len(results),
    worst_bridge_over_eta=worst['bridge'], claimed_bridge_constant=7*SQ2,
    worst_mirrors_over_eta=worst['mirrors'],
    worst_by_decay={str(d): dict(bridge=max(x['bridge_over_eta'] for x in results if x['decay'] == d),
                                 mirrors=max(x['mirrors_over_eta'] for x in results if x['decay'] == d),
                                 min_weight=min(x['min_weight'] for x in results if x['decay'] == d))
                    for d in sorted(set(x['decay'] for x in results))},
    worst_by_kind={str(k): dict(bridge=max(x['bridge_over_eta'] for x in results if x['kind'] == k),
                                mirrors=max(x['mirrors_over_eta'] for x in results if x['kind'] == k))
                   for k in range(5)},
    max_exactness_error_of_julia_transport=max(x['exact_err'] for x in results),
    eta_range=[min(x['eta'] for x in results), max(x['eta'] for x in results)],
    adversarial=adv)
json.dump(dict(summary=summary, samples=results[:40]), open(sys.argv[1] if len(sys.argv) > 1 else 'julia_mirror_stress.json', 'w'), indent=1)
print(json.dumps(summary, indent=1))
