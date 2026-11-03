# Checks for chat q (rounding) items that admit compute, and chat s Sec 9 (Boolean compiler).
import json, sys
import numpy as np
rng = np.random.default_rng(20261103)
out = {}

# ---------- q (10): common random spectral partition ----------
# X = C_L (x) I, Y = I (x) C_R commute. Bands: thresholds s_h = t 2^{-h}, h = 0..H, t ~ U[2/3, 3/4].
# Claim: E_t sum_h ||(1_h(X) - 1_h(Y)) v||^2 <= K 2^H ||v|| ||(X - Y) v||.
def band_index(x, t, H):
    # band h (0..H) = [t 2^{-h}, t 2^{-h+1}) with band 0 = [t, 1]; band H+1 = [0, t 2^{-H})
    out = np.full(x.shape, H + 1)
    for h in range(H, -1, -1):
        out = np.where(x >= t*2.0**(-h), h, out)
    return out
ratios = []
for trial in range(300):
    H = rng.integers(2, 12)
    nL, nR = 7, 7
    # adversarial spectra: cluster eigenvalues near the threshold windows [2/3,3/4] 2^{-h}
    def spec(n):
        hs = rng.integers(0, H + 1, n)
        return np.clip(2.0**(-hs)*rng.uniform(0.6, 0.8, n), 0, 1)
    cL, cR = spec(nL), spec(nR)
    # commuting pair: joint eigenbasis e_i (x) e_j, eigenvalues (cL_i, cR_j)
    v = rng.standard_normal((nL, nR)) + 1j*rng.standard_normal((nL, nR))
    # concentrate v on nearly matched pairs (small (X-Y)v) to stress the ratio
    gap = np.abs(cL[:, None] - cR[None, :])
    v = v*np.exp(-gap/ (2.0**(-H)*rng.uniform(0.01, 1)))
    w2 = np.abs(v)**2
    ts = np.linspace(2/3, 3/4, 2001)
    lhs = 0.0
    for t in ts:
        bL = band_index(cL, t, H)[:, None]; bR = band_index(cR, t, H)[None, :]
        lhs += np.sum(w2*2*(bL != bR))
    lhs /= len(ts)
    rhs = 2.0**H*np.sqrt(w2.sum())*np.sqrt(np.sum(w2*gap**2))
    if rhs > 0: ratios.append(lhs/rhs)
out['threshold_lemma_commuting_max_K'] = float(max(ratios))
out['threshold_lemma_commuting_analytic_K'] = 48.0   # 2 * 12 * sum_{h<=H} 2^h / 2^H <= 48
# Same-party (noncommuting) pair: the lemma FAILS (needs the opposite-party mirror route)
X = np.diag([1.0, 0.0]); Y = np.array([[0.5, 0.5], [0.5, -0.5]]); v = np.array([1, 1])/np.sqrt(2)
def proj_ge(A, s):
    vals, vecs = np.linalg.eigh(A); m = vals >= s
    return vecs[:, m] @ vecs[:, m].conj().T
grid = np.linspace(-1, 1, 4001)
integral = np.mean([np.linalg.norm((proj_ge(X, s) - proj_ge(Y, s)) @ v)**2 for s in grid])*2
out['noncommuting_counterexample'] = dict(norm_XminusY_v=float(np.linalg.norm((X - Y) @ v)),
                                          integral_mismatch_sq=float(integral))

# ---------- q: approximate critical recurrence ----------
# a_{k+1} = a_k - a_{k-1}/4 + e_k, |e_k| <= u, a_1 <= a_0 <= 1 (tail masses)  =>  a_k <= (k+1) 2^{-k} + 4u
worst = -np.inf
for trial in range(2000):
    Hh = rng.integers(3, 40); u = 10.0**rng.uniform(-12, -1)
    a0 = 1.0 if trial % 2 == 0 else rng.uniform(0, 1)
    a = [a0, rng.uniform(0, a0)]          # tail masses: a_1 <= a_0 <= 1
    for k in range(1, Hh):
        e = u*rng.choice([-1, 1]) if trial % 3 else u   # worst case: e_k = +u
        a.append(a[k] - a[k-1]/4 + e)
    worst = max(worst, max((a[k] - (k + 1)*2.0**(-k))/u for k in range(len(a))))
out['recurrence_max_excess_over_u'] = float(worst)
G = [(n + 1)*2.0**(-n) for n in range(200)]
out['green_function_mass'] = float(sum(G))

# ---------- s Sec 9: Hadamard columns and the star Laplacian ----------
def rand_reflection(k):
    Z = rng.standard_normal((k, k)) + 1j*rng.standard_normal((k, k))
    Q, _ = np.linalg.qr(Z); s = np.diag(rng.choice([-1, 1], k))
    return Q @ s @ Q.conj().T
res = {}
for n_bits in [2, 3, 4]:
    d = 2**n_bits; k = 3
    V = [np.eye(k)] + [rand_reflection(k) for _ in range(d - 1)]   # noncommuting reflections
    chis = lambda t, j: (-1)**bin(t & j).count('1')
    Jt = {}
    for t in range(d):
        Jm = np.zeros((d*k, k), complex)
        for j in range(d):
            Jm[j*k:(j+1)*k, :] = chis(t, j)*V[j]/np.sqrt(d)
        Jt[t] = Jm
    orth = max(np.abs(Jt[t].conj().T @ Jt[u] - (t == u)*np.eye(k)).max() for t in range(d) for u in range(d))
    tot = np.abs(sum(Jt[t] @ Jt[t].conj().T for t in range(d)) - np.eye(d*k)).max()
    min_eig_gap = np.inf
    for t in range(d):
        Kt = np.zeros((d*k, d*k), complex)
        for j in range(1, d):
            s = -chis(t, j)   # forbidden graph projection G_j^{-chi_t(j)}
            G = np.zeros((d*k, d*k), complex)
            G[0:k, 0:k] = np.eye(k)/2; G[j*k:(j+1)*k, j*k:(j+1)*k] = np.eye(k)/2
            G[0:k, j*k:(j+1)*k] = s*V[j]/2; G[j*k:(j+1)*k, 0:k] = s*V[j]/2
            Kt += G
        Pt = Jt[t] @ Jt[t].conj().T
        min_eig_gap = min(min_eig_gap, np.linalg.eigvalsh(Kt - 0.5*(np.eye(d*k) - Pt)).min())
    plus = np.ones(d)/np.sqrt(d); Rd = np.kron(np.outer(plus, plus), np.eye(k))
    Ssum = sum(V)
    col = np.abs(Jt[0].conj().T @ Rd @ Jt[0] - Ssum.conj().T @ Ssum/d**2).max()
    res[str(d)] = dict(Jt_orthonormal_err=float(orth), sum_Pt_err=float(tot),
                       min_eig_Kt_minus_half_I_minus_Pt=float(min_eig_gap), column_overlap_identity_err=float(col))
out['compiler_star_laplacian'] = res

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'rounding_and_compiler_checks.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
