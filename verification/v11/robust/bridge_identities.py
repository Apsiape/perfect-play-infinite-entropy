# Refutation-first check of chat s, Sec 3: identities (9), Bob-side analogue, (7), (12), (24)
# Engines: (1) sympy noncommutative algebra; (2) numpy random matrices, using the paper's own
# definitions (actual magic-square eigenvectors c_x, r_y; W_xy = diag(F, sigma_xy D / sqrt2)).
import json, sys
import numpy as np
import sympy as sp

out = {}
rng = np.random.default_rng(20261103)

# ---------- Engine 1: sympy noncommutative ----------
X0, X1, Y0T, Y1T, H, JL, JR = sp.symbols('X0 X1 Y0T Y1T H JL JR', commutative=False)
E00 = X0*H - H*Y0T
E01 = X0*H - H*Y1T
E10 = X1*H - H*Y0T
W11 = JL*H                    # = H*JR
E11 = X1*W11 + W11*Y1T        # sigma_11 = -1
lhs = (X0*JL + JL*X0)*H
rhs = E11 + (E00 - E10)*JR + JL*E01
diff = sp.expand(rhs - lhs)
diff = sp.expand(diff.subs(H*JR, JL*H))
out['sym_identity_9_residual'] = str(diff)
# Bob-side analogue: H {Y0^T, J_R} = E11 - E10 J_R + J_L (E01 - E00)
lhsB = H*(Y0T*JR + JR*Y0T)
rhsB = E11 - E10*JR + JL*(E01 - E00)
diffB = sp.expand(sp.expand(rhsB - lhsB).subs(H*JR, JL*H))
diffB = sp.expand(diffB.subs(H*JR, JL*H))
out['sym_identity_9_bob_residual'] = str(diffB)

# (24): with w = m w f, f m = 0:  m [w + w*, Z] f = w Z_f - Z_m w
# check numerically on random matrices (sympy would need projection rules)
def rand_c(n, m=None):
    m = n if m is None else m
    return rng.standard_normal((n, m)) + 1j*rng.standard_normal((n, m))
n = 6
fP = np.diag([1]*3 + [0]*3).astype(complex); mP = np.eye(n) - fP
w = mP @ rand_c(n) @ fP
Z = rand_c(n); Z = Z + Z.conj().T
Xh = w + w.conj().T
lhs24 = mP @ (Xh @ Z - Z @ Xh) @ fP
rhs24 = w @ (fP @ Z @ fP) - (mP @ Z @ mP) @ w
out['identity_24_residual'] = float(np.abs(lhs24 - rhs24).max())

# ---------- Engine 2: numpy with the paper's definitions ----------
# magic square canonical realization; c_x column-zero outcomes (+++),(+--) of (XI, IZ, XZ)
# r_y row-zero outcomes (+++),(+--) of (XI, IX, XX)
I2 = np.eye(2); Xp = np.array([[0, 1], [1, 0]]); Zp = np.diag([1, -1])
def joint_eig(ops, signs):
    P = np.eye(4)
    for O, s in zip(ops, signs):
        P = P @ (np.eye(4) + s*O)/2
    vals, vecs = np.linalg.eigh(P)
    return vecs[:, np.argmax(vals)]
XI = np.kron(Xp, I2); IZ = np.kron(I2, Zp); IX = np.kron(I2, Xp)
c = [joint_eig([XI, IZ], [1, 1]), joint_eig([XI, IZ], [1, -1])]
r = [joint_eig([XI, IX], [1, 1]), joint_eig([XI, IX], [1, -1])]
# fix phases: make the first nonzero entry positive
def fix(v):
    k = np.argmax(np.abs(v) > 1e-9); return v*np.sign(v[k])
c = [fix(v) for v in c]; r = [fix(v) for v in r]
ov = np.array([[c[x] @ r[y] for y in range(2)] for x in range(2)])
out['overlaps_c_x_r_y_times_sqrt2'] = (ov*np.sqrt(2)).round(12).tolist()
sig = np.array([[1, 1], [1, -1]])
out['sigma_matches_overlaps'] = bool(np.allclose(ov*np.sqrt(2), sig))
# NB: the relative phase of r_1 versus c_x is a convention; the predicate fixes sigma_xy

N = 5
F = rand_c(N); Dm = rand_c(N)   # arbitrary coefficient blocks (state need not be exact)
# Build the actual compressed coefficient W_xy = P_{S_x^A} K P_{S_y^B}^T in identified coordinates.
# Magic coefficient on C^4 (x) J : I_4 (x) Dm  (Alice coords c, Bob coords r). Compress:
def Wxy(x, y):
    W = np.zeros((2*N, 2*N), complex)
    W[:N, :N] = F
    W[N:, N:] = (c[x] @ r[y]) * Dm
    return W
Hm = np.zeros((2*N, 2*N), complex); Hm[:N, :N] = F; Hm[N:, N:] = Dm/np.sqrt(2)
J = np.diag([1]*N + [-1]*N).astype(complex)
out['W00=W01=W10=H'] = bool(np.allclose(Wxy(0, 0), Hm) and np.allclose(Wxy(0, 1), Hm) and np.allclose(Wxy(1, 0), Hm))
out['W11=J_L H=H J_R'] = bool(np.allclose(Wxy(1, 1), J @ Hm) and np.allclose(J @ Hm, Hm @ J))

def rand_sa_contraction(k, scale=1.0):
    A = rand_c(k); A = (A + A.conj().T)/2
    A = A*scale
    nrm = np.linalg.norm(A, 2)
    return A/max(1.0, nrm*1.0000001)
Xs = [rand_sa_contraction(2*N) for _ in range(2)]
Ys = [rand_sa_contraction(2*N) for _ in range(2)]
E = {}
for x in range(2):
    for y in range(2):
        E[(x, y)] = Xs[x] @ Wxy(x, y) - sig[x, y] * Wxy(x, y) @ Ys[y].T
lhs9 = (Xs[0] @ J + J @ Xs[0]) @ Hm
rhs9 = E[(1, 1)] + (E[(0, 0)] - E[(1, 0)]) @ J + J @ E[(0, 1)]
out['numeric_identity_9_residual'] = float(np.abs(lhs9 - rhs9).max())
lhs9b = Hm @ (Ys[0].T @ J + J @ Ys[0].T)
rhs9b = E[(1, 1)] - E[(1, 0)] @ J + J @ (E[(0, 1)] - E[(0, 0)])
out['numeric_identity_9_bob_residual'] = float(np.abs(lhs9b - rhs9b).max())

# ---- (7): defect identity, via actual vectors and POVM probabilities ----
def sqrtm_psd(A):
    vals, vecs = np.linalg.eigh((A + A.conj().T)/2)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.conj().T
def vec_of(W):  # Omega = sum W_ij |i>|j>, so (X(x)Y)Omega <-> X W Y^T
    return W.reshape(-1)
worst7 = 0.0
for trial in range(200):
    k = 4
    X = rand_sa_contraction(k); Y = rand_sa_contraction(k)
    W = rand_c(k)*rng.uniform(0.01, 3)
    s = rng.choice([1, -1])
    Ea = {+1: (np.eye(k) + X)/2, -1: (np.eye(k) - X)/2}
    Bb = {+1: (np.eye(k) + Y)/2, -1: (np.eye(k) - Y)/2}
    Om = vec_of(W)
    pbad = 0.0
    for a in (1, -1):
        for b in (1, -1):
            if a*b != s:
                v = np.kron(sqrtm_psd(Ea[a]), sqrtm_psd(Bb[b])) @ Om
                pbad += np.vdot(v, v).real
    rhs7 = (np.linalg.norm(X @ W - s * W @ Y.T)**2
            + np.linalg.norm(sqrtm_psd(np.eye(k) - X @ X) @ W)**2
            + np.linalg.norm(W @ sqrtm_psd(np.eye(k) - Y.T @ Y.T))**2)
    worst7 = max(worst7, abs(4*pbad - rhs7)/max(1e-300, rhs7))
out['identity_7_max_relative_residual'] = worst7

# ---- (12): weighted sharpness, exact identity behind the 5 eta^2 bound ----
worst12 = 0.0; worst12_bound = 0.0
for trial in range(200):
    X = rand_sa_contraction(2*N)
    a = X[:N, :N]; beta = X[N:, :N]; cc = X[N:, N:]
    lhs12 = (np.linalg.norm(sqrtm_psd(np.eye(N) - beta.conj().T @ beta) @ F)**2
             + 0.5*np.linalg.norm(sqrtm_psd(np.eye(N) - beta @ beta.conj().T) @ Dm)**2)
    diagXH = np.zeros_like(Hm); diagXH[:N, :N] = a @ F; diagXH[N:, N:] = cc @ Dm/np.sqrt(2)
    rhs12 = np.linalg.norm(sqrtm_psd(np.eye(2*N) - X @ X) @ Hm)**2 + np.linalg.norm(diagXH)**2
    worst12 = max(worst12, abs(lhs12 - rhs12)/rhs12)
    # diag bound from (9): ||diag(X0) H|| = ||{X0,J}H||/2
    worst12_bound = max(worst12_bound, abs(np.linalg.norm(diagXH) - np.linalg.norm((X @ J + J @ X) @ Hm)/2))
out['identity_12_max_relative_residual'] = worst12
out['diag_equals_half_anticommutator_residual'] = worst12_bound

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else 'bridge_identities.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
