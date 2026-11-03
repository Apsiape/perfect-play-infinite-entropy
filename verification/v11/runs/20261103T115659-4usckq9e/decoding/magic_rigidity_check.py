# Numerical check of the appendix's normal magic-square lemma (Sec. 4):
# order-32 group isometry V xi = 32^{-1/2} sum_g |g> (x) f(g) xi, constants 40h, 312h, 625h (stabilizer), 632h.
# Hostile instances: context-dependent unitary perturbations concentrated on tiny Schmidt weights; mixed state via E.
import numpy as np, itertools, json, sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1., -1.])
k = np.kron
a0, b0, c0, d0 = k(Z, I2), k(I2, Z), k(I2, X), k(X, I2)
cells0 = [[a0, b0, a0 @ b0], [c0, d0, c0 @ d0], [a0 @ c0, b0 @ d0, (a0 @ c0) @ (b0 @ d0)]]
# group elements for the nine cells: (t,p,q,r,s); w = e f = ac bd = -abcd
cellg = [[(0,1,0,0,0),(0,0,1,0,0),(0,1,1,0,0)],[(0,0,0,1,0),(0,0,0,0,1),(0,0,0,1,1)],[(0,1,0,1,0),(0,0,1,0,1),(1,1,1,1,1)]]
G = list(itertools.product([0,1], repeat=5)); idx = {g:i for i,g in enumerate(G)}
def mul(g, h):
    t,p,q,r,s = g; t2,p2,q2,r2,s2 = h
    return ((t+t2+s*p2+r*q2)%2,(p+p2)%2,(q+q2)%2,(r+r2)%2,(s+s2)%2)
def inv(g):
    for h in G:
        if mul(g,h) == (0,0,0,0,0): return h
def unitary(n, scale, support=None):
    A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n)); H = (A + A.conj().T)/2
    if support is not None:
        P = np.zeros((n,n)); P[np.ix_(support,support)] = 1; H = H*P
    w, U = np.linalg.eigh(H); return U @ np.diag(np.exp(1j*scale*w)) @ U.conj().T
def run(m, dE, pert, hostile):
    n = 4*m
    # junk eta on C^m (x) C^m (x) E with hostile Schmidt spectrum
    p = np.logspace(0, -10, m); p /= p.sum()
    eta = np.zeros((m, m, dE), complex)
    for i in range(m):
        v = rng.normal(size=dE) + 1j*rng.normal(size=dE); v /= np.linalg.norm(v)
        eta[i, i, :] = np.sqrt(p[i])*v
    Phi = np.eye(4)/2.0
    # Omega[(A4,Am),(B4,Bm),E]
    Om = np.einsum('ab,ijk->aibjk', Phi, eta).reshape(n, n, dE)
    sup = [4*0 + 0]  # placeholder
    small = [x for x in range(n) if (x % m) >= m - 2] if hostile else None  # tiny-weight junk directions
    rows = []; cols = []
    for i in range(3):
        U = unitary(n, pert, small); rows.append([U @ k(cells0[i][j], np.eye(m)) @ U.conj().T for j in range(3)])
    for j in range(3):
        U = unitary(n, pert, small); cols.append([U @ k(cells0[i][j], np.eye(m)) @ U.conj().T for i in range(3)])
    M = [[rows[i][j] for j in range(3)] for i in range(3)]
    N = [[cols[j][i] for j in range(3)] for i in range(3)]
    def L(op, v): return np.einsum('xy,yzk->xzk', op, v)
    def R(op, v): return np.einsum('zy,xyk->xzk', op, v)
    nrm = np.linalg.norm
    h = max(nrm(L(M[i][j], Om) - R(N[i][j], Om)) for i in range(3) for j in range(3))
    def fmap(cellsM):
        a, b, c, d = cellsM[0][0], cellsM[0][1], cellsM[1][0], cellsM[1][1]
        f = {}
        for g in G:
            t,p_,q,r,s = g; op = np.eye(n, dtype=complex)
            for flag, x in zip((p_,q,r,s),(a,b,c,d)):
                if flag: op = op @ x
            f[g] = (-1)**t*op
        return f
    fL, fR = fmap(M), fmap(N)
    # check multiplicativity defect on Omega
    multL = max(nrm(L(fL[g] @ fL[h_], Om) - L(fL[mul(g,h_)], Om)) for g in G for h_ in G)
    multR = max(nrm(R(fR[g] @ fR[h_], Om) - R(fR[mul(g,h_)], Om)) for g in G for h_ in G)
    # isometries V_s : n -> 32 n
    def V(f):
        return np.concatenate([f[g] for g in G], axis=0)/np.sqrt(32)
    VL, VR = V(fL), V(fR)
    Omh = np.einsum('xy,yzk->xzk', VL, np.einsum('wz,yzk->ywk', VR, Om))  # shape (32n,32n,dE)
    # right regular rep R(k)|g> = |g k^{-1}>
    def Rreg(kk):
        P = np.zeros((32,32))
        ki = inv(kk)
        for g in G: P[idx[mul(g,ki)], idx[g]] = 1
        return k(P, np.eye(n))
    # stabilizers on generators
    v = Omh.copy()
    gens = [(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0),(0,0,0,0,1)]
    defects = []
    for g in gens:
        Rg = Rreg(g); defects.append(nrm(L(Rg, Omh) - R(Rg, Omh)))
        v = 0.5*(v + L(Rg, R(Rg, v)))
    q = np.vdot(Omh.ravel(), v.ravel()).real
    dist = np.sqrt(max(0.0, 1 - q))
    # intertwining of all nine cells
    ints = []
    for i in range(3):
        for j in range(3):
            Rg = Rreg(cellg[i][j])
            ints.append(nrm(np.einsum('xy,yzk->xzk', VL @ M[i][j], Om) - np.einsum('xy,yzk->xzk', Rg @ VL, Om)))
            ints.append(nrm(np.einsum('zy,xyk->xzk', VR @ N[i][j], Om) - np.einsum('zy,xyk->xzk', Rg @ VR, Om)))
    return dict(m=m, dE=dE, pert=pert, hostile=hostile, h=h,
                mult_over_h=max(multL, multR)/h, stab_defect_over_h=max(defects)/h,
                dist_over_h=dist/h, intertwine_over_h=max(ints)/h, dist=dist)
out = []
for hostile in (False, True):
    for pert in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
        r = run(5, 2, pert, hostile); out.append(r); print(json.dumps(r), flush=True)
json.dump(out, open(__file__.replace('.py', '.json'), 'w'), indent=1)

assert all(r['mult_over_h'] <= 312+1e-6 and r['dist_over_h'] <= 1250+1e-6 and r['intertwine_over_h'] <= 632+1e-6 for r in out)
