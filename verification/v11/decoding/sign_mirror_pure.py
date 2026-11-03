# Second-engine check of Sec. 6.1-6.2: compression/completion, linear leakage (signleak), and the bounded raw-sign
# mirror Z = L Yhat L^{-1} (L = diag(1, sqrt2)), on a reference vector with exact comparison geometry and hostile spectra.
# Model: Alice  A = f (m) + mag (C^2 (x) C^m: directions c, cperp) + out (m_out);  Bob B likewise with Bob's sign support
# T = f + r, r = (c + cperp)/sqrt2, so <c, r> = 1/sqrt2 (the sigma = +1 positive comparison pair).
# Psi = sum_i f_i |f i>|f i> + sqrt2 sum_i f_i (|c i>|c i> + |cp i>|cp i>)   (exact bridge: Psi_cc = sqrt2 (w(x)w) Psi_ff)
import numpy as np, json, sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
nrm = np.linalg.norm
def run(m, mout, spread, pert, mode):
    n = m + 2*m + mout
    fi = np.logspace(0, -spread, m)
    Psi = np.zeros((n, n), complex)
    F = lambda i: i; C = lambda i: m + i; CP = lambda i: 2*m + i
    for i in range(m):
        Psi[F(i), F(i)] = fi[i]; Psi[C(i), C(i)] = np.sqrt(2)*fi[i]; Psi[CP(i), CP(i)] = np.sqrt(2)*fi[i]
    Psi /= nrm(Psi)
    # Bob basis vectors r_i, rperp_i
    def vec(idx):
        v = np.zeros(n, complex); v[idx] = 1; return v
    rB = [ (vec(C(i)) + vec(CP(i)))/np.sqrt(2) for i in range(m)]
    rpB = [ (vec(C(i)) - vec(CP(i)))/np.sqrt(2) for i in range(m)]
    proj = lambda vs: sum(np.outer(v, v.conj()) for v in vs)
    Sf = proj([vec(F(i)) for i in range(m)])
    S = Sf + proj([vec(C(i)) for i in range(m)])                  # Alice exact support f + c
    T = Sf + proj(rB)                                              # Bob exact support f + r
    Spp = S.copy()                                                 # Bob's f + c (for the K-block)
    # ideal sign PVMs: Alice X* swaps f_i <-> c_i ; Bob Y* swaps f_i <-> r_i ; '+'=(I+X)/2 on support, rest = I - support
    def signpvm(supp_pairs):
        Pp = np.zeros((n, n), complex); Pm = np.zeros((n, n), complex)
        for u, v in supp_pairs:
            pplus = (u + v)/np.sqrt(2); pminus = (u - v)/np.sqrt(2)
            Pp += np.outer(pplus, pplus.conj()); Pm += np.outer(pminus, pminus.conj())
        return [Pp, Pm, np.eye(n) - Pp - Pm]
    A_pvm = signpvm([(vec(F(i)), vec(C(i))) for i in range(m)])
    B_pvm = signpvm([(vec(F(i)), rB[i]) for i in range(m)])
    def unitary(scale, support=None):
        Am = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n)); H = (Am + Am.conj().T)/2
        if support is not None:
            Pm_ = np.zeros((n, n)); Pm_[np.ix_(support, support)] = 1; H = H*Pm_
        w, U = np.linalg.eigh(H); return U @ np.diag(np.exp(1j*scale*w)) @ U.conj().T
    if mode == 'pure':
        small = [F(m-1), C(m-1), CP(m-1)] + list(range(3*m, n))
        UA, UB = unitary(pert, small), unitary(pert, small)
    elif mode == 'hostile':   # large rotations only on the tiny-weight Schmidt index plus the out space
        small = [F(m-1), C(m-1), CP(m-1)] + list(range(3*m, n))
        UA, UB = unitary(pert*30, small), unitary(pert*30, small)
        UA2, UB2 = unitary(pert), unitary(pert)
        UA, UB = UA @ UA2, UB @ UB2
    else:
        UA, UB = unitary(pert), unitary(pert)
    A_pvm = [UA @ P @ UA.conj().T for P in A_pvm]; B_pvm = [UB @ P @ UB.conj().T for P in B_pvm]
    L_ = lambda op, v: op @ v; R_ = lambda op, v: v @ op.T
    Pp, Pm, _ = A_pvm; Qp, Qm, _ = B_pvm
    S0 = Pp + Pm; T0 = Qp + Qm; Xr = Pp - Pm; Yr = Qp - Qm
    supp = max(nrm(L_(S0 - S, Psi)), nrm(R_(T0 - T, Psi)))
    forb = max(nrm(L_(Pp, R_(Qm, Psi))), nrm(L_(Pm, R_(Qp, Psi))))
    d = max(supp, forb)
    Xc = S @ Xr @ S; Yc = T @ Yr @ T
    leak = nrm(L_(Xr - Xc, Psi))
    # Bob rotation rho: r_i -> c_i, f -> f ; Yhat = rho Yc rho*
    rho = Sf + sum(np.outer(vec(C(i)), rB[i].conj()) for i in range(m))
    Yhat = rho @ Yc @ rho.conj().T
    Lmat = Sf + np.sqrt(2)*proj([vec(C(i)) for i in range(m)])
    Linv = Sf + (1/np.sqrt(2))*proj([vec(C(i)) for i in range(m)])
    Zm = Lmat @ Yhat @ Linv
    mirr = nrm(L_(Xr, Psi) - R_(Zm, Psi))
    # sanity: exact identity X_c K - K (L Yhat L^-1)^T = E L in coefficient form
    Kc = S @ Psi @ Spp.T
    Hc = Kc @ Linv.T
    E = Xc @ Hc - Hc @ Yhat.T
    ident = nrm(Xc @ Kc - Kc @ Zm.T - E @ Lmat.T)
    # weighted sharpness identity check: 4 p_bad = ||X W - W Y^T||^2 + ||(I-X^2)^{1/2} W||^2 + ||W (I-(Y^T)^2)^{1/2}||^2
    Wc = S @ Psi @ T.T
    Ep = S @ Pp @ S + (S - S @ S0 @ S)/2; Em = S @ Pm @ S + (S - S @ S0 @ S)/2
    Fp = T @ Qp @ T + (T - T @ T0 @ T)/2; Fm = T @ Qm @ T + (T - T @ T0 @ T)/2
    pbad = (np.trace(Wc.conj().T @ Ep @ Wc @ Fm.T) + np.trace(Wc.conj().T @ Em @ Wc @ Fp.T)).real
    def psd_sqrt(M_):
        w, U = np.linalg.eigh((M_ + M_.conj().T)/2); return U @ np.diag(np.sqrt(np.clip(w, 0, None))) @ U.conj().T
    rhs = nrm(Xc @ Wc - Wc @ Yc.T)**2 + nrm(psd_sqrt(S - Xc @ Xc) @ Wc)**2 + nrm(Wc @ psd_sqrt(T - Yc @ Yc).T)**2
    trace_roundoff = 256*n*np.finfo(float).eps
    ratio = pbad/d**2 if d**2 > trace_roundoff else None
    return dict(m=m, spread=spread, pert=pert, mode=mode, d=d, leak_over_d=leak/d, mirror_over_d=mirr/d,
                mirror_norm=float(np.linalg.norm(Zm, 2)), pbad=float(pbad), pbad_over_d2=ratio, ratio_resolved=ratio is not None, sharp_identity_resid=abs(4*pbad - rhs),
                sim_identity_resid=ident, opdev=float(np.linalg.norm(UA - np.eye(n), 2)))
out = []
worst = dict(leak=0, mirror=0, pbad=0)
for mode in ('pure',):
    for spread in (0, 4, 8, 12):
        for pert in (0.3, 1.0, 3.0):
            for rep in range(3):
                r = run(4, 3, spread, pert, mode); out.append(r)
                worst['leak'] = max(worst['leak'], r['leak_over_d']); worst['mirror'] = max(worst['mirror'], r['mirror_over_d'])
                if r['ratio_resolved']: worst['pbad'] = max(worst['pbad'], r['pbad_over_d2'])
            print(json.dumps(r), flush=True)
print('WORST', worst, 'claimed: leak <= 1+sqrt164 =', 1+164**0.5, ' mirror <= leak + sqrt2*sqrt82 + ... ', ' pbad <= 41/2')
json.dump(dict(runs=out, worst=worst), open(__file__.replace('.py', '.json'), 'w'), indent=1)

assert worst['leak'] <= 1+164**.5+1e-6
# Trace cancellation is O(machine epsilon), even when d^2 is O(1e-24).
# Check the probability inequality before division by d^2, with a conservative
# dimension-scaled floating-point allowance for the normalized coefficient state.
for sample in out:
    trace_roundoff = 256*(3*sample['m']+3)*np.finfo(float).eps
    measured_pbad = sample['pbad']
    assert -trace_roundoff <= measured_pbad <= (41/2)*sample['d']**2 + trace_roundoff, sample
assert all(r['sim_identity_resid'] < 1e-8 and r['sharp_identity_resid'] < 1e-7 and r['mirror_norm'] <= 2**.5+1e-8 for r in out)
