# Checks of Sec. 8.1 repairs and of the proposed cover repair (Sec. 5.2).
# (cap)   ||(U~ - U) psi|| <= ||(U*U - I) psi||,  ||(U~* - U*) psi|| <= ||(UU* - I) psi||,   U~ = q min(|U|, I)
# (thr)   ||(1_[1/2,inf)(h) - h) psi|| <= 2 ||(h^2 - h) psi||  for self-adjoint h
# (polar) ||(w - beta) xi|| <= ||(I - beta*beta)^{1/2} xi|| for a contraction beta with unitary polar extension w
# (cover) for projections P, Q: ||P Q psi|| <= ||(P + Q - I) psi||, while the appendix's route bounds the SQUARED overlap only by
#         ||(P+Q-I) F psi||, i.e. loses a square root when the cover residual is O(sqrt r).
import numpy as np, json
rng = np.random.default_rng(3); nrm = np.linalg.norm
def rc(n, m=None): m = m or n; return rng.normal(size=(n, m)) + 1j*rng.normal(size=(n, m))
def fsa(H, f): w, V = np.linalg.eigh((H + H.conj().T)/2); return V @ np.diag(f(w)) @ V.conj().T
worst = dict(cap=0, capadj=0, thr=0, polar=0, polar_adj=0, cover=0)
for trial in range(400):
    n = 8; s = 10**rng.uniform(-4, 0.5)
    U = np.linalg.qr(rc(n))[0] @ (np.eye(n) + s*rc(n)/np.sqrt(n))
    Wl, sv, Vh = np.linalg.svd(U); q = Wl @ Vh; absU = Vh.conj().T @ np.diag(sv) @ Vh
    Ut = q @ Vh.conj().T @ np.diag(np.minimum(sv, 1)) @ Vh
    psi = rc(n, 1)[:, 0] * 10**rng.uniform(-6, 0, n)   # hostile weights
    lhs = nrm((Ut - U) @ psi); rhs = nrm((U.conj().T @ U - np.eye(n)) @ psi); worst['cap'] = max(worst['cap'], lhs/rhs)
    lhs = nrm((Ut.conj().T - U.conj().T) @ psi); rhs = nrm((U @ U.conj().T - np.eye(n)) @ psi); worst['capadj'] = max(worst['capadj'], lhs/rhs)
    h = fsa(rc(n), lambda w: np.clip(np.round(w/3) + s*w, -2, 3))
    P = fsa(h, lambda w: (w >= 0.5).astype(float))
    lhs = nrm((P - h) @ psi); rhs = 2*nrm((h @ h - h) @ psi); worst['thr'] = max(worst['thr'], lhs/max(rhs, 1e-300))
    beta = rc(n)/np.sqrt(n)*rng.uniform(0.2, 1); beta /= max(1, np.linalg.norm(beta, 2)*1.0001)
    Wb, sb, Vb = np.linalg.svd(beta); w = Wb @ Vb
    I = np.eye(n)
    lhs = nrm((w - beta) @ psi); rhs = nrm(fsa(I - beta.conj().T @ beta, lambda x: np.sqrt(np.clip(x, 0, None))) @ psi)
    worst['polar'] = max(worst['polar'], lhs/rhs)
    lhs = nrm((w.conj().T - beta.conj().T) @ psi); rhs = nrm(fsa(I - beta @ beta.conj().T, lambda x: np.sqrt(np.clip(x, 0, None))) @ psi)
    worst['polar_adj'] = max(worst['polar_adj'], lhs/rhs)
    # two projections nearly complementary
    A = np.linalg.qr(rc(n))[0]; Pp = A[:, :4] @ A[:, :4].conj().T
    B = np.linalg.qr(np.concatenate([A[:, 4:], A[:, :4]], 1) + s*rc(n)/10)[0]; Qp = B[:, :4] @ B[:, :4].conj().T
    lhs = nrm(Pp @ Qp @ psi); rhs = nrm((Pp + Qp - np.eye(n)) @ psi); worst['cover'] = max(worst['cover'], lhs/rhs)
print(json.dumps(worst))
# scalar grid checks
t = np.linspace(-5, 5, 200001)
print('thr scalar max ratio', np.max(np.abs((t >= 0.5) - t)/np.maximum(2*np.abs(t*t - t), 1e-300)))
s_ = np.linspace(0, 1, 100001); print('polar scalar', np.max(np.abs(1 - s_) - np.abs(1 - s_**2)))
s_ = np.linspace(0, 10, 100001); print('cap scalar', np.max(np.maximum(s_ - 1, 0) - np.abs(s_**2 - 1)))
print('sqrt scalar', np.max(np.abs(1 - np.sqrt(s_)) - np.abs(1 - s_)))
print('ceiling root', (33**0.5 - 1)/16, 8*((33**0.5 - 1)/16)**2 + (33**0.5 - 1)/16 - 1)
json.dump(worst, open(__file__.replace('.py', '.json'), 'w'))

assert all(np.isfinite(x) and x <= 1+1e-7 for x in worst.values()), worst
