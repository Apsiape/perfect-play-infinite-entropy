"""Item 3 (inverse bound (32)): convention (b) -- matched heights 0..H, a = A on heights <= H-2,
b = B on heights 2..H.  For matched vectors, ||T_H^n Phi||^2 = sum_h (M^n 1)_h ||Phi_h||^2 (same-length
path terms are orthogonal), so ||T_H^n|| = max_h sqrt((M^n 1)_h) and ||(I-T_H)^{-1}|| <= sum_n ||T_H^n||.
Also a lower bound: ||(I-T_H)^{-1}|| >= sqrt(Z_{H,H}) (top source)."""
import numpy as np, math, json
rows = []
for H in list(range(2, 13)) + [16, 20, 24, 28, 32, 40]:
    n = H + 1
    M = np.zeros((n, n))
    for h in range(n):
        if h <= H - 2: M[h, h + 1] = 0.25      # a: up, squared amplitude 1/4
        if 2 <= h <= H: M[h, h - 1] = 1.0      # b: down
    rho = max(abs(np.linalg.eigvals(M)))
    v = np.ones(n); s = 0.0; normT1 = math.sqrt(max(M @ v))
    for k in range(200000):
        s += math.sqrt(v.max())
        v = M @ v
        if math.sqrt(v.max()) < 1e-16 * s: break
    ZHH = (2 ** (H + 1) - H - 2) / H
    rows.append(dict(H=H, rho=rho, cos_pi_over_H=math.cos(math.pi / H), normT=normT1,
                     inv_upper=s, ratio_to_H2_2H2=s / (H * H * 2 ** (H / 2)), inv_lower=math.sqrt(ZHH)))
    print(rows[-1])
json.dump(rows, open("inverse_bound.json", "w"), indent=1)
