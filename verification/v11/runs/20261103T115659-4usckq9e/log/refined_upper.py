"""Item 1, (8): E_B(eps) <= (2/3)log2(1/eps) - (2/3)log2log2(1/eps) + O(1), with integer rounding.
For eps in [l(H), l(H-1)): E_B(eps) <= S_B(H).  Worst case of
  S_B(H) - (2/3)log2(1/eps) + (2/3)log2 log2(1/eps)  over that interval is bounded by
  S_B(H) - (2/3)log2(1/l(H-1)) + (2/3)log2 log2(1/l(H)).  Same for G (S_G = S_B+8, l_G = Lambda_G q/Q^2)."""
import mpmath as mp, json
mp.mp.dps = 40
LamB = mp.mpf(50808529) / 540000; LamG = mp.mpf(14844029) / 54000; Q = 169460
def SB(H):
    q = mp.mpf(2) ** -H
    Z = (2 * H + q) / (H + 1)
    m = (mp.mpf(2) / 3 * H * (H - 1) + q * H * (H + 5) / 6) / (2 * H + q)
    return mp.log(75, 2) - mp.mpf(8) / 5 + mp.log(Z, 2) + 2 * m
qH = lambda H: 1 / (H * mp.mpf(2) ** (H + 1) + 1)
rows = []
for H in list(range(8, 40)) + [60, 100, 200, 400, 800]:
    for name, Lam, shift, scale in (("B", LamB, 0, 1), ("G", LamG, 8, Q ** 2)):
        lH = Lam * qH(H) / scale; lH1 = Lam * qH(H - 1) / scale
        worst = SB(H) + shift - mp.mpf(2) / 3 * mp.log(1 / lH1, 2) + mp.mpf(2) / 3 * mp.log(mp.log(1 / lH, 2), 2)
        rows.append(dict(H=H, func=name, worst_const=float(worst)))
for r in rows:
    if r["H"] in (8, 10, 20, 39, 60, 100, 200, 400, 800): print(r)
print("max over H of the (8) constant: B", max(r["worst_const"] for r in rows if r["func"] == "B"),
      " G", max(r["worst_const"] for r in rows if r["func"] == "G"))
json.dump(rows, open("refined_upper.json", "w"), indent=1)
