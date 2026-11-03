"""Item 4: (a) radial truncation of psi_Cat: S * eta^2 -> const (sharpness of S >= c eta^-2);
(b) the two 'insufficient mechanism' values."""
import math, json
a = 1.0; Z = 0.0; M = 0.0; out = []
checkpoints = {10, 100, 1000, 10000, 100000, 1000000}
for n in range(0, 1000001):
    p = a / 2           # total mass of shell n (Cat_n 4^-n / 2)
    Z += p; M += p * 2 * n
    if n in checkpoints:
        S = math.log2(1 / 1) + (sum_ := None) if False else None
        # entropy of normalized truncated state: probs 4^-n/(2Z) each, multiplicity Cat_n
        S = math.log2(2 * Z) + M / Z
        eta2 = 2 * (1 - math.sqrt(Z))
        out.append(dict(N=n, S=S, eta2=eta2, S_times_eta2=S * eta2))
        print(out[-1])
    a = a * (2 * n + 1) / (2 * (n + 2))
shift = math.log2(4 / 3) + 2 / 3
hist = sum((h + 1) * 2.0 ** (-h - 1) for h in range(200))
print("unilateral shift p_n=(3/4)4^-n entropy:", -sum((0.75 * 4.0 ** -n) * math.log2(0.75 * 4.0 ** -n) for n in range(200)), " claimed", shift)
print("canonical height histogram entropy:", hist, "(heights masses 2^-(h+1))")
json.dump(dict(radial=out, shift=shift, hist=hist), open("smoothing.json", "w"), indent=1)
