"""Arithmetic of the assembly (Section 4) and exact-extension corollary, in log-space.
For every integer H>=64 we take the extreme L values producing that H (the cutoff floor), and check:
u<=2^-60, v<1/4, zeta*(N_hist+2)<1/16, final bound >= (2/9)L-(16/3)log2(L+2)-91, and the small-H range."""
import math
from mpmath import mp, mpf, log, sqrt, floor, ceil
mp.dps = 60

def lg(x): return log(x, 2)

def Hof(L): return int(floor(L / 3 - 8 * lg(L + 2) - 128))

def check_L(L):
    H = Hof(L)
    delta = mpf(2) ** (-L)
    if H < 64:
        rhs = mpf(2) / 9 * L - mpf(16) / 3 * lg(L + 2) - 200
        return ('small', H, rhs < 0)
    u = 40 * mpf(2) ** (mpf(H + 1) / 2) * sqrt(delta)
    v = (H + 1) * mpf(2) ** (-H) + 2561 * u
    Nh = 2 * H * H * int(ceil(8 * lg(2 * H)))
    zeta = 8192 * H ** 2 * mpf(2) ** (mpf(3 * H) / 4) * delta ** (mpf(1) / 4) + 2 * sqrt(H + 1) * mpf(2) ** (-mpf(H) / 2)
    # also the un-absorbed version
    zeta_raw = 2 * sqrt(v) + 4 * H * H * mpf(2) ** (mpf(H) / 2) * sqrt(140 * 128 * u)
    ok = (u <= mpf(2) ** -60) and (v < mpf(1) / 4) and (zeta * (Nh + 2) < mpf(1) / 16) and (zeta_raw <= zeta) and (Nh + 2 <= 18 * H ** 3)
    S = mpf(2) / 3 * H - mpf(14) / 3 - 4 * zeta * (Nh + 2)
    final = mpf(2) / 9 * L - mpf(16) / 3 * lg(L + 2) - 91
    return ('big', H, ok and S >= final, float(zeta * (Nh + 2)), float(S - final))

fails = []
Lmin = None
L = mpf(500)
cnt = 0
worst = 1
# scan L densely across the H>=64 onset and then geometrically
pts = [mpf(500) + k * mpf('0.25') for k in range(0, 8000)] + [mpf(2500) * mpf('1.01') ** k for k in range(0, 1200)]
for L in pts:
    r = check_L(L)
    if r[0] == 'small':
        if not r[2]: fails.append(('small', float(L), r[1]))
    else:
        if Lmin is None: Lmin = float(L)
        worst = min(worst, 1 / 16 - r[3])
        if not r[2]: fails.append(('big', float(L), r[1], r[3], r[4]))
print("first L with H>=64 ~", Lmin, " fails:", fails[:5], " #fails", len(fails), " min margin (1/16 - zeta(N+2)):", worst)
# constants
print("4*sqrt(128*140*40*sqrt2) =", 4 * math.sqrt(128 * 140 * 40 * math.sqrt(2)), "< 4096")
print("2*sqrt(2561*40*sqrt2) =", 2 * math.sqrt(2561 * 40 * math.sqrt(2)), "< 762")
Ks = 2 * (12 + 13 * math.sqrt(2)); print("K* =", Ks, " L*=2K*+18 =", 2 * Ks + 18, " 66+52sqrt2 =", 66 + 52 * math.sqrt(2))
print("lemma K formula at D=2,M=1,Lam=2:", 2 * (4 * 1 * 2 + 2 * 3 * 2 * 1 * math.sqrt(2) + math.sqrt(2) + 3 + 1))
print("6457/2 =", 6457 / 2, " 57^2 =", 57 ** 2, " lemma-orientation count 2*(16+1225+324+25+9) =", 2 * (16 + 1225 + 324 + 25 + 9))
print("(48+8*192) =", 48 + 8 * 192, " vs 1600 (=40^2); exact need 48*(1+2*(1.5+3.5+3+2.5)) =", 48 * (1 + 2 * (1.5 + 3.5 + 3 + 2.5)))
print("87/4 =", 87 / 4, " 3+21/4+1/2+10+3 =", 3 + 21 / 4 + 0.5 + 10 + 3, " 121+121+361 =", 121 + 121 + 361)
# exact-extension corollary constant, H>=64, u<=2^-60
worstc = 0
for H in range(64, 4000):
    for u in [0.0, 2.0 ** -60, 2.0 ** -200]:
        lhs = 2 * 140 * 128 * u + 14 * (16 * H * 2.0 ** -H + 10240 * u) + 108 * ((H + 1) * 2.0 ** -H + 2561 * u)
        rhs = 2 ** 21 * (u + H * 2.0 ** -H)
        if rhs > 0: worstc = max(worstc, lhs / rhs)
print("exact-extension: max lhs/rhs =", worstc, "(<1 required)")
# cumulative tail for a_{H-1}+a_H : (H-1)2^{-(H-2)} -> 4*(...)
H = 64
print("4*(H-1)2^{-(H-2)} vs 16 H 2^{-H}:", 4 * (H - 1) * 2.0 ** -(H - 2), 16 * H * 2.0 ** -H)
# smallest L at which the theorem's bound is positive
for Lv in range(500, 5000):
    if 2 / 9 * Lv - 16 / 3 * math.log2(Lv + 2) - 200 > 0:
        print("theorem bound first positive at L =", Lv, " (delta = 2^-L)"); break

assert not fails, fails
assert worstc < 1, worstc
