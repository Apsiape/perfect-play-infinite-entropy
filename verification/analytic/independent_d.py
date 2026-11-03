"""Independent exact scalar audit of Theorem D.

Standard library only. Run: python -B independent_d.py
Writes independent_d.json beside this file. No supplied producer or checker is imported.
This same-family audit does not constitute a different-model-family review.
"""
from fractions import Fraction
from math import comb, pi, sqrt
from pathlib import Path
import json


def envelope(x, scale):
    return Fraction(1) if x <= 0 else max(Fraction(0), 1 - (x / scale) ** 2)


def catalan_mass(n):
    return Fraction(comb(2 * n, n), (n + 1) * 4 ** n)


def exact_profile(scale):
    # The shifted support extends only through n=scale+1.
    z = Fraction(0)
    first = Fraction(0)
    second = Fraction(0)
    for n in range(scale + 2):
        a = catalan_mass(n)
        f0 = envelope(Fraction(n), scale)
        f1 = envelope(Fraction(n) - Fraction(1, 2), scale)
        f2 = envelope(Fraction(n) - 1, scale)
        z += a * (f0 * f0 + 8 * f1 * f1 + 16 * f2 * f2) / 50
        first += a * (f0 - f1) ** 2
        second += a * (f1 - f2) ** 2
    loss = Fraction(12, 25) * (first + 2 * second) / z
    return {"N": scale, "normalization": str(z), "D1": str(first),
            "D2": str(second), "bell_loss": str(loss)}


def main():
    expected = {
        1: ("241/400", "9/64", "1/64", "33/241"),
        2: ("8247/12800", "67/2048", "27/2048", "121/2749"),
        3: ("349541/518400", "1141/82944", "23/3072", "7149/349541"),
        5: ("22906777/32000000", "23009/5120000", "15769/5120000", "163641/22906777"),
    }
    rows = []
    for n, wanted in expected.items():
        row = exact_profile(n)
        actual = tuple(row[k] for k in ("normalization", "D1", "D2", "bell_loss"))
        assert actual == wanted, (n, actual, wanted)
        rows.append(row)
    # Independent integration of 4*x^(1/2) and x^(-1/2)*(1-x^2)^2.
    dirichlet_integral = 4 * Fraction(2, 3)
    entropy_integral = 2 - Fraction(4, 5) + Fraction(2, 9)
    assert dirichlet_integral == Fraction(8, 3)
    assert entropy_integral == Fraction(64, 45)
    # Each bridge has 48 coarse contexts and two forbidden sign pairs.
    per_bridge_events = 48 * 2
    assert Fraction(per_bridge_events, 200) == Fraction(12, 25)
    cl = float(Fraction(12, 25) * 3 * dirichlet_integral / 4) / sqrt(pi)
    cs = float(entropy_integral) / sqrt(pi)
    c = cs * cl ** Fraction(1, 5)
    game = c * 169460 ** Fraction(-2, 5)
    assert abs(c - 0.7097931163542254) < 1e-14
    assert abs(game - 0.005747843070476386) < 1e-15
    result = {
        "audit": "independently written same-family Theorem D scalar check",
        "arithmetic": "exact rational finite sums; floating point pi-based constants",
        "exact_profiles": rows,
        "integrals": {"dirichlet": str(dirichlet_integral), "entropy": str(entropy_integral)},
        "coarse_forbidden_events_per_bridge": per_bridge_events,
        "constants": {"C_L": cl, "C_S": cs, "bell_constant": c, "game_constant": game},
        "assertions_passed": True,
        "limitations": "Does not replace analytic asymptotics, prove an unrestricted entropy lower power, or constitute cross-family review.",
    }
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
