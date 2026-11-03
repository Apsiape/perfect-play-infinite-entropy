"""Exact arithmetic in Q(zeta_24) = Q(i, sqrt2, sqrt3), written for this review.
An element is an int64 vector of 8 coefficients in the basis 1,z,...,z^7 (z = exp(2 pi i/24),
z^8 = z^4 - 1), with a separate positive integer denominator carried by the caller.
Arrays carry the field axis first: shape (8, ...).
"""
import numpy as np
from fractions import Fraction as Fr
import cmath, math

# reduction table red[k] = coefficients of z^k, k = 0..47
red = np.zeros((48, 8), dtype=np.int64)
for k in range(48):
    if k < 8:
        red[k, k] = 1
    else:
        # z^k = z^(k-8) * (z^4 - 1)
        v = np.zeros(8, dtype=np.int64)
        prev = red[k - 8]
        # multiply prev by z^4
        sh = np.zeros(16, dtype=np.int64); sh[4:12] = prev
        sh[:8] -= prev
        # reduce degrees 8..15
        for d in range(15, 7, -1):
            c = sh[d]
            if c:
                sh[d] = 0
                sh[d - 4] += c
                sh[d - 8] -= c
        red[k] = sh[:8]
Z = cmath.exp(2j * math.pi / 24)
for k in range(48):
    assert abs(sum(int(red[k, m]) * Z**m for m in range(8)) - Z**k) < 1e-12
conjmat = np.array([red[(24 - m) % 24] for m in range(8)])  # conj(z^m)
SQRT2 = red[3] + red[21]
SQRT3 = red[2] + red[22]
assert abs(sum(int(SQRT2[m]) * Z**m for m in range(8)) - math.sqrt(2)) < 1e-12
assert abs(sum(int(SQRT3[m]) * Z**m for m in range(8)) - math.sqrt(3)) < 1e-12
ONE = red[0].copy()

def rou(k):
    return red[k % 24]

def emul(a, b):
    """elementwise field product of arrays shaped (8,...)"""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.int64)
    nzA = [i for i in range(8) if a[i].any()]
    nzB = [j for j in range(8) if b[j].any()]
    prod = {}
    for i in nzA:
        for j in nzB:
            t = a[i] * b[j]
            prod[i + j] = prod.get(i + j, 0) + t
    for k, t in prod.items():
        r = red[k]
        for m in range(8):
            if r[m]:
                out[m] += r[m] * t
    return out

def fmatmul(A, B):
    """field matrix product, A (8,n,p), B (8,p,q)"""
    n, q = A.shape[1], B.shape[2]
    out = np.zeros((8, n, q), dtype=np.int64)
    nzA = [i for i in range(8) if A[i].any()]
    nzB = [j for j in range(8) if B[j].any()]
    prod = {}
    for i in nzA:
        for j in nzB:
            t = A[i] @ B[j]
            prod[i + j] = prod.get(i + j, 0) + t
    for k, t in prod.items():
        r = red[k]
        for m in range(8):
            if r[m]:
                out[m] += r[m] * t
    return out

def fconj(a):
    """complex conjugate, elementwise"""
    return np.tensordot(conjmat.T, a, axes=(1, 0))

def fadj(A):
    """conjugate transpose of (8,n,n)"""
    return np.swapaxes(fconj(A), 1, 2)

def to_fracs(v, den=1):
    return tuple(Fr(int(x), den) for x in v)

def is_rational(v):
    return not np.any(v[1:])

def to_complex(v, den=1):
    return sum(complex(int(v[m])) * Z**m for m in range(8)) / den
