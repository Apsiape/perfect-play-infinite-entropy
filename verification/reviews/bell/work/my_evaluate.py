# Independent exact evaluation of the literal small table (out5 functional-small.zip)
# on a canonical truncated Catalan strategy that I build myself from the definitions
# (out3 Section 1/5.1, out4 Sections 3-7, out5 Sections 2 and 4).
#
# Written from scratch for this review; imports nothing from out3/out4/out5 and runs no
# earlier checker.  Arithmetic is exact in the cyclotomic field Q(zeta_24) (which contains
# i, sqrt2, sqrt3), with elements stored as integer coordinate vectors on the power basis
# 1, z, ..., z^7 (z = exp(i pi/12), z^8 = z^4 - 1) and a per-operator integer denominator.
# Every distinct effect is checked to be a Hermitian projection, every FMS/refined question
# to be a complete measurement, every one of the 4,529 projection rows is evaluated as an
# operator, and then every literal coefficient record is contracted.
#
# usage: python -I my_evaluate.py <extracted small archive dir> N [N ...]
import json, sys, struct, hashlib, time, itertools
from fractions import Fraction as Fr
from math import lcm
import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------- field Q(zeta_24)
DEG = 8
RED = np.zeros((48, DEG), dtype=np.int64)     # RED[k] = coordinates of z^k
cur = np.zeros(DEG, dtype=np.int64); cur[0] = 1
for k in range(48):
    RED[k] = cur
    nxt = np.zeros(DEG, dtype=np.int64); nxt[1:] = cur[:-1]
    top = cur[-1]                               # z^8 = z^4 - 1
    nxt[4] += top; nxt[0] -= top
    cur = nxt
assert (RED[24] == np.eye(DEG, dtype=np.int64)[0]).all()      # z^24 = 1
assert (RED[12] == -np.eye(DEG, dtype=np.int64)[0]).all()     # z^12 = -1
MULT = np.zeros((DEG, DEG, DEG), dtype=np.int64)
for u in range(DEG):
    for v in range(DEG):
        MULT[u, v] = RED[u + v]
CONJ = np.zeros((DEG, DEG), dtype=np.int64)    # coordinates of conj(z^u) = z^(24-u)
for u in range(DEG):
    CONJ[u] = RED[(24 - u) % 24]
def zpow(k): return RED[k % 24].copy()
ONE = zpow(0)
SQRT2 = zpow(3) + zpow(21)                     # z^3 + z^-3 = 2 cos(pi/4)
def fmul(a, b):                                 # vectorised field product, shapes (...,8)
    return np.einsum('...u,...v,uvk->...k', a, b, MULT)
def fconj(a): return a @ CONJ
assert (fmul(SQRT2, SQRT2) == 2 * ONE).all()
assert (fmul(zpow(6), zpow(6)) == -ONE).all()          # i^2 = -1
s3 = zpow(2) + zpow(22)                                # 2cos(pi/6) = sqrt3
assert (fmul(s3, s3) == 3 * ONE).all()

# ---------------------------------------------------------------- sparse field matrices
class Op:
    """sparse d x d matrix over Q(zeta24): entries coords[n]/den at (rows[n], cols[n])."""
    __slots__ = ('r', 'c', 'x', 'den')
    def __init__(self, r, c, x, den=1):
        self.r, self.c, self.x, self.den = r, c, x, den
def coalesce(r, c, x, den, d):
    key = r.astype(np.int64) * d + c
    order = np.argsort(key, kind='stable')
    key, x = key[order], x[order]
    if len(key) == 0:
        return Op(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, DEG), np.int64), den)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    xs = np.add.reduceat(x, starts, axis=0)
    ks = key[starts]
    nz = np.any(xs != 0, axis=1)
    ks, xs = ks[nz], xs[nz]
    # reduce the common denominator
    gg = int(np.gcd.reduce(np.abs(xs).ravel())) if xs.size else 0
    g2 = np.gcd(gg, den) if gg else den
    if g2 > 1:
        xs = xs // g2; den //= g2
    return Op(ks // d, ks % d, xs, int(den))
def lincomb(terms, d):
    """terms: list of (Fraction coefficient, Op)."""
    L = 1
    for q, o in terms:
        q = Fr(q); L = lcm(L, q.denominator * o.den)
    rs, cs, xs = [], [], []
    for q, o in terms:
        q = Fr(q)
        if q == 0 or len(o.r) == 0: continue
        f = L // (q.denominator * o.den) * q.numerator
        rs.append(o.r); cs.append(o.c); xs.append(o.x * f)
    if not rs:
        return Op(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, DEG), np.int64), 1)
    return coalesce(np.concatenate(rs), np.concatenate(cs), np.concatenate(xs), L, d)
def kron(fr, base, Kd):
    """fr: dict {(i,j): Fraction} on the frame C^3 ;  base: Op on C^Kd."""
    terms = []
    d = 3 * Kd
    for (i, j), q in fr.items():
        terms.append((q, Op(base.r + i * Kd, base.c + j * Kd, base.x, base.den)))
    return lincomb(terms, d)
def conj_op(o): return Op(o.r, o.c, fconj(o.x), o.den)
def adjoint(o, d):
    return coalesce(o.c, o.r, fconj(o.x), o.den, d)
def equal_ops(a, b, d):
    z = lincomb([(1, a), (-1, b)], d)
    return len(z.r) == 0
def to_csr(o, d):
    return [sp.csr_matrix((o.x[:, u], (o.r, o.c)), shape=(d, d), dtype=np.int64) for u in range(DEG)]
def matmul(a, b, d):
    A, B = to_csr(a, d), to_csr(b, d)
    rs, cs, xs = [], [], []
    for u in range(DEG):
        for v in range(DEG):
            P = (A[u] @ B[v]).tocoo()
            if P.nnz == 0: continue
            rs.append(P.row); cs.append(P.col)
            xs.append(P.data[:, None] * RED[u + v][None, :])
    if not rs:
        return Op(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, DEG), np.int64), 1)
    return coalesce(np.concatenate(rs).astype(np.int64), np.concatenate(cs).astype(np.int64),
                    np.concatenate(xs), a.den * b.den, d)
def is_projection(o, d):
    return equal_ops(o, adjoint(o, d), d) and equal_ops(matmul(o, o, d), o, d)

# ---------------------------------------------------------------- the Catalan cutoff
def catalan_words(N):
    words = [()]
    frontier = [()]
    while frontier:
        new = []
        for w in frontier:
            n = w.count('A'); ht = n - w.count('B')
            if n < N: new.append(w + ('A',))
            if ht >= 2: new.append(w + ('B',))
        words += new; frontier = new
    words.sort(key=lambda w: (w.count('A'), len(w), w))
    return words

def build(N, core):
    W = catalan_words(N)
    Dn = len(W)
    widx = {w: i for i, w in enumerate(W)}
    nA = [w.count('A') for w in W]; ht = [w.count('A') - w.count('B') for w in W]
    Kd = 9 * Dn; d = 3 * Kd
    corner_w = [1, 2, 2, 2, 2, 4, 4, 4, 4]
    def kidx(c, wi): return c * Dn + wi
    # ---- monomial unitaries on the base C^Kd:  U e_k = z^ph[k] e_perm[k]
    def mono(perm, ph): return (np.array(perm, np.int64), np.array(ph, np.int64) % 24)
    def ident(): return mono(range(Kd), [0] * Kd)
    def mmul(U, V):
        pu, fu = U; pv, fv = V
        return (pu[pv], (fv + fu[pv]) % 24)
    def minv(U):
        p, f = U; pi = np.empty_like(p); pi[p] = np.arange(len(p)); fi = np.empty_like(f); fi[p] = -f
        return (pi, fi % 24)
    def mono_op(U):
        p, f = U
        return Op(p.copy(), np.arange(Kd, dtype=np.int64), RED[f].copy(), 1)
    # word-space maps on H_N
    def permB():           # unitary extension of B inside each n-block (B preserves n)
        perm = [None] * Dn
        for n in range(N + 1):
            blk = [i for i in range(Dn) if nA[i] == n]
            dom = []; img = set()
            for i in blk:
                if ht[i] >= 2:
                    j = widx[W[i] + ('B',)]; perm[i] = j; img.add(j)
                else:
                    dom.append(i)
            cod = [j for j in blk if j not in img]
            assert len(dom) == len(cod)
            for i, j in zip(dom, cod): perm[i] = j
        return perm
    pB = permB()
    def corner_unitary(c, kind):
        perm = list(range(Kd)); ph = [0] * Kd
        for i in range(Dn):
            k = kidx(c, i)
            if kind == 'bplus':
                perm[k] = kidx(c, pB[i])
            elif kind == 'bminus':
                perm[k] = kidx(c, pB[i]); ph[k] = 0 if ht[i] >= 2 else 12
            elif kind == 'p':
                ph[k] = 0 if len(W[i]) == 0 else 12
            elif kind == 'q':
                ph[k] = 0 if ht[i] == 1 else 12
            elif kind == 'c':
                # U_c = 2C - I + 2i sqrt(C(I-C)), C = 2^-ht : ht0 -> 1, ht1 -> i, ht2 -> exp(2 pi i/3)
                ph[k] = {0: 0, 1: 6, 2: 8}[ht[i]]
        return mono(perm, ph)
    rc = {0: 0, 1: 1, 2: 5}           # corners e0, e1 = m1_00, e2 = m2_00
    magic = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}   # m_g,ab at magic[g][2a+b]
    flag = {1: 0, 2: 1}
    toks = {}
    for r in range(3):
        for k in ('bplus', 'bminus', 'p', 'q', 'c'):
            toks['u%d_%s' % (r, k)] = corner_unitary(rc[r], k)
    for g in (1, 2):
        for k in ('bplus', 'bminus', 'p', 'q', 'c'):
            perm = list(range(Kd)); ph = [0] * Kd
            loc = corner_unitary(0, k)            # the Catalan corner unitary, transported identically
            for c in [flag[g]] + magic[g]:
                for i in range(Dn):
                    j = int(loc[0][kidx(0, i)]) - 0     # image inside corner 0
                    perm[kidx(c, i)] = kidx(c, j % Dn); ph[kidx(c, i)] = int(loc[1][kidx(0, i)])
            toks['v%d_%s' % (g, k)] = mono(perm, ph)
    # a = |e2><e0| (x) A_N, completed inside density eigenspaces
    rho_key = lambda c, i: (Fr(corner_w[c], 4 ** nA[i]))
    init = {kidx(0, i): kidx(5, widx[W[i] + ('A',)]) for i in range(Dn) if nA[i] < N}
    blocks = {}
    for c in range(9):
        for i in range(Dn):
            blocks.setdefault(rho_key(c, i), []).append(kidx(c, i))
    permA = list(range(Kd))
    for lam, blk in blocks.items():
        dom = [k for k in blk if k not in init]
        img = {init[k] for k in blk if k in init}
        for k in blk:
            if k in init:
                assert rho_key(5, init[k] - 5 * Dn) == lam
                permA[k] = init[k]
        cod = [k for k in blk if k not in img]
        assert len(dom) == len(cod)
        for k, j in zip(sorted(dom), sorted(cod)): permA[k] = j
    toks['a_plus'] = mono(permA, [0] * Kd)
    toks['a_minus'] = mono(permA, [0 if k in init else 12 for k in range(Kd)])
    # FMS magic structure inside M_g = C^4 (x) H_N in the column-0 basis |ab>
    def on_magic(g, f4perm):
        perm = list(range(Kd))
        for ab in range(4):
            for i in range(Dn):
                perm[kidx(magic[g][ab], i)] = kidx(magic[g][f4perm[ab]], i)
        return perm
    flipA = [2, 3, 0, 1]; flipB = [1, 0, 3, 2]
    for g in (1, 2):
        toks['axisX%d' % g] = mono(on_magic(g, flipA), [0] * Kd)
        toks['axisZ%d' % g] = mono(on_magic(g, flipB), [0] * Kd)
        out = [k for k in range(Kd) if k // Dn not in magic[g]]
        ph = [12 if k in out else 0 for k in range(Kd)]
        toks['s_X1plus%d' % g] = mono(on_magic(g, flipA), ph)
        toks['s_Z2plus%d' % g] = mono(on_magic(g, flipB), ph)
    corner_of = {'root': 0}
    for g in (1, 2):
        for ab in range(4):
            corner_of['m%d_%d%d' % (g, ab // 2, ab % 2)] = magic[g][ab]
    for E, c in corner_of.items():
        toks['s_' + E] = mono(range(Kd), [0 if k // Dn == c else 12 for k in range(Kd)])
    import os
    NEG = os.environ.get('NEG', '')
    if NEG == 'ucphase':     # negative control: conjugate the u0_c completion only (v1_c keeps the old one)
        p_, f_ = toks['u0_c']; f_ = f_.copy()
        for i in range(Dn):
            if ht[i] == 1: f_[kidx(0, i)] = 18
        toks['u0_c'] = (p_, f_)
    for t in list(toks):
        if not t.startswith('s_'):
            toks[t + '~'] = minv(toks[t])
    # word unitaries
    words = [tuple(w['word']) for w in core['unitary_word_labels']]
    WU = []
    for w in words:
        U = ident()
        for t in w: U = mmul(U, toks[t])
        WU.append(U)
    # ---- base projections (Ops on C^Kd)
    def diag_op(vals_den):   # list of Fractions on the diagonal
        L = 1
        for v in vals_den: L = lcm(L, Fr(v).denominator)
        idx = [k for k in range(Kd) if vals_den[k] != 0]
        x = np.zeros((len(idx), DEG), np.int64)
        for n, k in enumerate(idx): x[n, 0] = Fr(vals_den[k]) * L
        return Op(np.array(idx, np.int64), np.array(idx, np.int64), x, L)
    def corner_proj(cs): return diag_op([1 if k // Dn in cs else 0 for k in range(Kd)])
    Ibase = corner_proj(range(9))
    def mono_proj_half(Uperm):   # (P_supp + U)/2 restricted: used for cell eigenprojections
        pass
    base_e = {E: corner_proj([c]) for E, c in corner_of.items()}
    for g in (1, 2):
        for nm, fp in (('X1plus%d' % g, flipA), ('Z2plus%d' % g, flipB)):
            Mg = corner_proj(magic[g])
            Uo = mono_op(mono(on_magic(g, fp), [0] * Kd))
            # (M_g + swap restricted to M_g)/2  ; swap acts as identity outside M_g, so subtract it
            outside = corner_proj([c for c in range(9) if c not in magic[g]])
            base_e[nm] = lincomb([(Fr(1, 2), Mg), (Fr(1, 2), Uo), (Fr(-1, 2), outside)], Kd)
    # ---- frame helpers
    E3 = lambda i, j: {(i, j): Fr(1)}
    Id3 = {(0, 0): Fr(1), (1, 1): Fr(1), (2, 2): Fr(1)}
    full_I = kron(Id3, Ibase, Kd)
    def graph(i, j, U):     # G_ij(U) = (S + E_ij (x) U + E_ji (x) U*)/2
        Uo = mono_op(U); Ua = mono_op(minv(U))
        return lincomb([(Fr(1, 2), kron({(i, i): 1, (j, j): 1}, Ibase, Kd)),
                        (Fr(1, 2), kron(E3(i, j), Uo, Kd)), (Fr(1, 2), kron(E3(j, i), Ua, Kd))], d)
    def offd(i, j, U):      # X_ij(U) = E_ij (x) U + h.c.
        return lincomb([(1, kron(E3(i, j), mono_op(U), Kd)), (1, kron(E3(j, i), mono_op(minv(U)), Kd))], d)
    pos = {'12': (0, 1), '13': (0, 2), '23': (1, 2)}
    rows_by_name = {r['name']: r for r in core['projection_linear_relations']}
    symbols = sorted({k for r in core['projection_linear_relations'] for k in r['terms'] if k != 'I'} | {'active1', 'active2'})
    S_g = {g: corner_proj([flag[g]] + magic[g]) for g in (1, 2)}
    cacheG = {}
    def sym_op(s):
        if s in ('F1', 'F2', 'F3'):
            i = int(s[1]) - 1; return kron(E3(i, i), Ibase, Kd)
        if s in ('S12', 'S13', 'S23'):
            i, j = pos[s[1:]]; return kron({(i, i): 1, (j, j): 1}, Ibase, Kd)
        if s.startswith('active'):
            return kron(Id3, S_g[int(s[-1])], Kd)
        for pre in ('K_G', 'K_H', 'G', 'H'):
            if s.startswith(pre) and s[len(pre):len(pre) + 2] in pos and s[len(pre) + 2] == '_' and s[len(pre) + 3:].isdigit():
                ij = s[len(pre):len(pre) + 2]; i, j = pos[ij]; w = int(s[len(pre) + 3:])
                U = WU[w]
                Sij = kron({(i, i): 1, (j, j): 1}, Ibase, Kd)
                X = offd(i, j, U)
                Z = lincomb([(1, kron(E3(i, i), Ibase, Kd)), (-1, kron(E3(j, j), Ibase, Kd))], d)
                G = lincomb([(Fr(1, 2), Sij), (Fr(1, 2), X)], d)
                if pre == 'G': return G
                if pre == 'K_G': return lincomb([(1, Sij), (-1, G)], d)
                H = lincomb([(Fr(1, 2), Sij), (Fr(3, 10), Z), (Fr(4, 10), X)], d)
                if pre == 'H': return H
                return lincomb([(1, Sij), (-1, H)], d)
        if s.startswith('D_'):
            r = rows_by_name[s[2:]]['terms']
            gs = [k for k, v in r.items() if Fr(v) == -2]
            assert len(gs) == 3 and Fr(r[s]) == 3 and Fr(r['I']) == 1
            terms = [(1, full_I)]
            for gsym in gs:
                ij = gsym[1:3]; i, j = pos[ij]; w = int(gsym[4:])
                terms.append((1, offd(i, j, WU[w])))
            return lincomb([(Fr(1, 3) * q, o) for q, o in terms], d)
        if s.startswith('EF_') or s.startswith('NF_'):
            E = s[3:-2]; i = int(s[-1]) - 1
            e = base_e[E] if s[0] == 'E' else lincomb([(1, Ibase), (-1, base_e[E])], Kd)
            return kron(E3(i, i), e, Kd)
        if s.startswith('E_'):
            return kron(Id3, base_e[s[2:]], Kd)
        if s.startswith('J_'):
            body = s[2:]; E, ij, ab = body.rsplit('_', 2)
            i, j = pos[ij]
            Sfr = {(i, i): Fr(1, 2), (j, j): Fr(1, 2)}
            plus = dict(Sfr); plus[(i, j)] = Fr(1, 2); plus[(j, i)] = Fr(1, 2)
            minus = dict(Sfr); minus[(i, j)] = Fr(-1, 2); minus[(j, i)] = Fr(-1, 2)
            fr = plus if ab[1] == '1' else minus                       # G_ref or S - G_ref
            e = base_e[E] if ab[0] == '1' else lincomb([(1, Ibase), (-1, base_e[E])], Kd)   # A or S - A
            return kron(fr, e, Kd)
        raise KeyError(s)
    # ---- FMS base effects (on C^Kd), then (x) I_3
    ROWS = [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == 1]
    # cells in the |ab> basis (a = XI bit, b = IZ bit):  4x4 real monomial matrices
    I2 = np.eye(2, dtype=np.int64); X2 = np.array([[0, 1], [1, 0]]); Z2 = np.diag([1, -1])
    Y2r = np.array([[0, -1], [1, 0]])   # Y = i * Y2r ;  Y(x)Y = - Y2r(x)Y2r
    kr = np.kron
    CELL = [[kr(Z2, I2), kr(I2, X2), kr(Z2, X2)],
            [kr(I2, Z2), kr(X2, I2), kr(X2, Z2)],
            [kr(Z2, Z2), kr(X2, X2), kr(Y2r, Y2r)]]   # YY -> -(Y(x)Y) = +Y2r(x)Y2r
    for rr in range(3):    # sanity: products of rows = +I, columns = +I,+I,-I
        assert (CELL[rr][0] @ CELL[rr][1] @ CELL[rr][2] == np.eye(4)).all()
    for cc, sg in zip(range(3), (1, 1, -1)):
        assert (CELL[0][cc] @ CELL[1][cc] @ CELL[2][cc] == sg * np.eye(4)).all()
    def proj4(ops, signs):
        P = np.eye(4, dtype=object)
        for O, sgn in zip(ops, signs):
            P = P.dot((np.eye(4, dtype=object) + sgn * O.astype(object))) * Fr(1, 2)
        return P
    def embed4(g, M4, scale_sqrt2=None):
        """M4: 4x4 Fraction matrix acting on magic corners of g, (x) I_H."""
        terms_r, terms_c, vals = [], [], []
        for p in range(4):
            for q in range(4):
                if M4[p][q] != 0:
                    for i in range(Dn):
                        terms_r.append(kidx(magic[g][p], i)); terms_c.append(kidx(magic[g][q], i)); vals.append(Fr(M4[p][q]))
        return vals_op(terms_r, terms_c, vals)
    def vals_op(rr, cc, vals, sq2=None):
        # vals: Fractions (times sqrt2 where sq2[n] is True)
        L = 1
        for v in vals: L = lcm(L, Fr(v).denominator)
        x = np.zeros((len(vals), DEG), np.int64)
        for n, v in enumerate(vals):
            base = SQRT2 if (sq2 is not None and sq2[n]) else ONE
            x[n] = base * int(Fr(v) * L)
        return coalesce(np.array(rr, np.int64), np.array(cc, np.int64), x, L, Kd)
    def flagP(g): return corner_proj([flag[g]])
    def inactive(g): return lincomb([(1, Ibase), (-1, S_g[g])], Kd)
    vec_c = {(1, 1, 1): {0: Fr(1)}, (1, -1, -1): {1: Fr(1)}}        # c_0 = |00>, c_1 = |01>
    # r_y = row-0 (+++) / (+--): XI=+ (a=0), IX=+/- : (|00> +/- |01>)/sqrt2  -> stored as coefficient * (1/sqrt2)
    vec_r = {(1, 1, 1): {0: Fr(1), 1: Fr(1)}, (1, -1, -1): {0: Fr(1), 1: Fr(-1)}}
    def sign_effects(g, vec, inv_sqrt2):
        """(Pi_f + Pi_v +/- (|v><f| + |f><v|))/2 with v = sum vec[ab]|ab> (times 1/sqrt2 if inv_sqrt2)."""
        effs = []
        for sgn in (1, -1):
            rr, cc, vals, s2 = [], [], [], []
            for i in range(Dn):
                f = kidx(flag[g], i)
                rr.append(f); cc.append(f); vals.append(Fr(1, 2)); s2.append(False)
                for p, cp in vec.items():
                    for q, cq in vec.items():
                        rr.append(kidx(magic[g][p], i)); cc.append(kidx(magic[g][q], i))
                        vals.append(cp * cq * (Fr(1, 2) if inv_sqrt2 else 1) / 2); s2.append(False)
                    vp = kidx(magic[g][p], i)
                    # off-diagonal: sgn * cp/2 (times 1/sqrt2 = sqrt2/2)
                    for (a_, b_) in ((vp, f), (f, vp)):
                        rr.append(a_); cc.append(b_)
                        if inv_sqrt2:
                            vals.append(sgn * cp / 4); s2.append(True)
                        else:
                            vals.append(sgn * cp / 2); s2.append(False)
            effs.append(vals_op(rr, cc, vals, s2))
        return effs
    fmsA_base, fmsB_base = {}, {}
    COLANS = {j: [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == (1, 1, -1)[j]] for j in range(3)}
    for g in (1, 2):
        for i in range(3):
            effs = [embed4(g, proj4([CELL[i][0], CELL[i][1]], t[:2])) for t in ROWS]
            fmsA_base[(g, ('R', i))] = effs + [flagP(g), inactive(g)]
            effsB = [embed4(g, proj4([CELL[0][i], CELL[1][i]], t[:2])) for t in COLANS[i]]
            fmsB_base[(g, ('C', i))] = effsB + [flagP(g), inactive(g)]
        for x, cx in ((0, (1, 1, 1)), (1, (1, -1, -1))):
            ret = [t for t in COLANS[0] if t != cx]
            fmsA_base[(g, ('T', x))] = sign_effects(g, vec_c[cx], False) + \
                [embed4(g, proj4([CELL[0][0], CELL[1][0]], t[:2])) for t in ret] + [inactive(g)]
        for y, ry in ((0, (1, 1, 1)), (1, (1, -1, -1))):
            ret = [t for t in ROWS if t != ry]
            fmsB_base[(g, ('U', y))] = sign_effects(g, vec_r[ry], True) + \
                [embed4(g, proj4([CELL[0][0], CELL[0][1]], t[:2])) for t in ret] + [inactive(g)]
    if NEG == 'bobsign':     # negative control: Bob's g=1 U0 answers '+' and '-' exchanged
        L_ = fmsB_base[(1, ('U', 0))]; L_[0], L_[1] = L_[1], L_[0]
    weights = np.array([corner_w[k // Dn] for k in range(Kd)]), np.array([nA[k % Dn] for k in range(Kd)])
    return dict(N=N, W=W, Dn=Dn, Kd=Kd, d=d, sym_op=sym_op, symbols=symbols, fmsA=fmsA_base, fmsB=fmsB_base,
                Id3=Id3, Ibase=Ibase, full_I=full_I, weights=weights, kron=lambda fr, b: kron(fr, b, Kd), WU=WU,
                words=words, corner_proj=corner_proj, magic=magic, flag=flag)

def run(N, archive_dir):
    t0 = time.time()
    Q = json.load(open(archive_dir + '/questions.json'))
    core = json.load(open(archive_dir + '/core.json'))
    raw = open(archive_dir + '/coefficients.bin', 'rb').read()
    sha = hashlib.sha256(raw).hexdigest()
    fmt = json.load(open(archive_dir + '/format.json'))
    assert sha == fmt['coefficient_sha256']
    M = build(N, core)
    d, Kd, Dn = M['d'], M['Kd'], M['Dn']
    res = {'N': N, 'catalan_dim': Dn, 'local_dim': d, 'coefficient_sha256': sha}
    # ---------- operators for every symbol
    symA = {s: M['sym_op'](s) for s in M['symbols']}
    nonproj = [s for s, o in symA.items() if not is_projection(o, d)]
    res['binary_symbols'] = len(symA); res['binary_symbols_not_projection'] = nonproj[:10]
    # ---------- every projection row as an operator
    bad_rows = {}
    for r in core['projection_linear_relations']:
        terms = []
        for k, v in r['terms'].items():
            terms.append((Fr(v), M['full_I'] if k == 'I' else symA[k]))
        R = lincomb(terms, d)
        if len(R.r): bad_rows[r['name']] = R
    res['nonzero_rows'] = sorted(bad_rows)
    # the boundary prediction:  -(E12 + E21) (x) e0 (x) Pi_N
    piN = M['corner_proj']([])  # initial corner projection
    vals = [Fr(1) if (k // Dn == 0 and M['W'][k % Dn].count('A') == N) else Fr(0) for k in range(Kd)]
    idx = [k for k in range(Kd) if vals[k]]
    xx = np.zeros((len(idx), DEG), np.int64); xx[:, 0] = 1
    e0PiN = Op(np.array(idx, np.int64), np.array(idx, np.int64), xx, 1)
    pred = M['kron']({(0, 1): Fr(-1), (1, 0): Fr(-1)}, e0PiN)
    if 'R_isometry' in bad_rows:
        res['R_isometry_row_equals_predicted_boundary'] = equal_ops(bad_rows['R_isometry'], pred, d)
    # ---------- question effects for both parties
    QA, Q_bob = Q['A'], Q['B']
    Id3 = M['Id3']
    effA, effB = {}, {}
    def fms_full(base): return M['kron'](Id3, base)
    fmsA = {k: [fms_full(b) for b in v] for k, v in M['fmsA'].items()}
    fmsB = {k: [fms_full(b) for b in v] for k, v in M['fmsB'].items()}
    def norm(a): return tuple(norm(t) for t in a) if isinstance(a, list) else a
    problems = []
    for party, QS, eff, fms in (('A', QA, effA, fmsA), ('B', Q_bob, effB, fmsB)):
        for x in QS:
            if x['kind'] == 'anchor':
                eff[x['id']] = [M['full_I']]
            elif x['kind'] == 'binary':
                P = symA[x['name']] if party == 'A' else conj_op(symA[x['name']])
                eff[x['id']] = [lincomb([(1, M['full_I']), (-1, P)], d), P]
            elif x['kind'] == 'fms':
                eff[x['id']] = fms[(x['g'], tuple(x['tag']))]
            elif x['kind'] == 'refined':
                coarse = eff[x['coarse']] if x['coarse'] in eff else None
                cq = [y for y in QS if y['id'] == x['coarse']][0]
                coarse = fms[(cq['g'], tuple(cq['tag']))]
                cans = [norm(a) for a in cq['answers']]
                Z = symA[x['label']] if party == 'A' else conj_op(symA[x['label']])
                Zc = lincomb([(1, M['full_I']), (-1, Z)], d)
                lst = []
                for a in x['answers']:
                    c, z = norm(a)
                    E = coarse[cans.index(c)]
                    lst.append(matmul(E, Z if z == 1 else Zc, d))
                eff[x['id']] = lst
            if x['kind'] in ('fms', 'refined'):
                tot = lincomb([(1, e) for e in eff[x['id']]], d)
                if not equal_ops(tot, M['full_I'], d): problems.append(('incomplete', party, x['name']))
                for ia, e in enumerate(eff[x['id']]):
                    if not is_projection(e, d): problems.append(('not projection', party, x['name'], ia))
    # Bob binary effects are Galois conjugates of projections, hence projections; check a sample anyway
    for x in Q_bob[1:200]:
        if x['kind'] == 'binary' and not is_projection(effB[x['id']][1], d):
            problems.append(('bob binary not projection', x['name']))
    res['measurement_problems'] = problems[:20]
    # ---------- density weights  r_i r_k = omega_ik / (75 Z_N 4^N)
    cw, nn = M['weights']
    cw = np.tile(cw, 3); nn = np.tile(nn, 3)
    ZN = sum(Fr(len([w for w in M['W'] if w.count('A') == n]), 4 ** n) for n in range(N + 1))
    def omega(r, c):
        prod_w = cw[r] * cw[c]
        out = np.zeros((len(r), DEG), np.int64)
        two = 2 ** (2 * N - nn[r] - nn[c])
        for val, vec in ((1, ONE), (2, SQRT2), (4, 2 * ONE), (8, 2 * SQRT2), (16, 4 * ONE)):
            m = prod_w == val
            out[m] = vec[None, :] * two[m][:, None]
        return out
    tilde = {}
    def tA(x, a):
        k = (x, a)
        if k not in tilde:
            o = effA[x][a]
            tilde[k] = (o.r * d + o.c, fmul(omega(o.r, o.c), o.x), o.den)
        return tilde[k]
    flatB = {}
    def fB(x, b):
        k = (x, b)
        if k not in flatB:
            o = effB[x][b]; flatB[k] = (o.r * d + o.c, o.x, o.den)
        return flatB[k]
    acc = {}; acc_ext = {}; nonzero_ext = 0; irr = 0; nrec = 0
    kind = {('A', x['id']): x['kind'] for x in QA}; kind.update({('B', x['id']): x['kind'] for x in Q_bob})
    for key, num in struct.iter_unpack('<Qq', raw):
        ea, eb = key >> 32, key & 0xffffffff
        xa, a = divmod(ea, 16); xb, b = divmod(eb, 16)
        kA, vA, dA = tA(xa, a); kB, vB, dB = fB(xb, b)
        _, ia, ib = np.intersect1d(kA, kB, assume_unique=True, return_indices=True)
        Mx = vA[ia].T @ vB[ib]
        p = np.einsum('uv,uvk->k', Mx, MULT)
        pv = [int(t) for t in p]
        den = dA * dB
        if any(pv[1:]): irr += 1
        ext = kind[('A', xa)] in ('fms', 'refined') or kind[('B', xb)] in ('fms', 'refined')
        tgt = acc_ext if ext else acc
        cur = tgt.setdefault(den, [0] * DEG)
        for u in range(DEG): cur[u] += num * pv[u]
        if ext and any(pv): nonzero_ext += 1
        nrec += 1
    scale = Fr(1, 2048) / (75 * ZN * 4 ** N)
    def total(ac):
        out = [Fr(0)] * DEG
        for den, vec in ac.items():
            for u in range(DEG): out[u] += Fr(vec[u], den) * scale
        return out
    tot_core = total(acc); tot_ext = total(acc_ext)
    tot = [tot_core[u] + tot_ext[u] for u in range(DEG)]
    tN = Fr(len([w for w in M['W'] if w.count('A') == N]), 4 ** N) / ZN
    res.update({'records_contracted': nrec, 'Z_N': str(ZN), 't_N': str(tN),
                'score': str(tot[0]), 'score_nonrational_coordinates_zero': all(t == 0 for t in tot[1:]),
                'predicted_minus_2tN_over_75': str(-2 * tN / 75), 'score_matches_prediction': tot[0] == -2 * tN / 75 and all(t == 0 for t in tot[1:]),
                'fms_refined_subtotal': [str(t) for t in tot_ext], 'fms_refined_records_with_nonzero_probability': nonzero_ext,
                'records_with_irrational_probability': irr, 'seconds': round(time.time() - t0, 1)})
    try:
        import psutil
        res['peak_working_set_MiB'] = round(psutil.Process().memory_info().peak_wset / 2 ** 20, 1)
    except Exception:
        pass
    return res

if __name__ == '__main__':
    out = []
    for N in map(int, sys.argv[2:]):
        r = run(N, sys.argv[1]); out.append(r)
        print(json.dumps(r, indent=1), flush=True)
    import os
    tag = os.environ.get('NEG', '')
    json.dump(out, open('my_evaluate_results_' + (tag + '_' if tag else '') + '_'.join(sys.argv[2:]) + '.json', 'w'), indent=1)
