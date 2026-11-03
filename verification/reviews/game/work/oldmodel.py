"""Reviewer's own construction of the cutoff-N canonical OLD strategy (turn-5 small presentation),
built from the definitions in out5/REPORT.md Sections 2 and 4 and the polynomial/row data in
core.json only. Nothing is imported from out5/out6 scripts.

Local old space: C^3 (scalar frame) x C^9 (corners) x l2(W_N).
Corners: 0 root(e0); 1..4 = m1_ab (ab=00,01,10,11), corner 1 = e1 = m1_00; 5..8 = m2_ab, corner 5 = e2.
Density lambda(f,c,w) = kappa_c 4^-n(w) /(75 Z_N), kappa = 1,2,4 for root, m1, m2.
Magic block basis m_ab: XI = (-1)^a, IZ = (-1)^b, ZI = sigma_x on a, IX = sigma_x on b.
All base unitaries are monomial (permutation x 24th-root phase) matrices on the corner space.
"""
import numpy as np
from fractions import Fraction as Fr
from math import comb
import field24 as F

def catalan(n): return comb(2 * n, n) // (n + 1)

class Mono:
    """U e_x = z^ph[x] e_perm[x]"""
    def __init__(self, perm, ph):
        self.perm = np.asarray(perm, dtype=np.int64); self.ph = np.asarray(ph, dtype=np.int64) % 24
    def __matmul__(self, o):  # (self o)(x) = self(o(x))
        return Mono(self.perm[o.perm], o.ph + self.ph[o.perm])
    def adj(self):
        inv = np.empty_like(self.perm); inv[self.perm] = np.arange(len(self.perm))
        return Mono(inv, -self.ph[inv])
    def dense(self):
        m = len(self.perm); A = np.zeros((8, m, m), dtype=np.int64)
        A[:, self.perm, np.arange(m)] = F.red[self.ph % 24].T
        return A
    def eq(self, o): return np.array_equal(self.perm, o.perm) and np.array_equal(self.ph % 24, o.ph % 24)

class OldModel:
    def __init__(self, N):
        self.N = N
        words = ['']
        frontier = ['']
        while frontier:
            nxt = []
            for w in frontier:
                n, h = w.count('A'), w.count('A') - w.count('B')
                if n < N: nxt.append(w + 'A')
                if h >= 2: nxt.append(w + 'B')
            words += nxt; frontier = nxt
        words = sorted(set(words), key=lambda w: (w.count('A'), len(w), w))
        self.words = words; NW = self.NW = len(words)
        self.nA = np.array([w.count('A') for w in words]); self.ht = np.array([w.count('A') - w.count('B') for w in words])
        for n in range(N + 1): assert (self.nA == n).sum() == catalan(n)
        self.widx = {w: i for i, w in enumerate(words)}
        self.m = 9 * NW; self.n = 3 * self.m
        self.ZN = sum(Fr(catalan(n), 4**n) for n in range(N + 1))
        kappa = np.array([1, 2, 2, 2, 2, 4, 4, 4, 4])
        self.kappa = kappa
        # corner-space density key (exponent of 2): log2(kappa) - 2n
        self.ckey = np.array([int(np.log2(kappa[c])) - 2 * self.nA[w] for c in range(9) for w in range(NW)])
        self.key = np.concatenate([self.ckey] * 3)
        self.lam = [Fr(int(kappa[c]), 4**int(self.nA[w])) / (75 * self.ZN) for f in range(3) for c in range(9) for w in range(NW)]
        self._build_base()
        self.mono_cache = {}

    # --- index helpers
    def ci(self, c, w): return c * self.NW + w

    def wmono(self, fn_perm_ph):
        """lift a W-operator given as (perm over W, phases) acting on corner set"""
        raise NotImplementedError

    def _build_base(self):
        N, NW, m = self.N, self.NW, self.m
        words, widx, ht, nA = self.words, self.widx, self.ht, self.nA
        ident = np.arange(m)
        def on_corners(corners, wperm, wph):
            perm = ident.copy(); ph = np.zeros(m, dtype=np.int64)
            for c in corners:
                for w in range(NW):
                    perm[self.ci(c, w)] = self.ci(c, wperm[w]); ph[self.ci(c, w)] = wph[w]
            return Mono(perm, ph)
        # W-operators as monomials (perm, ph)
        Wid = (np.arange(NW), np.zeros(NW, dtype=np.int64))
        up = (np.arange(NW), np.array([0 if words[w] == '' else 12 for w in range(NW)]))
        uq = (np.arange(NW), np.array([0 if ht[w] == 1 else 12 for w in range(NW)]))
        phc = {0: 0, 1: 6, 2: 8}  # u_c eigenvalue with Re = 2C-1, C = 2^-ht: 1, i, exp(2 pi i/3)
        assert ht.max() <= 2
        uc = (np.arange(NW), np.array([phc[int(ht[w])] for w in range(NW)]))
        # B and its completion Wb
        bperm = -np.ones(NW, dtype=np.int64)
        for w in range(NW):
            if ht[w] >= 2: bperm[w] = widx[words[w] + 'B']
        wbperm = bperm.copy()
        for n in range(N + 1):
            ker = [w for w in range(NW) if nA[w] == n and ht[w] <= 1]
            cok = [w for w in range(NW) if nA[w] == n and (words[w] == '' or words[w].endswith('A'))]
            assert len(ker) == len(cok)
            for x, y in zip(ker, cok): wbperm[x] = y
        assert sorted(wbperm.tolist()) == list(range(NW))
        isB = np.array([ht[w] >= 2 for w in range(NW)])
        ubp = (wbperm, np.zeros(NW, dtype=np.int64))
        ubm = (wbperm, np.where(isB, 0, 12))
        self.Wops = {'p': up, 'q': uq, 'c': uc, 'bplus': ubp, 'bminus': ubm}
        rc = {0: 0, 1: 1, 2: 5}
        base = {}
        for r in range(3):
            for k, op in self.Wops.items():
                base[f'u{r}_{k}'] = on_corners([rc[r]], *op)
        flag = {1: 0, 2: 1}; mag = {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}
        for g in (1, 2):
            for k, op in self.Wops.items():
                base[f'v{g}_{k}'] = on_corners([flag[g]] + mag[g], *op)
        # magic corner c = base + 2a + b
        def flipmono(g, which, outside_sign):
            perm = ident.copy(); ph = np.full(m, 12 if outside_sign < 0 else 0, dtype=np.int64)
            b0 = mag[g][0]
            for a in (0, 1):
                for b in (0, 1):
                    c = b0 + 2 * a + b
                    c2 = b0 + 2 * (1 - a) + b if which == 'a' else b0 + 2 * a + (1 - b)
                    for w in range(NW):
                        perm[self.ci(c, w)] = self.ci(c2, w); ph[self.ci(c, w)] = 0
            return Mono(perm, ph)
        for g in (1, 2):
            base[f'axisX{g}'] = flipmono(g, 'a', +1)
            base[f'axisZ{g}'] = flipmono(g, 'b', +1)
            base[f's_X1plus{g}'] = flipmono(g, 'a', -1)
            base[f's_Z2plus{g}'] = flipmono(g, 'b', -1)
        def cornerrefl(corner):
            ph = np.full(m, 12, dtype=np.int64)
            for w in range(NW): ph[self.ci(corner, w)] = 0
            return Mono(ident.copy(), ph)
        base['s_root'] = cornerrefl(0)
        for g in (1, 2):
            for a in (0, 1):
                for b in (0, 1):
                    base[f's_m{g}_{a}{b}'] = cornerrefl(mag[g][0] + 2 * a + b)
        # a_plus/a_minus = atilde +- Y
        dom = {self.ci(0, w): self.ci(5, widx[words[w] + 'A']) for w in range(NW) if nA[w] < N}
        rng = set(dom.values())
        perm = -np.ones(m, dtype=np.int64)
        for x, y in dom.items(): perm[x] = y
        dcomp = [x for x in range(m) if x not in dom]; rcomp = [y for y in range(m) if y not in rng]
        for kv in sorted(set(self.ckey.tolist())):
            dd = [x for x in dcomp if self.ckey[x] == kv]; rr = [y for y in rcomp if self.ckey[y] == kv]
            assert len(dd) == len(rr), kv
            for x, y in zip(dd, rr): perm[x] = y
        assert sorted(perm.tolist()) == list(range(m))
        isdom = np.zeros(m, dtype=bool); isdom[list(dom)] = True
        base['a_plus'] = Mono(perm, np.zeros(m, dtype=np.int64))
        base['a_minus'] = Mono(perm, np.where(isdom, 0, 12))
        for k, u in base.items():  # every base unitary commutes with the density (key preserved)
            assert np.array_equal(self.ckey[u.perm], self.ckey), k
        self.base = base
        self.Imono = Mono(ident.copy(), np.zeros(m, dtype=np.int64))
        # parameter projections on corner space, dense (8,m,m), denominator 2
        self.mag, self.flag = mag, flag

    def word(self, w):
        key = tuple(w)
        if key not in self.mono_cache:
            U = self.Imono
            for s in w:
                if s.endswith('~'): U = U @ self.base[s[:-1]].adj()
                else: U = U @ self.base[s]
            self.mono_cache[key] = U
        return self.mono_cache[key]

    def param_proj_c(self, name):
        """parameter projection on corner space, as (int array (8,m,m), den=2)"""
        m = self.m
        s = self.base['s_' + name]
        # E = (I + s)/2
        A = s.dense(); A[0] += np.eye(m, dtype=np.int64)
        return A, 2

    def cornerproj_c(self, corners):
        A = np.zeros((8, self.m, self.m), dtype=np.int64)
        for c in corners:
            for w in range(self.NW): A[0, self.ci(c, w), self.ci(c, w)] = 1
        return A
