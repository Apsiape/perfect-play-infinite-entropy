"""Reviewer's exact evaluation of the OLD part of the cutoff-N canonical strategy.
usage: python -I old_eval.py N [--noproj]
Outputs work/old-N{N}.json with:
  * polynomial relations, all 4,529 projection rows (exact), boundary residual of R_isometry
  * every one of the 5,592 binary projections: Hermitian, P^2=P, [P,h]=0, Tr(hP), sync loss
  * own FMS / refined canonical measurements: POVM checks and all 6,916 old forbidden events
  * per row gadget: Tr(h Sigma^* Sigma) with Sigma = sum_j V_j built from predicate data
"""
import sys, json, re, time, collections, math
import numpy as np
from fractions import Fraction as Fr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import field24 as F
from oldmodel import OldModel, Mono
import gamespec as GS

N = int(sys.argv[1]); DOPROJ = '--noproj' not in sys.argv
t0 = time.time()
G, Q, core = GS.load()
M = OldModel(N)
m, n = M.m, M.n
DEN = 60
rows = core['projection_linear_relations']
Wd = {w['id']: w['word'] for w in core['unitary_word_labels']}
out = {'N': N, 'old_local_dimension': n, 'NW': M.NW, 'Z_N': str(M.ZN)}

# ---------- polynomial relations
bad = []
for rel in core['polynomial_relations']:
    den = math.lcm(*(Fr(t['coefficient']).denominator for t in rel['terms']))
    acc = np.zeros((8, m, m), dtype=np.int64)
    for t in rel['terms']:
        acc += int(Fr(t['coefficient']) * den) * M.word(t['word']).dense()
    if acc.any(): bad.append(rel['name'])
out['polynomial_relations_failing'] = bad

# ---------- projections
def blocks_to_full(blocks):
    A = np.zeros((8, n, n), dtype=np.int64)
    for (fi, fj), B in blocks.items():
        A[:, fi*m:(fi+1)*m, fj*m:(fj+1)*m] += B
    return A
Ic = np.zeros((8, m, m), dtype=np.int64); Ic[0] = np.eye(m, dtype=np.int64)
def Gform(i, j, U, kind):
    """kind G: (1/2)(F_i+F_j+|i><j|U+|j><i|U*) ; H: (1/5)(4F_i+F_j+2|i><j|U+2|j><i|U*); numerators over DEN"""
    Ud = U.dense(); Ua = U.adj().dense()
    if kind == 'G':
        return {(i, i): 30*Ic, (j, j): 30*Ic, (i, j): 30*Ud, (j, i): 30*Ua}
    return {(i, i): 48*Ic, (j, j): 12*Ic, (i, j): 24*Ud, (j, i): 24*Ua}
# triangles from rows
tri = {}
for r in rows:
    t = r['terms']
    ds = [k for k in t if k.startswith('D_')]
    if ds:
        assert len(ds) == 1 and Fr(t[ds[0]]) == 3 and Fr(t['I']) == 1
        gs = {k: Fr(v) for k, v in t.items() if k.startswith('G')}
        assert len(gs) == 3 and all(v == -2 for v in gs.values())
        w = {}
        for k in gs:
            pr, wid = k[1:3], int(k.split('_')[1]); w[pr] = wid
        assert ds[0] not in tri; tri[ds[0]] = (w['12'], w['13'], w['23'])
params = core['parameters']
def paramE(p):
    A, d = M.param_proj_c(p)  # den 2
    return A
def proj(name):
    """dense numerator array over DEN"""
    if name in ('F1', 'F2', 'F3'):
        i = int(name[1]) - 1; return blocks_to_full({(i, i): DEN*Ic})
    mm = re.fullmatch(r'S(\d)(\d)', name)
    if mm:
        i, j = int(mm[1])-1, int(mm[2])-1; return blocks_to_full({(i, i): DEN*Ic, (j, j): DEN*Ic})
    mm = re.fullmatch(r'(K_)?([GH])(\d)(\d)_(\d+)', name)
    if mm:
        i, j = int(mm[3])-1, int(mm[4])-1
        B = blocks_to_full(Gform(i, j, M.word(Wd[int(mm[5])]), mm[2]))
        if mm[1]: B = blocks_to_full({(i, i): DEN*Ic, (j, j): DEN*Ic}) - B
        return B
    if name.startswith('D_'):
        a, b, c = (M.word(Wd[x]) for x in tri[name])
        return blocks_to_full({(0, 0): 20*Ic, (1, 1): 20*Ic, (2, 2): 20*Ic,
                               (0, 1): 20*a.dense(), (1, 0): 20*a.adj().dense(),
                               (0, 2): 20*b.dense(), (2, 0): 20*b.adj().dense(),
                               (1, 2): 20*c.dense(), (2, 1): 20*c.adj().dense()})
    mm = re.fullmatch(r'E_(.+)', name)
    if mm and mm[1] in params:
        E = paramE(mm[1]); return blocks_to_full({(f, f): 30*E for f in range(3)})
    mm = re.fullmatch(r'(EF|NF)_(.+)_(\d)', name)
    if mm and mm[2] in params:
        E = paramE(mm[2]); f = int(mm[3]) - 1
        return blocks_to_full({(f, f): 30*E if mm[1] == 'EF' else 30*(2*Ic - E)})
    mm = re.fullmatch(r'J_(.+)_(\d)(\d)_(\d)(\d)', name)
    if mm and mm[1] in params:
        E = paramE(mm[1]); nE = 2*Ic - E   # both over den 2
        i, j = int(mm[2])-1, int(mm[3])-1; ea, gb = int(mm[4]), int(mm[5])
        X = E if ea else nE
        # G0 = (1/2)[[I, I],[I, I]] on (i,j); S-G0 = (1/2)[[I,-I],[-I,I]]; X commutes with frame
        sgn = 1 if gb else -1
        return blocks_to_full({(i, i): 15*X, (j, j): 15*X, (i, j): sgn*15*X, (j, i): sgn*15*X})
    mm = re.fullmatch(r'active(\d)', name)
    if mm:
        g = int(mm[1]); C = M.cornerproj_c([M.flag[g]] + M.mag[g]); return blocks_to_full({(f, f): DEN*C for f in range(3)})
    raise KeyError(name)

key = M.key
lam = M.lam
lamarr_num = None
# h as rational diagonal: lambda_i = kappa 4^-n/(75 Z_N). Use integer weights q_i = kappa 4^(N-n)
qint = np.array([int(M.kappa[c]) * 4**(N - int(M.nA[w])) for f in range(3) for c in range(9) for w in range(M.NW)], dtype=np.int64)
hden = 75 * M.ZN * 4**N  # lambda_i = qint_i / hden
def trh(A, den):
    """Tr(h A) for a field matrix A/den, returns Fraction (asserts rational)"""
    d = np.einsum('kii->ki', A)  # (8, n)
    v = (d * qint[None, :]).sum(axis=1)
    assert not v[1:].any()
    return Fr(int(v[0]), den) / hden

binnames = [q['name'] for q in Q['A'][:G['old_questions']] if q['kind'] == 'binary']
trhE = {}
projchk = collections.Counter()
if DOPROJ:
    for nm in binnames:
        A = proj(nm)
        herm = np.array_equal(A, F.fadj(A))
        sq = np.array_equal(F.fmatmul(A, A), DEN * A)
        comm = not np.any(A[:, key[:, None] != key[None, :]])
        projchk['hermitian'] += herm; projchk['idempotent'] += sq; projchk['commutes_h'] += comm
        t = trh(A, DEN); trhE[nm] = t
        # literal sync loss: Tr(hE)+Tr(hE) - 2 Tr(K E K E), here [E,K]=0 verified so Tr(KEKE)=sum_ij lam_i E_ij E_ji
        EE = F.emul(A, np.swapaxes(A, 1, 2)).sum(axis=2)  # (8,n): sum_j E_ij E_ji
        v = (EE * qint[None, :]).sum(axis=1); assert not v[1:].any()
        tkek = Fr(int(v[0]), DEN*DEN) / hden
        if 2*t - 2*tkek != 0: projchk['sync_nonzero'] += 1
    out['binary_projection_checks'] = dict(projchk)
    out['binary_projections'] = len(binnames)
else:
    for nm in binnames: trhE[nm] = trh(proj(nm), DEN)
out['trhE'] = {k: str(v) for k, v in trhE.items()}
print('projections done', time.time() - t0, dict(projchk), flush=True)

# ---------- rows
cache = {}
def cproj(nm):
    if nm not in cache:
        if len(cache) > 400: cache.clear()
        cache[nm] = proj(nm)
    return cache[nm]
rowfail = {}
for ri, r in enumerate(rows):
    al = {k: Fr(v) for k, v in r['terms'].items()}
    L = math.lcm(*(v.denominator for v in al.values()))
    acc = np.zeros((8, n, n), dtype=np.int64)
    for k, v in al.items():
        c = int(v * L)
        if k == 'I': acc[0] += c * DEN * np.eye(n, dtype=np.int64)
        else: acc += c * cproj(k)
    if acc.any():
        # residual S = acc/(L*DEN); record Tr(h S^2) and the matrix support
        S2 = F.emul(acc, np.swapaxes(acc, 1, 2)).sum(axis=2)
        v = (S2 * qint[None, :]).sum(axis=1); assert not v[1:].any()
        nz = np.argwhere(np.any(acc != 0, axis=0))
        rowfail[r['name']] = dict(L=L, trhS2=str(Fr(int(v[0]), (L*DEN)**2) / hden), nnz=len(nz), acc=acc)
out['rows_checked'] = len(rows)
out['rows_failing'] = sorted(rowfail)
# boundary: expected S_iso = -(|1><2|+|2><1|) (x) e0 (x) Pi_N
if 'R_isometry' in rowfail:
    rf = rowfail['R_isometry']; acc = rf['acc']
    exp = np.zeros((8, n, n), dtype=np.int64)
    for w in range(M.NW):
        if M.nA[w] == N:
            x = M.ci(0, w)
            exp[0, 0*m + x, 1*m + x] = -1; exp[0, 1*m + x, 0*m + x] = -1
    out['R_isometry_boundary_matches'] = bool(np.array_equal(acc, exp * rf['L'] * DEN))
    out['R_isometry_trhS2'] = rf['trhS2']
    tN = Fr(math.comb(2*N, N)//(N+1), 4**N) / M.ZN
    out['t_N'] = str(tN); out['2t_N/75'] = str(2*tN/75)
print('rows done', time.time() - t0, out['rows_failing'], flush=True)

# ---------- row gadgets: Sigma = sum_j V_j from predicate data
gad = GS.row_gadgets(G, Q)
rz = {}
for dr, g in gad.items():
    d = g['d']
    acc = np.zeros((8, n, n), dtype=np.int64); acc[0] += DEN * np.eye(n, dtype=np.int64)   # V_0 = I
    for j, s in g['slots'].items():
        if s['var'] is None: acc[0] += s['sgn'] * DEN * np.eye(n, dtype=np.int64)
        else:
            E = cproj(s['var']); V = 2*E; V[0] -= DEN*np.eye(n, dtype=np.int64); acc += s['sgn'] * V
    if acc.any():

        # Tr(h Sigma^* Sigma) = sum_{i,j} lam_i conj(Sigma_ji) Sigma_ji
        Sg = F.emul(F.fconj(acc), acc)  # |Sigma_ji|^2 entrywise at (j,i)
        v = (Sg.sum(axis=1) * qint[None, :]).sum(axis=1)  # sum over j, weight lam_i (column index i)
        assert not v[1:].any()
        rz[dr] = str(Fr(int(v[0]), DEN*DEN) / hden)
out['row_gadgets_with_nonzero_Sigma'] = {Q['A'][dr]['row']: (rows[Q['A'][dr]['row']]['name'], gad[dr]['d'], val) for dr, val in rz.items()}
print('gadgets done', time.time() - t0, out['row_gadgets_with_nonzero_Sigma'], flush=True)

# ---------- FMS / refined canonical measurements (reviewer's construction)
P2 = {'Za': np.diag([1, 1, -1, -1]), 'Zb': np.diag([1, -1, 1, -1]),
      'Xa': np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]), 'Xb': np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]])}
I4 = np.eye(4, dtype=np.int64)
cell = [[P2['Za'], P2['Xb'], P2['Za'] @ P2['Xb']],
        [P2['Zb'], P2['Xa'], P2['Xa'] @ P2['Zb']],
        [P2['Za'] @ P2['Zb'], P2['Xa'] @ P2['Xb'], None]]
cell[2][2] = cell[2][0] @ cell[2][1]
for i in range(3):
    assert np.array_equal(cell[i][0] @ cell[i][1] @ cell[i][2], I4)
for j in range(3):
    assert np.array_equal(cell[0][j] @ cell[1][j] @ cell[2][j], I4 if j < 2 else -I4)
for i in range(3):
    for j in range(3):
        assert np.array_equal(cell[i][j], cell[i][j].T) and np.array_equal(cell[i][j] @ cell[i][j], I4)
# out5 binding: m_ab <-> column-0 triple (XI, IZ, XZ) = ((-1)^a, (-1)^b, (-1)^(a+b)); X1plus = cell(1,1), Z2plus = cell(0,1)
def magic_to_corner(g, Mat4, den):
    """embed a 4x4 rational matrix (numerators over den) acting on m_ab (x) I_W into corner space"""
    A = np.zeros((8, m, m), dtype=np.int64)
    b0 = M.mag[g][0]
    for x in range(4):
        for y in range(4):
            if Mat4[x, y]:
                for w in range(M.NW):
                    A[0, M.ci(b0 + x, w), M.ci(b0 + y, w)] += Mat4[x, y]
    return A
def joint4(cells_, signs):
    P = np.eye(4, dtype=np.int64)
    for C, s in zip(cells_, signs):
        P = P @ (I4 + s * C)
    return P  # numerators over 2^len
def full_from_corner(Ac):
    return blocks_to_full({(f, f): Ac for f in range(3)})
FD = 8   # denominator for FMS effects
def corner_proj_full(corners, den=FD):
    return full_from_corner(den * M.cornerproj_c(corners))
def swap_corner(c1, c2):
    A = np.zeros((8, m, m), dtype=np.int64)
    for w in range(M.NW):
        A[0, M.ci(c1, w), M.ci(c2, w)] = 1; A[0, M.ci(c2, w), M.ci(c1, w)] = 1
    return A
def fms_effect(party, q):
    """returns list of (8,n,n) numerator arrays over FD for each answer, in the questions.json order"""
    g = q['g']; tag = q['tag']; ans = q['answers']
    fl = M.flag[g]; mg = M.mag[g]
    act = [fl] + mg
    inact = [c for c in range(9) if c not in act]
    res = []
    for a in ans:
        if a == 'inactive': res.append(corner_proj_full(inact)); continue
        if a == 'f': res.append(corner_proj_full([fl])); continue
        if tag[0] in 'RC':
            idx = tag[1]
            cl = [cell[idx][j] for j in range(3)] if tag[0] == 'R' else [cell[i][idx] for i in range(3)]
            P = joint4(cl, a)  # over 8
            res.append(full_from_corner(magic_to_corner(g, P, 8))); continue
        if tag[0] == 'T':
            x = tag[1]; cx = mg[0] + x   # c_0 = m00 (+++), c_1 = m01 (+--)
            if a in ('+', '-'):
                s = 1 if a == '+' else -1
                Ac = 4 * M.cornerproj_c([fl, cx]) + s * 4 * swap_corner(fl, cx)
                res.append(full_from_corner(Ac)); continue
            # retained column-0 triple (XI, IZ, XZ)
            aa, bb = (1 - a[0]) // 2, (1 - a[1]) // 2
            assert a[2] == a[0] * a[1]
            res.append(corner_proj_full([mg[0] + 2*aa + bb])); continue
        if tag[0] == 'U':
            y = tag[1]
            # row-0 (XI, IX, XX) eigenvector for triple tau: |a> (x) (|0> + tau1 |1>)/sqrt2
            def rvec(tau):
                aa = (1 - tau[0]) // 2; v = np.zeros(4, dtype=np.int64); v[2*aa] = 1; v[2*aa+1] = tau[1]; return v
            ry = rvec([1, 1, 1] if y == 0 else [1, -1, -1])
            if a in ('+', '-'):
                s = 1 if a == '+' else -1
                # (f + Pi_r + s(|r><f| + |f><r|))/2 ; Pi_r = v v^T/2 ; |r> = v/sqrt2
                Ac = np.zeros((8, m, m), dtype=np.int64)
                Ac += 4 * M.cornerproj_c([fl])
                Ac += 2 * magic_to_corner(g, np.outer(ry, ry), 1)  # 4 * Pi_r/... : Pi_r = vv^T/2 -> 8*Pi_r/2 = 2 vv^T
                # s (|r><f| + |f><r|) /2 * 8 = 4 s (v/sqrt2)(..) = 2 sqrt2 s v
                for k in range(4):
                    if ry[k]:
                        for w in range(M.NW):
                            i1, i2 = M.ci(mg[0] + k, w), M.ci(fl, w)
                            Ac[:, i1, i2] += s * 2 * int(ry[k]) * F.SQRT2
                            Ac[:, i2, i1] += s * 2 * int(ry[k]) * F.SQRT2
                res.append(full_from_corner(Ac)); continue
            v = rvec(a); assert a[2] == a[0] * a[1]
            res.append(full_from_corner(magic_to_corner(g, 4 * np.outer(v, v), 1))); continue
        raise ValueError((party, q['name'], a))
    return res
def label_proj(lbl):
    return proj(lbl)  # over DEN
EFF = {'A': {}, 'B': {}}
for party in 'AB':
    for q in Q[party][:G['old_questions']]:
        if q['kind'] == 'fms':
            EFF[party][q['id']] = (fms_effect(party, q), FD)
    for q in Q[party][:G['old_questions']]:
        if q['kind'] == 'refined':
            coarse, cden = EFF[party][q['coarse']]
            cq = Q[party][q['coarse']]
            Z = label_proj(q['label']); IZ = -Z.copy(); IZ[0] += DEN * np.eye(n, dtype=np.int64)
            res = []
            for (orig, z) in q['answers']:
                k = cq['answers'].index(orig)
                Eo = coarse[k]
                res.append(F.fmatmul(Eo, Z if z == 1 else IZ))
            EFF[party][q['id']] = (res, FD * DEN)
# POVM / projection / commutation checks for FMS and refined effects
fmschk = collections.Counter()
for party in 'AB':
    for qid, (effs, den) in EFF[party].items():
        tot = sum(effs)
        fmschk['povm_sum_identity'] += np.array_equal(tot, den * np.eye(n, dtype=np.int64)[None] * F.ONE[:, None, None])
        for E in effs:
            fmschk['effects'] += 1
            fmschk['hermitian'] += np.array_equal(E, F.fadj(E))
            fmschk['idempotent'] += np.array_equal(F.fmatmul(E, E), den * E)
out['fms_refined_checks'] = dict(fmschk); out['fms_refined_questions'] = {p: len(EFF[p]) for p in 'AB'}
print('fms built', time.time() - t0, dict(fmschk), flush=True)
# weights w_ij = s_i s_j 2^(2N - n_i - n_j), s = sqrt(kappa)
sv = []
for f in range(3):
    for c in range(9):
        for w in range(M.NW):
            kap = int(M.kappa[c]); e = int(M.nA[w])
            base = {1: F.ONE, 2: F.SQRT2, 4: 2 * F.ONE}[kap]
            sv.append(base * 2**(N - e))
sv = np.array(sv).T  # (8, n)
Wt = F.emul(sv[:, :, None], sv[:, None, :])  # (8,n,n)
def effect_of(party, qid, a):
    q = Q[party][qid]
    if q['kind'] == 'anchor': return DEN * np.eye(n, dtype=np.int64)[None] * F.ONE[:, None, None], DEN
    if q['kind'] == 'binary':
        E = cproj(q['name'])
        if a == 1: return E, DEN
        X = -E.copy(); X[0] += DEN * np.eye(n, dtype=np.int64); return X, DEN
    effs, den = EFF[party][qid]; return effs[a], den
WA_cache = {}
nonzero_events = []
for (qa, a, qb, b) in G['old_forbidden']:
    kA = (qa, a)
    if kA not in WA_cache:
        A_, dA = effect_of('A', qa, a); WA_cache[kA] = (F.emul(Wt, A_), dA)
    WA, dA = WA_cache[kA]
    B_, dB = effect_of('B', qb, b)
    s = F.emul(WA, np.swapaxes(B_, 1, 2)).sum(axis=(1, 2))   # sum_ij w_ij A_ij B_ji, B = Bob Alice-form
    if s.any(): nonzero_events.append((qa, a, qb, b, s.tolist()))
out['old_forbidden_events_evaluated'] = len(G['old_forbidden'])
out['old_forbidden_events_nonzero'] = len(nonzero_events)
out['old_forbidden_nonzero_sample'] = nonzero_events[:10]
# normalisation sanity: total probability of one FMS question pair equals 1
qa0 = [q['id'] for q in Q['A'][:G['old_questions']] if q['kind'] == 'fms'][0]
qb0 = [q['id'] for q in Q['B'][:G['old_questions']] if q['kind'] == 'fms'][3]
tot = np.zeros(8, dtype=np.int64)
for a in range(len(Q['A'][qa0]['answers'])):
    A_, dA = effect_of('A', qa0, a)
    for b in range(len(Q['B'][qb0]['answers'])):
        B_, dB = effect_of('B', qb0, b)
        tot += F.emul(F.emul(Wt, A_), np.swapaxes(B_, 1, 2)).sum(axis=(1, 2))
assert not tot[1:].any()
out['normalisation_check_fms_pair'] = str(Fr(int(tot[0]), FD*FD) / hden)
out['seconds'] = time.time() - t0
json.dump(out, open(Path(__file__).parent / f'old-N{N}.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != 'trhE'}, indent=1, default=str))
