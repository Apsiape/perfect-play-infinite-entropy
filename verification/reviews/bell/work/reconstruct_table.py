# Rebuild the intended Bell functional of out5 Sections 2-3 from the SPECIFICATION
# (core rows in core.json + the FMS predicate + the stated bindings), with my own
# conventions, and compare it with the literal coefficients.bin as linear functionals on
# all no-signalling behaviours (Collins-Gisin canonical form).  Written from scratch;
# imports nothing from out3/out4/out5.
import json, sys, struct, hashlib, itertools
from fractions import Fraction as Fr
from collections import defaultdict

d = sys.argv[1]
Q = json.load(open(d + '/questions.json'))
core = json.load(open(d + '/core.json'))
raw = open(d + '/coefficients.bin', 'rb').read()
QA, Q_bob = Q['A'], Q['B']
for qs in (QA, Q_bob):
    assert all(qs[i]['id'] == i for i in range(len(qs)))
    assert max(len(x['answers']) for x in qs) <= 16

def norm_ans(a):
    return tuple(norm_ans(t) for t in a) if isinstance(a, list) else a
ansA = [[norm_ans(a) for a in x['answers']] for x in QA]
ansB = [[norm_ans(a) for a in x['answers']] for x in Q_bob]
binA = {x['name']: x['id'] for x in QA if x['kind'] == 'binary'}
binB = {x['name']: x['id'] for x in Q_bob if x['kind'] == 'binary'}
assert binA == binB
for x in QA + Q_bob:
    if x['kind'] == 'binary':
        assert x['answers'] == [0, 1]
anchorA = [x['id'] for x in QA if x['kind'] == 'anchor']; anchorB = [x['id'] for x in Q_bob if x['kind'] == 'anchor']
assert len(anchorA) == 1 and len(anchorB) == 1
anA, anB = anchorA[0], anchorB[0]

# ---------------- the literal table ----------------
lit = defaultdict(Fr)
prev = -1
for key, num in struct.iter_unpack('<Qq', raw):
    assert key > prev; prev = key
    ea, eb = key >> 32, key & 0xffffffff
    xa, a = divmod(ea, 16); xb, b = divmod(eb, 16)
    assert a < len(ansA[xa]) and b < len(ansB[xb]), 'answer index out of range'
    lit[(xa, a, xb, b)] += Fr(num, 2048)

# ---------------- my reconstruction ----------------
rec = defaultdict(Fr)
def add(xa, a, xb, b, c):
    rec[(xa, a, xb, b)] += c
def add_event_pair(xa, Aset, xb, Bset, c):
    for a in Aset:
        for b in Bset:
            add(xa, a, xb, b, c)

# 1. F_alpha for every projection row:  (a0 + sum ai Ei)(a0 + sum aj Fj) + k sum(Ei + Fi - 2EiFi)
for r in core['projection_linear_relations']:
    terms = {k: Fr(v) for k, v in r['terms'].items()}
    a0 = terms.pop('I', Fr(0))
    kk = 1 + sum(v * v for v in terms.values()) / 2
    add(anA, 0, anB, 0, -a0 * a0)
    for s, al in terms.items():
        add(anA, 0, binB[s], 1, -a0 * al)
        add(binA[s], 1, anB, 0, -a0 * al)
    for s1, a1 in terms.items():
        for s2, a2 in terms.items():
            add(binA[s1], 1, binB[s2], 1, -a1 * a2)
    for s in terms:
        add(binA[s], 1, anB, 0, -kk)
        add(anA, 0, binB[s], 1, -kk)
        add(binA[s], 1, binB[s], 1, 2 * kk)
# 2. one synchrony mismatch per binary symbol
for s in binA:
    add(binA[s], 1, binB[s], 0, Fr(-1)); add(binA[s], 0, binB[s], 1, Fr(-1))

# 3. FMS predicate (my statement of out5 Section 2.2)
CX = {0: (1, 1, 1), 1: (1, -1, -1)}; RY = {0: (1, 1, 1), 1: (1, -1, -1)}
def active_ok(ta, tb, a, b):
    if ta[0] == 'R' and tb[0] == 'C':
        i, j = ta[1], tb[1]
        if a == 'f' and b == 'f': return True
        if (a == 'f') != (b == 'f'): return False
        return a[j] == b[i]
    if ta[0] == 'T' and tb == ('C', 0):
        if a in ('+', '-'): return b == 'f' or b == CX[ta[1]]
        return a == b
    if ta == ('R', 0) and tb[0] == 'U':
        if b in ('+', '-'): return a == 'f' or a == RY[tb[1]]
        return a == b
    if ta[0] == 'T' and tb[0] == 'U':
        if a in ('+', '-') and b in ('+', '-'):
            return (1 if a == '+' else -1) * (1 if b == '+' else -1) == (-1) ** (ta[1] * tb[1])
        return True
    return True
def pred(ta, tb, a, b):
    ia, ib = a == 'inactive', b == 'inactive'
    if ia or ib: return ia and ib
    return active_ok(ta, tb, a, b)

fmsA = {(x['g'], tuple(x['tag'])): x['id'] for x in QA if x['kind'] == 'fms'}
fmsB = {(x['g'], tuple(x['tag'])): x['id'] for x in Q_bob if x['kind'] == 'fms'}
assert len(fmsA) == 10 and len(fmsB) == 10
# check answer lists against the stated FMS alphabet
ROWS = [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == 1]
def cols(j): return [t for t in itertools.product([1, -1], repeat=3) if t[0] * t[1] * t[2] == (1, 1, -1)[j]]
for g in (1, 2):
    for i in range(3):
        assert ansA[fmsA[(g, ('R', i))]] == ROWS + ['f', 'inactive']
        assert ansB[fmsB[(g, ('C', i))]] == cols(i) + ['f', 'inactive']
    for x in range(2):
        assert ansA[fmsA[(g, ('T', x))]] == ['+', '-'] + [c for c in cols(0) if c != CX[x]] + ['inactive']
        assert ansB[fmsB[(g, ('U', x))]] == ['+', '-'] + [r for r in ROWS if r != RY[x]] + ['inactive']
for (g, ta), xa in fmsA.items():
    for (g2, tb), xb in fmsB.items():
        if g2 != g: continue
        for ia, a in enumerate(ansA[xa]):
            for ib, b in enumerate(ansB[xb]):
                if not pred(ta, tb, a, b):
                    add(xa, ia, xb, ib, Fr(-1))
# 4. refined questions
refA = [x for x in QA if x['kind'] == 'refined']; refB = [x for x in Q_bob if x['kind'] == 'refined']
assert len(refA) == 22 and len(refB) == 22
labels_expected = {1: ['F1', 'F2', 'F3', 'G12_0', 'G13_0', 'G23_0'] + [core['bridge_graph_labels']['1_' + k] for k in ('bplus', 'bminus', 'p', 'q', 'c')],
                   2: ['F1', 'F2', 'F3', 'G12_0', 'G13_0', 'G23_0'] + [core['bridge_graph_labels']['2_' + k] for k in ('bplus', 'bminus', 'p', 'q', 'c')]}
def check_bridge_labels():
    # bridge_graph_labels must be G12 of the words v_g_k
    wid = {tuple(w['word']): w['id'] for w in core['unitary_word_labels']}
    for g in (1, 2):
        for k in ('bplus', 'bminus', 'p', 'q', 'c'):
            assert core['bridge_graph_labels']['%d_%s' % (g, k)] == 'G12_%d' % wid[('v%d_%s' % (g, k),)]
check_bridge_labels()
for party, refs, fmsSame, fmsOpp, binOpp in (('A', refA, fmsA, fmsB, binB), ('B', refB, fmsB, fmsA, binA)):
    for g in (1, 2):
        got = sorted(x['label'] for x in refs if x['g'] == g)
        assert got == sorted(labels_expected[g]), (party, g, got)
    for x in refs:
        g = x['g']
        coarse_tag = ('T', 0) if party == 'A' else ('U', 0)
        assert x['coarse'] == fmsSame[(g, coarse_tag)]
        cans = (ansA if party == 'A' else ansB)[x['coarse']]
        rans = [norm_ans(a) for a in x['answers']]
        assert rans == [(c, z) for c in cans for z in (0, 1)], 'refined alphabet'
        opp_tags = [('U', 0), ('U', 1), ('C', 0)] if party == 'A' else [('T', 0), ('T', 1), ('R', 0)]
        for tb in opp_tags:
            xo = fmsOpp[(g, tb)]
            oans = (ansB if party == 'A' else ansA)[xo]
            for ir, (c, z) in enumerate(rans):
                for io, o in enumerate(oans):
                    ok = pred(coarse_tag, tb, c, o) if party == 'A' else pred(tb, coarse_tag, o, c)
                    if not ok:
                        if party == 'A': add(x['id'], ir, xo, io, Fr(-1))
                        else: add(xo, io, x['id'], ir, Fr(-1))
        # label matching z vs opposite binary Z
        Z = binOpp[x['label']]
        one = [i for i, (c, z) in enumerate(rans) if z == 1]; zero = [i for i, (c, z) in enumerate(rans) if z == 0]
        act = [i for i, (c, z) in enumerate(rans) if c != 'inactive']; ina = [i for i, (c, z) in enumerate(rans) if c == 'inactive']
        Act = binOpp['active%d' % g]
        for evs, P, val in ((one, Z, 0), (zero, Z, 1), (act, Act, 0), (ina, Act, 1)):
            for i in evs:
                if party == 'A': add(x['id'], i, P, val, Fr(-1))
                else: add(P, val, x['id'], i, Fr(-1))
# 5. bindings for the original FMS questions:  event E vs opposite binary P:  p(E,P=0)+p(notE,P=1)
def bind(party, xq, Eset, sym):
    n = len((ansA if party == 'A' else ansB)[xq])
    P = (binB if party == 'A' else binA)[sym]
    for i in range(n):
        val = 0 if i in Eset else 1
        if party == 'A': add(xq, i, P, val, Fr(-1))
        else: add(P, val, xq, i, Fr(-1))
flagsym = {1: 'E_root', 2: 'E_m1_00'}
for g in (1, 2):
    for (gg, t), xq in fmsA.items():
        if gg != g: continue
        bind('A', xq, [i for i, a in enumerate(ansA[xq]) if a != 'inactive'], 'active%d' % g)
    for (gg, t), xq in fmsB.items():
        if gg != g: continue
        bind('B', xq, [i for i, a in enumerate(ansB[xq]) if a != 'inactive'], 'active%d' % g)
    for i in range(3):
        xa = fmsA[(g, ('R', i))]; bind('A', xa, [ansA[xa].index('f')], flagsym[g])
        xb = fmsB[(g, ('C', i))]; bind('B', xb, [ansB[xb].index('f')], flagsym[g])
    mtrip = {'00': (1, 1, 1), '01': (1, -1, -1), '10': (-1, 1, -1), '11': (-1, -1, 1)}
    for ab, t in mtrip.items():
        xa = fmsA[(g, ('T', 1 if ab == '00' else 0))]
        bind('A', xa, [ansA[xa].index(t)], 'E_m%d_%s' % (g, ab))
        xb = fmsB[(g, ('C', 0))]
        bind('B', xb, [ansB[xb].index(t)], 'E_m%d_%s' % (g, ab))
    # X1 = cell (1,1): Alice R1 component 1, Bob C1 component 1 ; Z2 = cell (0,1): Alice R0 comp 1, Bob C1 comp 0
    xa = fmsA[(g, ('R', 1))]; bind('A', xa, [i for i, a in enumerate(ansA[xa]) if isinstance(a, tuple) and a[1] == 1], 'E_X1plus%d' % g)
    xb = fmsB[(g, ('C', 1))]; bind('B', xb, [i for i, a in enumerate(ansB[xb]) if isinstance(a, tuple) and a[1] == 1], 'E_X1plus%d' % g)
    xa = fmsA[(g, ('R', 0))]; bind('A', xa, [i for i, a in enumerate(ansA[xa]) if isinstance(a, tuple) and a[1] == 1], 'E_Z2plus%d' % g)
    xb = fmsB[(g, ('C', 1))]; bind('B', xb, [i for i, a in enumerate(ansB[xb]) if isinstance(a, tuple) and a[0] == 1], 'E_Z2plus%d' % g)

rec = {k: v for k, v in rec.items() if v != 0}

# ---------------- Collins-Gisin canonical form ----------------
nA = [len(a) for a in ansA]; nB = [len(b) for b in ansB]
def to_cg(tab):
    cg = defaultdict(Fr)
    for (x, a, y, b), c in tab.items():
        lx, ly = nA[x] - 1, nB[y] - 1
        xa_anchor, yb_anchor = (x == anA), (y == anB)
        if xa_anchor and yb_anchor:
            cg[('1',)] += c; continue
        if yb_anchor:
            if a < lx: cg[('A', x, a)] += c
            else:
                cg[('1',)] += c
                for a2 in range(lx): cg[('A', x, a2)] -= c
            continue
        if xa_anchor:
            if b < ly: cg[('B', y, b)] += c
            else:
                cg[('1',)] += c
                for b2 in range(ly): cg[('B', y, b2)] -= c
            continue
        if a < lx and b < ly:
            cg[('AB', x, a, y, b)] += c
        elif a == lx and b < ly:
            cg[('B', y, b)] += c
            for a2 in range(lx): cg[('AB', x, a2, y, b)] -= c
        elif a < lx and b == ly:
            cg[('A', x, a)] += c
            for b2 in range(ly): cg[('AB', x, a, y, b2)] -= c
        else:
            cg[('1',)] += c
            for a2 in range(lx): cg[('A', x, a2)] -= c
            for b2 in range(ly): cg[('B', y, b2)] -= c
            for a2 in range(lx):
                for b2 in range(ly): cg[('AB', x, a2, y, b2)] += c
    return {k: v for k, v in cg.items() if v != 0}

cl, cr = to_cg(lit), to_cg(rec)
diff = {k: cl.get(k, 0) - cr.get(k, 0) for k in set(cl) | set(cr)}
diff = {k: v for k, v in diff.items() if v != 0}
out = {
    'coefficient_sha256': hashlib.sha256(raw).hexdigest(),
    'literal_records': len(lit), 'reconstructed_full_entries': len(rec),
    'literal_equals_reconstruction_entrywise': lit == rec,
    'cg_terms_literal': len(cl), 'cg_terms_reconstructed': len(cr),
    'cg_difference_terms': len(diff),
    'cg_difference_sample': [(str(k), str(v)) for k, v in list(diff.items())[:20]],
    'literal_constant_term_cg': str(cl.get(('1',), 0)),
}
# entrywise differences (conventions may differ legitimately)
ed = {k: lit.get(k, 0) - rec.get(k, 0) for k in set(lit) | set(rec)}
ed = {k: v for k, v in ed.items() if v != 0}
out['entrywise_difference_count'] = len(ed)
out['entrywise_difference_sample'] = [(str(k), str(v)) for k, v in list(ed.items())[:10]]
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps(out, indent=1))
