# Independent audit of the small core (turn 5, Section 7).
# Written from scratch for this review; imports nothing from out3/out4/out5.
#  Part A: re-expand the 41 polynomial inputs from the Section 7 formulas
#          (my own noncommutative expansion) and compare with core.json.
#  Part B: regenerate all 4,529 projection rows from the out4 compiler rules
#          (G1, supports, copy/product/inverse triangles, G3, G4) and compare.
#  Part C: compare the original 49-input presentation (out4/presentation.json)
#          with the small one, input by input, to see exactly what changed.
import json, sys, itertools
from fractions import Fraction as Fr

core_path = sys.argv[1]
orig_path = sys.argv[2] if len(sys.argv) > 2 else None
core = json.load(open(core_path))

INVOL = lambda t: t.startswith('s_')          # parameter involutions s_E

def inv_tok(t):
    if INVOL(t):
        return t
    return t[:-1] if t.endswith('~') else t + '~'

def reduce_word(w):
    out = []
    for t in w:
        if out and out[-1] == inv_tok(t):
            out.pop()
        else:
            out.append(t)
    return tuple(out)

class NC:
    def __init__(self, d=None):
        self.d = {}
        if d:
            for k, v in d.items():
                self.add(k, v)
    def add(self, w, c):
        w = reduce_word(w)
        v = self.d.get(w, 0) + c
        if v == 0:
            self.d.pop(w, None)
        else:
            self.d[w] = v
    def __add__(s, o):
        o = lift(o); r = NC(s.d)
        for k, v in o.d.items(): r.add(k, v)
        return r
    __radd__ = __add__
    def __neg__(s): return NC({k: -v for k, v in s.d.items()})
    def __sub__(s, o): return s + (-lift(o))
    def __rsub__(s, o): return lift(o) - s
    def __mul__(s, o):
        o = lift(o); r = NC()
        for k1, v1 in s.d.items():
            for k2, v2 in o.d.items():
                r.add(k1 + k2, v1 * v2)
        return r
    def __rmul__(s, o): return lift(o) * s
    def adj(s):
        return NC({tuple(inv_tok(t) for t in reversed(k)): v for k, v in s.d.items()})  # real coefficients
    def __eq__(s, o): return s.d == lift(o).d

def lift(x):
    if isinstance(x, NC): return x
    return NC({(): Fr(x)}) if x != 0 else NC()

I = lift(1)
def g(t): return NC({(t,): Fr(1)})
def e(name): return (I + g('s_' + name)) * Fr(1, 2)

root = e('root')
m = {(gg, ab): e('m%d_%s' % (gg, ab)) for gg in (1, 2) for ab in ('00', '01', '10', '11')}
M = {gg: sum((m[(gg, ab)] for ab in ('00', '01', '10', '11')), NC()) for gg in (1, 2)}
EX = {gg: e('X1plus%d' % gg) for gg in (1, 2)}
EZ = {gg: e('Z2plus%d' % gg) for gg in (1, 2)}
er = {0: root, 1: m[(1, '00')], 2: m[(2, '00')]}
K = ['bplus', 'bminus', 'p', 'q', 'c']
u = {(r, k): g('u%d_%s' % (r, k)) for r in range(3) for k in K}
v = {(gg, k): g('v%d_%s' % (gg, k)) for gg in (1, 2) for k in K}
ax = {gg: g('axisX%d' % gg) for gg in (1, 2)}
az = {gg: g('axisZ%d' % gg) for gg in (1, 2)}
a = (g('a_plus') + g('a_minus')) * Fr(1, 2)

# Section 7 small definitions (final e removed)
def b_(r): return er[r] * (u[(r, 'bplus')] + u[(r, 'bminus')]) * Fr(1, 2)
def p_(r): return er[r] * (I + u[(r, 'p')]) * Fr(1, 2)
def q_(r): return er[r] * (I + u[(r, 'q')]) * Fr(1, 2)
def c_(r): return er[r] * (2 * I + u[(r, 'c')] + u[(r, 'c')].adj()) * Fr(1, 4)

expected = {}
for gg in (1, 2):
    expected['axis_X_%d' % gg] = ax[gg] - (I - 2 * M[gg] + 2 * EX[gg])
    expected['axis_Z_%d' % gg] = az[gg] - (I - 2 * M[gg] + 2 * EZ[gg])
expected['cover'] = root + M[1] + M[2] - I
for r in range(3):
    for k in K:
        expected['unitary_outside_%d_%s' % (r, k)] = (u[(r, k)] - I) * (I - er[r])
expected['reflection_p'] = u[(0, 'p')] * u[(0, 'p')] - I
expected['reflection_q'] = u[(0, 'q')] * u[(0, 'q')] - I
for gg in (1, 2):
    R = {('0', '0'): I, ('1', '0'): ax[gg], ('0', '1'): az[gg], ('1', '1'): ax[gg] * az[gg]}
    for k in K:
        rhs = I - (er[gg - 1] + M[gg]) + er[gg - 1] * u[(gg - 1, k)]
        for key, Rab in R.items():
            rhs = rhs + Rab * er[gg] * u[(gg, k)] * Rab.adj()
        expected['bridge_block_%d_%s' % (gg, k)] = v[(gg, k)] - rhs
expected['a_support'] = a - er[2] * a * er[0]
expected['R_isometry'] = a.adj() * a - er[0]
expected['R_range'] = a * a.adj() + b_(2) * b_(2).adj() + p_(2) - er[2]
expected['R_B_initial'] = b_(0).adj() * b_(0) - (er[0] - p_(0) - q_(0))
expected['PQ'] = p_(0) * q_(0)
expected['R_CP'] = c_(0) * p_(0) - p_(0)
expected['R_CQ'] = c_(0) * q_(0) - q_(0) * Fr(1, 2)
expected['R_CA'] = c_(2) * a - a * c_(0) * Fr(1, 2)
expected['R_CB'] = c_(0) * b_(0) - 2 * b_(0) * c_(0)

def from_json(terms):
    r = NC()
    for t in terms:
        r.add(tuple(t['word']), Fr(t['coefficient']))
    return r

report = {'A': {}, 'B': {}, 'C': {}}
polys = {p['name']: from_json(p['terms']) for p in core['polynomial_relations']}
okA = True
for name, exp in expected.items():
    got = polys.get(name)
    if got is None:
        report['A'][name] = 'MISSING'; okA = False; continue
    # compare up to scalar
    k0 = next(iter(exp.d))
    if k0 not in got.d:
        report['A'][name] = 'MISMATCH'; okA = False; continue
    lam = got.d[k0] / exp.d[k0]
    same = (got == exp * lam)
    report['A'][name] = ('match, scale %s' % lam) if same else 'MISMATCH'
    okA &= same
extra = set(polys) - set(expected)
report['A']['_extra_inputs_in_core'] = sorted(extra)
report['A']['_count'] = len(polys)
report['A']['_all_match'] = okA and not extra

# ---------------- Part B: regenerate the projection rows -----------------
words = [tuple(w['word']) for w in core['unitary_word_labels']]
wid = {w: i for i, w in enumerate(words)}
assert len(wid) == len(words), 'duplicate words'
assert all(core['unitary_word_labels'][i]['id'] == i for i in range(len(words)))
tokens = sorted({t for w in words for t in w})
probs = []
for w in words:
    if len(w) >= 2 and w[:-1] not in wid: probs.append(('missing prefix', w))
    for t in w:
        if (t,) not in wid: probs.append(('missing token word', t))
    if reduce_word(w) != w: probs.append(('unreduced word', w))
for t in tokens:
    if (inv_tok(t),) not in wid: probs.append(('missing inverse token', t))
for name, p in polys.items():
    for w in p.d:
        if w not in wid: probs.append(('poly word not compiled', name, w))
report['B']['word_problems'] = probs[:20]
report['B']['n_words'] = len(words)
report['B']['n_tokens'] = len(tokens)
report['B']['max_word_len'] = max(len(w) for w in words)

def row(**kw):
    return {k: Fr(v) for k, v in kw.items() if Fr(v) != 0}

gen = {}
def put(name, terms):
    assert name not in gen, name
    gen[name] = {k: Fr(v) for k, v in terms.items() if Fr(v) != 0}

put('frame_partition', {'F1': 1, 'F2': 1, 'F3': 1, 'I': -1})
pairs = [('1', '2'), ('1', '3'), ('2', '3')]
for i, j in pairs:
    put('S%s%s' % (i, j), {'S%s%s' % (i, j): 1, 'F' + i: -1, 'F' + j: -1})
def tri(name, d, g12, g23, g13):
    t = {d: 3, 'I': 1}
    for s in (g12, g23, g13):
        t[s] = t.get(s, 0) - 2
    put(name, t)
for idx, w in enumerate(words):
    for i, j in pairs:
        S = 'S%s%s' % (i, j); G = 'G%s%s_%d' % (i, j, idx); H = 'H%s%s_%d' % (i, j, idx)
        put('support_' + G, {G: 1, 'K_' + G: 1, S: -1})
        put('support_' + H, {H: 1, 'K_' + H: 1, S: -1})
        put('offdiag_' + G, {H: 5, S: 1, 'F' + i: -3, G: -4})
tri('reference', 'D_reference', 'G12_0', 'G23_0', 'G13_0')
for idx, w in enumerate(words):
    if idx == 0: continue
    tri('copy13_%d' % idx, 'D_copy13_%d' % idx, 'G12_%d' % idx, 'G23_0', 'G13_%d' % idx)
    tri('copy23_%d' % idx, 'D_copy23_%d' % idx, 'G12_0', 'G23_%d' % idx, 'G13_%d' % idx)
    if len(w) >= 2:
        tri('product_%d' % idx, 'D_product_%d' % idx, 'G12_%d' % wid[w[:-1]], 'G23_%d' % wid[w[-1:]], 'G13_%d' % idx)
gens = sorted({t for t in tokens if not t.endswith('~')})
for t in gens:
    tri('inverse_' + t, 'D_inverse_' + t, 'G12_%d' % wid[(t,)], 'G23_%d' % wid[(inv_tok(t),)], 'G13_0')
for name, p in polys.items():
    t = {}
    tot = Fr(0)
    for w, q in p.d.items():
        key = 'G12_%d' % wid[w]
        t[key] = t.get(key, 0) + 2 * q
        tot += q
    if tot != 0:
        t['S12'] = t.get('S12', 0) - tot
    put(name, t)
for E in core['parameters']:
    put('param_E_' + E, {**{'EF_%s_%s' % (E, i): 1 for i in '123'}, 'E_' + E: -1})
    put('param_notE_' + E, {**{'NF_%s_%s' % (E, i): 1 for i in '123'}, 'E_' + E: 1, 'I': -1})
    for i in '123':
        put('param_frame_%s_%s' % (E, i), {'EF_%s_%s' % (E, i): 1, 'NF_%s_%s' % (E, i): 1, 'F' + i: -1})
    for i, j in pairs:
        ij = i + j
        J = lambda ab: 'J_%s_%s_%s' % (E, ij, ab)
        put('param_joint_ES_%s_%s' % (E, ij), {J('10'): 1, J('11'): 1, 'EF_%s_%s' % (E, i): -1, 'EF_%s_%s' % (E, j): -1})
        put('param_joint_notES_%s_%s' % (E, ij), {J('00'): 1, J('01'): 1, 'NF_%s_%s' % (E, i): -1, 'NF_%s_%s' % (E, j): -1})
        put('param_joint_G_%s_%s' % (E, ij), {J('01'): 1, J('11'): 1, 'G%s_0' % ij: -1})
        put('parameter_graph_%s_%s' % (E, ij), {'G%s_%d' % (ij, wid[('s_' + E,)]): 1, J('00'): -1, J('11'): -1})

file_rows = {}
for r in core['projection_linear_relations']:
    assert r['name'] not in file_rows
    file_rows[r['name']] = {k: Fr(v) for k, v in r['terms'].items()}
mism = []; scales = {}
for name in set(gen) | set(file_rows):
    a_, b_r = gen.get(name), file_rows.get(name)
    if a_ is None or b_r is None:
        mism.append((name, 'only in ' + ('file' if a_ is None else 'regenerated'))); continue
    if set(a_) != set(b_r):
        mism.append((name, 'support differs')); continue
    k0 = next(iter(a_)); lam = b_r[k0] / a_[k0]
    if any(b_r[k] != lam * a_[k] for k in a_):
        mism.append((name, 'coefficients differ'))
    else:
        scales[str(lam)] = scales.get(str(lam), 0) + 1
report['B']['n_rows_file'] = len(file_rows)
report['B']['n_rows_regenerated'] = len(gen)
report['B']['mismatches'] = mism[:30]
report['B']['n_mismatches'] = len(mism)
report['B']['row_scale_histogram'] = scales
syms = sorted({k for r in file_rows.values() for k in r if k != 'I'})
report['B']['n_row_symbols'] = len(syms)

# ---------------- Part C: compare with the original 49-input presentation --------
if orig_path:
    orig = json.load(open(orig_path))
    opolys = {p['name']: from_json(p['terms']) for p in orig['polynomial_relations']}
    report['C']['orig_names'] = sorted(opolys)
    report['C']['orig_count'] = len(opolys)
    report['C']['orig_parameters'] = orig.get('parameters')
    report['C']['orig_n_rows'] = len(orig['projection_linear_relations'])
    report['C']['orig_n_words'] = len(orig['unitary_word_labels'])
    # Original definitions (out4 P2/P3 with both e factors, t built from expanded reflections)
    xg = {gg: I - 2 * M[gg] + 2 * EX[gg] for gg in (1, 2)}
    zg = {gg: I - 2 * M[gg] + 2 * EZ[gg] for gg in (1, 2)}
    def bo(r): return er[r] * (u[(r, 'bplus')] + u[(r, 'bminus')]) * er[r] * Fr(1, 2)
    def po(r): return er[r] * (I + u[(r, 'p')]) * er[r] * Fr(1, 2)
    def qo(r): return er[r] * (I + u[(r, 'q')]) * er[r] * Fr(1, 2)
    def co(r): return er[r] * (2 * I + u[(r, 'c')] + u[(r, 'c')].adj()) * er[r] * Fr(1, 4)
    oexp = {}
    oexp['cover'] = expected['cover']
    for r in range(3):
        for k in K:
            oexp['unitary_outside_%d_%s' % (r, k)] = expected['unitary_outside_%d_%s' % (r, k)]
    oexp['reflection_p'] = expected['reflection_p']; oexp['reflection_q'] = expected['reflection_q']
    tt = {}
    for gg in (1, 2):
        for ab in ('00', '01', '10', '11'):
            tt[(gg, ab)] = (xg[gg] if ab[0] == '1' else I) * (zg[gg] if ab[1] == '1' else I) * m[(gg, '00')]
        for k in K:
            rhs = I - (er[gg - 1] + M[gg]) + er[gg - 1] * u[(gg - 1, k)] * er[gg - 1]
            for ab in ('00', '01', '10', '11'):
                rhs = rhs + tt[(gg, ab)] * u[(gg, k)] * tt[(gg, ab)].adj()
            oexp['bridge_block_%d_%s' % (gg, k)] = v[(gg, k)] - rhs
        for ab in ('01', '10', '11'):
            oexp['t_source_%d_%s' % (gg, ab)] = tt[(gg, ab)].adj() * tt[(gg, ab)] - m[(gg, '00')]
            oexp['t_range_%d_%s' % (gg, ab)] = tt[(gg, ab)] * tt[(gg, ab)].adj() - m[(gg, ab)]
    oexp['a_support'] = expected['a_support']
    oexp['R_isometry'] = expected['R_isometry']
    oexp['R_range'] = a * a.adj() + bo(2) * bo(2).adj() + po(2) - er[2]
    oexp['R_B_initial'] = bo(0).adj() * bo(0) - (er[0] - po(0) - qo(0))
    oexp['PQ'] = po(0) * qo(0)
    oexp['R_CP'] = co(0) * po(0) - po(0)
    oexp['R_CQ'] = co(0) * qo(0) - qo(0) * Fr(1, 2)
    oexp['R_CA'] = co(2) * a - a * co(0) * Fr(1, 2)
    oexp['R_CB'] = co(0) * bo(0) - 2 * bo(0) * co(0)
    cmp = {}
    for name, op in opolys.items():
        # find my expected under a plausible name
        cand = [k for k in oexp if k == name]
        if not cand:
            cmp[name] = 'no regenerated counterpart'; continue
        ex = oexp[cand[0]]
        if not ex.d:
            cmp[name] = 'regenerated is identically zero'; continue
        k0 = next(iter(ex.d))
        if k0 not in op.d:
            cmp[name] = 'MISMATCH'; continue
        lam = op.d[k0] / ex.d[k0]
        cmp[name] = ('match, scale %s' % lam) if op == ex * lam else 'MISMATCH'
    report['C']['orig_vs_regenerated'] = cmp
    # which of my regenerated t equations vanish identically
    report['C']['t_identically_zero'] = sorted(k for k, x in oexp.items() if not x.d)

json.dump(report, open(sys.argv[3] if len(sys.argv) > 3 else 'audit_core.json', 'w'), indent=1, default=str)
print(json.dumps({'A_all_match': report['A']['_all_match'], 'A_count': report['A']['_count'],
                  'B_rows': (report['B']['n_rows_file'], report['B']['n_rows_regenerated']),
                  'B_mismatches': report['B']['n_mismatches'], 'B_word_problems': len(probs),
                  'B_scales': report['B']['row_scale_histogram']}, indent=1))
for k, v_ in report['A'].items():
    print('A', k, v_)
if orig_path:
    for k, v_ in sorted(report['C']['orig_vs_regenerated'].items()):
        print('C', k, v_)
    print('C t identically zero:', report['C']['t_identically_zero'])
    print('C orig counts', report['C']['orig_count'], report['C']['orig_n_rows'], report['C']['orig_n_words'])
