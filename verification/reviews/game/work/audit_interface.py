"""Independent interface audit of out6/game.zip against out5/functional-small.zip.
Reads only the extracted JSON/binary data (unz/, unz5/). Imports nothing from out5/out6.
Checks:
  (1) the first 5625 game questions are the out5 questions, both parties; core.json identical.
  (2) the out5 coefficient table splits as: FMS/refined-involving records (all negative) +
      binary/anchor records, and the binary/anchor part equals -(sum F_alpha + sum sync) as a
      bilinear form in (1,E_x) x (1,F_y)  [so no row/penalty is lost or mis-selected].
  (3) game.json old_forbidden == exactly the FMS/refined-involving events.
  (4) game.json rows == core rows; slot expansions satisfy (5.1); d power of 4; counts.
  (5) rule index: no rule touches an FMS/refined question together with a new question, etc.
"""
import json, struct, sys, collections, math
from fractions import Fraction as Fr
from pathlib import Path
W = Path(__file__).resolve().parent.parent
G = json.load(open(W/'unz'/'game.json'))
Q = json.load(open(W/'unz'/'questions.json'))
core = json.load(open(W/'unz'/'core.json'))
old = json.load(open(W/'unz5'/'questions.json'))
raw = open(W/'unz5'/'coefficients.bin','rb').read()
out = {}
oldQ = G['old_questions']
assert oldQ == len(old['A']) == len(old['B']) == 5625
for s in 'AB':
    for i in range(oldQ):
        assert Q[s][i] == old[s][i], (s, i)
out['old_questions_identical'] = True
# binary questions have same name/id in both parties
binA = {q['id']: q['name'] for q in old['A'] if q['kind'] == 'binary'}
binB = {q['id']: q['name'] for q in old['B'] if q['kind'] == 'binary'}
assert binA == binB
assert all(old[s][i]['answers'] == [0, 1] for s in 'AB' for i in binA)
assert old['A'][0]['kind'] == 'anchor' == old['B'][0]['kind'] and len(old['A'][0]['answers']) == 1
kinds = collections.Counter(q['kind'] for q in old['A'])
out['old_kinds'] = dict(kinds)
# ---- (2) coefficient table split
recs = list(struct.iter_unpack('<Qq', raw))
keys = [k for k, _ in recs]
assert keys == sorted(keys) and len(set(keys)) == len(keys)
fms_kinds = ('fms', 'refined')
ext = set(); bil = collections.defaultdict(Fr)
def vecA(q, a, P):
    k = P[q]['kind']
    if k == 'anchor': return {'1': 1}
    assert k == 'binary'
    nm = P[q]['name']
    return {nm: 1} if a == 1 else {'1': 1, nm: -1}
nneg = 0
for key, num in recs:
    ea, eb = key >> 32, key & 0xffffffff
    qa, a = divmod(ea, 16); qb, b = divmod(eb, 16)
    assert a < len(old['A'][qa]['answers']) and b < len(old['B'][qb]['answers'])
    if old['A'][qa]['kind'] in fms_kinds or old['B'][qb]['kind'] in fms_kinds:
        assert num < 0; nneg += 1
        ext.add((qa, a, qb, b))
    else:
        c = Fr(num, 2048)
        va = vecA(qa, a, old['A']); vb = vecA(qb, b, old['B'])
        for x, cx in va.items():
            for y, cy in vb.items():
                bil[x, y] += c*cx*cy
out['fms_records'] = nneg
out['fms_distinct_events'] = len(ext)
# expected: -(sum_rows F_alpha + sum_binary sync)
exp = collections.defaultdict(Fr)
rows = core['projection_linear_relations']
for r in rows:
    al = {k: Fr(v) for k, v in r['terms'].items()}
    a0 = al.pop('I', Fr(0))
    vec = {'1': a0}; vec.update(al)
    for x, cx in vec.items():
        for y, cy in vec.items():
            exp[x, y] -= cx*cy
    k = 1 + Fr(1, 2)*sum(v*v for v in al.values())
    for nm in al:
        exp[nm, '1'] -= k; exp['1', nm] -= k; exp[nm, nm] += 2*k
for nm in binA.values():
    exp[nm, '1'] -= 1; exp['1', nm] -= 1; exp[nm, nm] += 2
diff = {k: v for k, v in ((k, bil.get(k, 0) - exp.get(k, 0)) for k in set(bil) | set(exp)) if v != 0}
out['binary_anchor_bilinear_form_equals_minus_Falpha_minus_sync'] = (len(diff) == 0)
out['bilinear_entries'] = sum(1 for v in exp.values() if v != 0)
if diff: out['diff_sample'] = [(str(k), str(v)) for k, v in list(diff.items())[:10]]
# ---- (3) game old_forbidden == ext
gof = set(tuple(e) for e in G['old_forbidden'])
out['game_old_forbidden'] = len(G['old_forbidden'])
out['game_old_forbidden_equals_table_fms_events'] = (gof == ext and len(gof) == len(G['old_forbidden']))
# ---- (4) rows and slot expansions
assert len(G['rows']) == len(rows)
okrows = 0; dims = collections.Counter(); Ls = collections.Counter()
for plan, r in zip(G['rows'], rows):
    assert plan['name'] == r['name']
    al = {k: Fr(v) for k, v in r['terms'].items()}
    L = math.lcm(*(v.denominator for v in al.values()))
    assert plan['den'] == L
    d = plan['d']; slots = plan['slots']
    assert len(slots) == d and d in (4, 16, 64, 256)
    assert slots[0] == [None, 1]
    # sum V_j = c*I + sum_i c_i*(2e_i - I) must equal 2 L S
    tot = collections.defaultdict(int)
    for nm, sg in slots:
        assert sg in (1, -1)
        if nm is None: tot['I'] += sg
        else:
            tot[nm] += 2*sg; tot['I'] -= sg
    want = {k: 2*L*v for k, v in al.items()}
    assert {k: v for k, v in tot.items() if v} == {k: v for k, v in want.items() if v}, r['name']
    # minimality of d among powers of 4 (not needed for soundness, recorded)
    dims[d] += 1; Ls[L] += 1; okrows += 1
out['rows_slot_identity_5_1_checked'] = okrows
out['row_dims'] = dict(dims); out['row_denominators'] = dict(Ls)
iso = [p for p in G['rows'] if p['name'] == 'R_isometry'][0]
out['R_isometry'] = {'d': iso['d'], 'den': iso['den'], 'slots': iso['slots'][:8]}
# ---- (5) rule index
NQ = len(Q['A'])
kindA = [q['kind'] for q in Q['A']]; kindB = [q['kind'] for q in Q['B']]
assert all(Q['A'][i]['kind'] == Q['B'][i]['kind'] for i in range(oldQ, NQ))
assert all(Q['A'][i] == Q['B'][i] for i in range(oldQ, NQ))
ix = collections.defaultdict(list)
for q in G['synchrony']: ix[q, q].append(('sync',))
for q, a, r, b in G['old_forbidden']: ix[q, r].append(('old', a, b))
for q, mq, r, mr in G['bindings']:
    ix[q, r].append(('bind', int(mq), int(mr))); ix[r, q].append(('bind', int(mr), int(mq)))
for q, r, j, s in G['orthogonality']:
    ix[q, r].append(('graph', j, s) if j else ('rowzero', s))
out['indexed_pairs'] = len(ix)
syncset = set(G['synchrony'])
assert len(syncset) == len(G['synchrony'])
expect_sync = set(binA) | set(range(oldQ, NQ))
out['sync_is_all_old_binary_plus_all_new'] = (syncset == expect_sync)
bad = []
pairkinds = collections.Counter()
for (q, r), rules in ix.items():
    kq, kr = kindA[q], kindB[r]
    newq, newr = q >= oldQ, r >= oldQ
    if (newq and kr in fms_kinds) or (newr and kq in fms_kinds): bad.append((q, r))
    for ru in rules: pairkinds[(ru[0], kq, kr)] += 1
out['rules_linking_fms_to_new'] = len(bad)
out['rule_kind_counts'] = {f'{a}|{b}|{c}': v for (a, b, c), v in sorted(pairkinds.items())}
# all masks within answer range
for q, mq, r, mr in G['bindings']:
    mq, mr = int(mq), int(mr)
    assert 0 < mq < (1 << len(Q['A'][q]['answers'])) and 0 < mr < (1 << len(Q['B'][r]['answers']))
json.dump(out, open(W/'work'/'audit_interface.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
