"""Parse the serialized game (unz/game.json, unz/questions.json) and reconstruct, from PREDICATE data
only (synchrony, bindings, orthogonality), the structure needed for a canonical strategy:
  * for every row_columns question dr: its slots j=1..d-1 with (jr, sgn) and the old variable bound
    to jr's e-bit (or None if bound to the one-answer anchor), the scalar_pair question bound to jr's
    k-events and the answer partition of that binding; the rowzero mask.
Also builds the literal rule index (my own implementation of the predicate).
"""
import json, collections
from fractions import Fraction as Fr
from pathlib import Path
W = Path(__file__).resolve().parent.parent

def load():
    G = json.load(open(W/'unz'/'game.json'))
    Q = json.load(open(W/'unz'/'questions.json'))
    core = json.load(open(W/'unz'/'core.json'))
    return G, Q, core

def rule_index(G):
    ix = collections.defaultdict(list)
    for q in G['synchrony']: ix[q, q].append(('sync',))
    for q, a, r, b in G['old_forbidden']: ix[q, r].append(('old', a, b))
    for q, mq, r, mr in G['bindings']:
        ix[q, r].append(('bind', int(mq), int(mr))); ix[r, q].append(('bind', int(mr), int(mq)))
    for q, r, j, s in G['orthogonality']:
        ix[q, r].append(('graph', j, s) if j else ('rowzero', s))
    return ix

def my_forbidden(Q, rules, qa, a, qb, b):
    """reviewer's literal predicate for one indexed pair (rules = ix[qa,qb])"""
    if not (0 <= a < len(Q['A'][qa]['answers']) and 0 <= b < len(Q['B'][qb]['answers'])): return True
    for ru in rules:
        t = ru[0]
        if t == 'sync' and a != b: return True
        if t == 'old' and (a, b) == (ru[1], ru[2]): return True
        if t == 'bind' and (((ru[1] >> a) & 1) != ((ru[2] >> b) & 1)): return True
        if t == 'graph':
            j, sgn = ru[1], ru[2]
            e, k = Q['B'][qb]['answers'][b]
            chi = -1 if bin(a & j).count('1') % 2 else 1
            if k < 2 and (1 if k == 0 else -1) == -chi * sgn * (2 * e - 1): return True
        if t == 'rowzero' and a == 0 and (b & ru[1]) == 0: return True
    return False

def mask_members(mask, nans):
    return [i for i in range(nans) if (mask >> i) & 1]

def row_gadgets(G, Q):
    oldQ = G['old_questions']
    A = Q['A']
    binds_by_q = collections.defaultdict(list)
    for q, mq, r, mr in G['bindings']:
        binds_by_q[q].append((int(mq), r, int(mr)))
    orth_by_dr = collections.defaultdict(list)
    rowzero = {}
    for q, r, j, s in G['orthogonality']:
        if j: orth_by_dr[q].append((j, r, s))
        else: rowzero[q] = (r, s)
    gad = {}
    for dr in sorted(set(orth_by_dr) | set(rowzero)):
        assert A[dr]['kind'] == 'row_columns'
        d = len(A[dr]['answers']); assert A[dr]['answers'] == list(range(d))
        slots = {}
        for j, jr, s in orth_by_dr[dr]:
            assert A[jr]['kind'] == 'row_joint' and s in (1, -1) and 0 < j < d and j not in slots
            ans = A[jr]['answers']; assert ans == [[e, k] for e in (0, 1) for k in (0, 1, 2)]
            var = 'UNSET'; kmap = {}; pr = None
            for mq, r, mr in binds_by_q[jr]:
                mem = mask_members(mq, len(ans))
                if sorted(mem) == [i for i, (e, k) in enumerate(ans) if e == 1]:
                    # e-bit binding: r is an old binary question (mask = answer 1) or the anchor
                    if A[r]['kind'] == 'anchor':
                        assert mr == 1; var = None
                    else:
                        assert A[r]['kind'] == 'binary' and mr == 2; var = A[r]['name']
                else:
                    ks = {ans[i][1] for i in mem}; assert len(ks) == 1 and len(mem) == 2
                    k = ks.pop()
                    assert A[r]['kind'] == 'scalar_pair'; pr = pr if pr is not None else r; assert pr == r
                    kmap[k] = mask_members(mr, len(A[r]['answers']))
            assert var != 'UNSET' and sorted(kmap) == [0, 1, 2]
            # the three pr-events must partition pr's answers
            allm = sorted(kmap[0] + kmap[1] + kmap[2]); assert allm == list(range(len(A[pr]['answers'])))
            slots[j] = dict(jr=jr, sgn=s, var=var, pr=pr, kmap=kmap)
        assert sorted(slots) == list(range(1, d))
        gX, mask = rowzero[dr]
        assert A[gX]['kind'] == 'scalar_global' and A[gX]['axis'] == 'X'
        gad[dr] = dict(d=d, slots=slots, rowzero=(gX, mask), row=A[dr].get('row'))
    return gad

def check_rows_against_core(gad, core, Q):
    """sum_j V_j (V_0 = I, V_j = sgn (2E-I) or sgn I) must equal 2 L S_row, using predicate data only"""
    rows = core['projection_linear_relations']
    res = {}
    for dr, g in gad.items():
        r = rows[g['row']]
        al = {k: Fr(v) for k, v in r['terms'].items()}
        import math
        L = math.lcm(*(v.denominator for v in al.values()))
        tot = collections.defaultdict(int); tot['I'] += 1
        for j, s in g['slots'].items():
            if s['var'] is None: tot['I'] += s['sgn']
            else: tot[s['var']] += 2 * s['sgn']; tot['I'] -= s['sgn']
        want = {k: 2 * L * v for k, v in al.items()}
        ok = {k: v for k, v in tot.items() if v} == {k: v for k, v in want.items() if v}
        res[dr] = (ok, L)
    return res
