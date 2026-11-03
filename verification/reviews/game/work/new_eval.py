"""Reviewer's exact evaluation of the full game for the cutoff-N canonical strategy
(old canonical strategy of old_eval.py) (x) Phi_256 with the new measurements of Sections 4-5, as
reconstructed by the reviewer. usage: python -I new_eval.py N   (needs work/old-N{N}.json)

Scalar operators on C^256 are exact elements of the Pauli group algebra:
  {(xbits, zbits): Fraction}  meaning  sum c X^x Z^z,  X^x Z^z = prod X_i^{x_i} prod Z_i^{z_i}.
Old factors are labels I, E, nE=(I-E) of ONE old projection E per question; old_eval.py verified
exactly that every old projection is a Hermitian idempotent commuting with h, so
  Tr(K O K O') = Tr(h O O'):  c(E,E)=c(E,I)=t, c(nE,nE)=c(nE,I)=1-t, c(E,nE)=0, c(I,I)=1, t=Tr(hE).
For a pair of Alice effect O(x)S and Bob effect (Alice form) O'(x)S':
  p = Tr(K_old O K_old O') * Tr(S S') / 256.
Row-column questions D_t = sum_{i,i'} chi_t(i)chi_t(i')/d |i><i'|(x)I_sp (x) V_i V_i' are handled by the
partial-trace formula  Tr(K D_t K (O (x) R)) = sum_{i,i'} chi chi/d Tr(h V_i V_i' O) T_R[i',i]/256,
T_R[i',i] = sum_s <i',s|R|i,s>.
"""
import sys, json, time, collections, math, itertools
from fractions import Fraction as Fr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gamespec as GS

N = int(sys.argv[1])
t0 = time.time()
G, Q, core = GS.load()
OLD = json.load(open(Path(__file__).parent / f'old-N{N}.json'))
trhE = {k: Fr(v) for k, v in OLD['trhE'].items()}
oldQ = G['old_questions']; NQ = len(Q['A'])
A = Q['A']
NQB = 8; DIM = 256

# ---------------- Pauli group algebra
def pc(b): return bin(b).count('1')
def pmul(P, R):
    out = collections.defaultdict(Fr)
    for (a, b), c in P.items():
        for (cc, d), e in R.items():
            s = -1 if pc(b & cc) % 2 else 1
            out[a ^ cc, b ^ d] += s * c * e
    return {k: v for k, v in out.items() if v}
def padd(*Ps):
    out = collections.defaultdict(Fr)
    for P in Ps:
        for k, v in P.items(): out[k] += v
    return {k: v for k, v in out.items() if v}
def pscale(P, s): return {k: v * s for k, v in P.items() if v * s}
def ptr2(P, R):
    """Tr(P R)/256"""
    if len(P) > len(R): P, R = R, P
    s = Fr(0)
    for (a, b), c in P.items():
        e = R.get((a, b))
        if e: s += c * e * (-1 if pc(a & b) % 2 else 1)
    return s
IDP = {(0, 0): Fr(1)}
def Xs(i): return {(1 << i, 0): Fr(1)}
def Zs(i): return {(0, 1 << i): Fr(1)}
def proj_pm(Mop, s):  # (I + s M)/2
    return padd(pscale(IDP, Fr(1, 2)), pscale(Mop, Fr(s, 2)))
def single_string(P):
    assert len(P) == 1; (k, v), = P.items(); return k, v
def commute(P, R): return pmul(P, R) == pmul(R, P)
def is_reflection(P): return pmul(P, P) == IDP and all(True for _ in P)

checks = collections.Counter()
# ---------------- canonical scalar operators
axes = G['axes']
cellop = {}
for i, ax in enumerate(axes):
    cellop[ax['Z']] = Zs(i); cellop[ax['X']] = Xs(i)
cells = {}
for q in A[oldQ:]:
    if q['kind'] == 'scalar_cell': cells[q['g'], q['i'], q['j']] = q['id']
for g in range(4):
    cc = lambda i, j: cells[g, i, j]
    # axes must be the four corner cells exactly as assumed
    assert cellop[cc(0, 0)] == Zs(2*g) and cellop[cc(1, 1)] == Xs(2*g) and cellop[cc(0, 1)] == Zs(2*g+1) and cellop[cc(1, 0)] == Xs(2*g+1)
    cellop[cc(0, 2)] = pmul(cellop[cc(0, 0)], cellop[cc(0, 1)])
    cellop[cc(1, 2)] = pmul(cellop[cc(1, 0)], cellop[cc(1, 1)])
    cellop[cc(2, 0)] = pmul(cellop[cc(0, 0)], cellop[cc(1, 0)])
    cellop[cc(2, 1)] = pmul(cellop[cc(0, 1)], cellop[cc(1, 1)])
    cellop[cc(2, 2)] = pmul(cellop[cc(2, 0)], cellop[cc(2, 1)])
    for i in range(3):
        for j in range(3):
            assert pmul(cellop[cc(i, j)], cellop[cc(i, j)]) == IDP and all(v.denominator == 1 for v in cellop[cc(i, j)].values())
    for i in range(3):  # rows: commuting, product +I
        r = [cellop[cc(i, j)] for j in range(3)]
        assert all(commute(r[a], r[b]) for a in range(3) for b in range(3))
        assert pmul(pmul(r[0], r[1]), r[2]) == IDP
    for j in range(3):
        c = [cellop[cc(i, j)] for i in range(3)]
        assert all(commute(c[a], c[b]) for a in range(3) for b in range(3))
        assert pmul(pmul(c[0], c[1]), c[2]) == (IDP if j < 2 else pscale(IDP, -1))
checks['magic_square_identities_exact'] = 4

# ---------------- canonical fine effects / coarse events
gad = GS.row_gadgets(G, Q)
jr_info = {}
for dr, g in gad.items():
    for j, s in g['slots'].items(): jr_info[s['jr']] = dict(s, dr=dr, j=j, d=g['d'])
def walsh_coarse(Qops, members, nbits):
    """sum_{z in members} prod_m (I + (-1)^{z_m} Q_m)/2 for commuting reflections Q_m (Pauli strings)"""
    for a in range(nbits):
        for b in range(a + 1, nbits): assert commute(Qops[a], Qops[b])
        assert pmul(Qops[a], Qops[a]) == IDP
    size = 1 << nbits
    ind = [0] * size
    for z in members: ind[z] = 1
    # Walsh: c_S = sum_{z in T} (-1)^{|S & z|}
    c = ind[:]
    h = 1
    while h < size:
        for i in range(0, size, 2 * h):
            for k in range(i, i + h):
                x, y = c[k], c[k + h]; c[k], c[k + h] = x + y, x - y
        h *= 2
    out = collections.defaultdict(Fr)
    # products Q_S
    QS = {0: IDP}
    for S in range(1, size):
        low = S & -S; m = low.bit_length() - 1
        QS[S] = pmul(QS[S ^ low], Qops[m])
    for S in range(size):
        if c[S]:
            for k, v in QS[S].items(): out[k] += Fr(c[S], size) * v
    return {k: v for k, v in out.items() if v}
pair_ops = {}
def pair_Qops(qid):
    if qid not in pair_ops:
        q = A[qid]
        if q['kind'] == 'scalar_global':
            ops = [Xs(i) if q['axis'] == 'X' else Zs(i) for i in range(q['n'])]
        else:
            j = q['j']; ops = [{(j, 0): Fr(1)}] + [{(0, l): Fr(1)} for l in q['kernel']]
            assert len(ops) == q['n']
        pair_ops[qid] = ops
    return pair_ops[qid]
coarse_cache = {}
def oldvar(qid):
    q = A[qid]
    if q['kind'] == 'binary': return q['name']
    if q['kind'] == 'commute_old': return A[q['p']]['name']
    if q['kind'] == 'row_joint': return jr_info[qid]['var']
    return None
def fine(qid, a):
    """canonical fine effect of answer a: dict label->Pauli element; label in I,E,nE"""
    q = A[qid]; k = q['kind']; ans = q['answers'][a]
    if k == 'anchor': return {'I': IDP}
    if k == 'binary': return {'E' if ans == 1 else 'nE': IDP}
    if k == 'scalar_cell': return {'I': proj_pm(cellop[qid], 1 if ans == 1 else -1)}
    if k == 'scalar_ms':
        g, o, i = q['g'], q['orientation'], q['index']
        cl = [cells[g, i, j] for j in range(3)] if o == 'R' else [cells[g, j, i] for j in range(3)]
        P = IDP
        for cid, s in zip(cl, ans): P = pmul(P, proj_pm(cellop[cid], s))
        return {'I': P}
    if k == 'cross_axes':
        return {'I': pmul(proj_pm(cellop[q['p']], 1 if ans[0] else -1), proj_pm(cellop[q['q']], 1 if ans[1] else -1))}
    if k == 'commute_old':
        return {'E' if ans[0] == 1 else 'nE': proj_pm(cellop[q['q']], 1 if ans[1] else -1)}
    if k == 'row_joint':
        info = jr_info[qid]; e, kk = ans
        R = coarse(info['pr'], sum(1 << z for z in info['kmap'][kk]))['I']
        if info['var'] is None: return {'I': R} if e == 1 else {}
        return {'E' if e == 1 else 'nE': R}
    raise KeyError(k)
def skey(qid):
    q = A[qid]; k = q['kind']
    if k in ('anchor', 'binary'): return (k,)
    if k == 'commute_old': return ('co', q['q'])
    if k == 'row_joint':
        info = jr_info[qid]; return ('rj', info['pr'], info['var'] is None, tuple(tuple(info['kmap'][x]) for x in (0, 1, 2)))
    return (k, qid)
def coarse(qid, mask):
    key = (skey(qid), mask)
    if key in coarse_cache: return coarse_cache[key]
    q = A[qid]; nans = len(q['answers'])
    mem = [a for a in range(nans) if (mask >> a) & 1]
    if q['kind'] in ('scalar_global', 'scalar_pair'):
        res = {'I': walsh_coarse(pair_Qops(qid), mem, q['n'])} if mem else {}
        # answer z <-> bits: global: bit i negative X_i/Z_i; pair: bit0 X^j negative, bit k+1 Z^{l_k} negative
    else:
        acc = collections.defaultdict(dict)
        for a in mem:
            for lab, P in fine(qid, a).items(): acc[lab] = padd(acc[lab], P) if lab in acc else P
        res = {lab: P for lab, P in acc.items() if P}
        # merge E and nE parts carrying the same scalar operator into I
        if 'E' in res and 'nE' in res and res['E'] == res['nE']:
            P = res.pop('E'); res.pop('nE'); res['I'] = padd(res.get('I', {}), P)
    coarse_cache[key] = res
    return res
def cfac(l1, l2, t):
    tab = {('I', 'I'): 1, ('E', 'I'): t, ('I', 'E'): t, ('nE', 'I'): 1 - t, ('I', 'nE'): 1 - t,
           ('E', 'E'): t, ('nE', 'nE'): 1 - t, ('E', 'nE'): 0, ('nE', 'E'): 0}
    return tab[l1, l2]
def pair_prob(cq, vq, cr, vr):
    """sum over labels of Tr(K O K O') Tr(S S')/256 for coarse dicts"""
    vs = {v for v in (vq, vr) if v is not None}
    if len(vs) > 1:
        # two different old variables: only allowed if one side carries only 'I'
        assert set(cq) <= {'I'} or set(cr) <= {'I'}, (vq, vr)
    v = (vs.pop() if vs else None)
    t = trhE[v] if v is not None else None
    s = Fr(0)
    for l1, P in cq.items():
        for l2, R in cr.items():
            if l1 == 'I' and l2 == 'I': f = 1
            else: f = cfac(l1, l2, t)
            if f: s += f * ptr2(P, R)
    return s
nans = lambda qid: len(A[qid]['answers'])
full = lambda qid: (1 << nans(qid)) - 1
# ---------------- POVM checks for every distinct new question structure
povm_checked = set()
def povm_check(qid):
    q = A[qid]
    if q['kind'] in ('row_columns',): return
    key = (q['kind'], q.get('g'), q.get('orientation'), q.get('index'), q.get('i'), q.get('j'), q.get('p') if q['kind'] == 'cross_axes' else None,
           q.get('q'), q.get('axis'), q.get('n'), tuple(q.get('kernel', ())), jr_info.get(qid, {}).get('pr'), jr_info.get(qid, {}).get('var') is None)
    if key in povm_checked: return
    povm_checked.add(key)
    tot = coarse(qid, full(qid))
    assert tot == {'I': IDP}, (q['kind'], tot.keys())
    if q['kind'] not in ('scalar_global', 'scalar_pair'):
        for a in range(nans(qid)):
            for lab, P in fine(qid, a).items(): assert pmul(P, P) == P
    checks['povm_structures_checked'] += 1
for qid in range(oldQ, NQ): povm_check(qid)
print('povm checks', dict(checks), time.time() - t0, flush=True)

# ---------------- partial traces for D_t formulas
def ptrace_low(P, nlow):
    """T[i',i] = sum_s <i',s|P|i,s>  over spectator qubits nlow..7, returns dict (i',i)->Fraction"""
    lowmask = (1 << nlow) - 1
    T = collections.defaultdict(Fr)
    for (a, b), c in P.items():
        if (a & ~lowmask) or (b & ~lowmask): continue
        mult = 1 << (NQB - nlow)
        for i in range(1 << nlow):
            # X^a Z^b |i> = (-1)^{b.i} |i ^ a>
            T[i ^ a, i] += c * mult * (-1 if pc(b & i) % 2 else 1)
    return {k: v for k, v in T.items() if v}
def chi(t, i): return -1 if pc(t & i) % 2 else 1

# ---------------- evaluate every indexed pair touching a new question
ix = GS.rule_index(G)
loss_by_kind = collections.Counter(); nonzero_pairs = []
nrules = collections.Counter(); structural = collections.Counter()
graph_cache = {}; rowzero_T = {}; bind_cache = {}
for (q, r), rules in ix.items():
    if q < oldQ and r < oldQ: continue
    ptot = Fr(0)
    for ru in rules:
        t = ru[0]; nrules[t] += 1
        if t == 'sync':
            if A[q]['kind'] == 'row_columns':
                structural['row_columns_sync'] += 1; continue   # PVM commuting with h (justified separately)
            # 1 - sum_a Tr(K A_a K A_a)
            v = oldvar(q); s = Fr(0)
            key = ('sync', q if A[q]['kind'] not in ('commute_old', 'row_joint') else (A[q]['kind'], A[q].get('q'), jr_info.get(q, {}).get('pr'), v is None), v)
            if key in graph_cache: s = graph_cache[key]
            else:
                if A[q]['kind'] in ('scalar_global', 'scalar_pair'):
                    # P_z = 2^-n sum_S (-1)^{|S&z|} Q_S with Q_S distinct Pauli strings (asserted), so
                    # Tr(P_z P_z)/256 = 4^-n sum_S sign(Q_S^2) for every z
                    ops = pair_Qops(q); nb = len(ops); QS = {0: IDP}
                    for S in range(1, 1 << nb):
                        low = S & -S; QS[S] = pmul(QS[S ^ low], ops[low.bit_length() - 1])
                    keys = [single_string(P)[0] for P in QS.values()]; assert len(set(keys)) == len(keys)
                    per = sum(ptr2(P, P) for P in QS.values()) / Fr(4**nb)
                    s = nans(q) * per
                    # literal spot check on two answers
                    for a in (0, nans(q) - 1):
                        Pa = coarse(q, 1 << a)['I']; assert ptr2(Pa, Pa) == per
                else:
                    for a in range(nans(q)):
                        Fa = fine(q, a); s += pair_prob(Fa, v, Fa, v)
                graph_cache[key] = s
            p = 1 - s
        elif t == 'bind':
            mq, mr = ru[1], ru[2]
            vq, vr = oldvar(q), oldvar(r)
            tv = trhE[vq] if vq is not None else (trhE[vr] if vr is not None else None)
            bkey = (skey(q), mq, skey(r), mr, vq is None, vr is None, tv)
            if bkey in bind_cache: p = bind_cache[bkey]
            else:
                cq, cqn = coarse(q, mq), coarse(q, full(q) ^ mq)
                cr, crn = coarse(r, mr), coarse(r, full(r) ^ mr)
                p = pair_prob(cq, vq, crn, vr) + pair_prob(cqn, vq, cr, vr)
                bind_cache[bkey] = p
            if vq is not None and vr is not None: assert vq == vr
        elif t == 'graph':
            j, sgn = ru[1], ru[2]
            dr, jr = q, r
            info = jr_info[jr]; assert info['dr'] == dr and info['j'] == j and info['sgn'] == sgn
            d = gad[dr]['d']; nlow = d.bit_length() - 1
            v = info['var']; tE = trhE[v] if v is not None else Fr(1)
            gkey = (d, j, sgn, v is None, tE)
            if gkey in graph_cache: p = graph_cache[gkey]
            else:
                Tk = {k: ptrace_low(coarse(info['pr'], sum(1 << z for z in info['kmap'][k]))['I'], nlow) for k in (0, 1)}
                # old traces Tr(h V_i V_i' E^e), V_0 = I, V_j = sgn(2E-I) (or sgn I)
                def oldtr(i, i2, e):
                    for x in (i, i2): assert x in (0, j)
                    te = tE if e == 1 else 1 - tE
                    if v is None and e == 0: return Fr(0)
                    nV = (i == j) + (i2 == j)
                    if nV % 2 == 0: return te
                    # Tr(h V_j E^e) = sgn * Tr(h (2E-I) E^e) = sgn * (+t  or -(1-t))
                    return sgn * (te if e == 1 else -te)
                p = Fr(0)
                for tt in range(d):
                    for (e, kk) in A[jr]['answers']:
                        if kk < 2 and (1 if kk == 0 else -1) == -chi(tt, j) * sgn * (2 * e - 1):
                            for (i2, i), val in Tk[kk].items():
                                p += Fr(chi(tt, i) * chi(tt, i2), d) * oldtr(i, i2, e) * val / DIM
                graph_cache[gkey] = p
        elif t == 'rowzero':
            mask = ru[1]; dr = q; d = gad[dr]['d']; nlow = d.bit_length() - 1
            assert r == gad[dr]['rowzero'][0] and mask == gad[dr]['rowzero'][1]
            if d not in rowzero_T:
                mem = [b for b in range(nans(r)) if (b & mask) == 0]
                TR = ptrace_low(coarse(r, sum(1 << b for b in mem))['I'], nlow)
                vals = set(TR.values()); assert len(TR) == d * d and len(vals) == 1
                rowzero_T[d] = vals.pop()
            # p = sum_{i,i'} (1/d) Tr(h V_i V_i') T = (T/d) Tr(h Sigma^2), T constant
            row = gad[dr]['row']
            sig = OLD['row_gadgets_with_nonzero_Sigma'].get(str(row))
            trS = Fr(sig[2]) if sig else Fr(0)
            p = Fr(1, d) * rowzero_T[d] * trS / DIM
        else:
            raise ValueError(t)
        assert p >= 0
        if p: loss_by_kind[t] += p
        ptot += p
    if ptot:
        nonzero_pairs.append((q, r, [ru[0] for ru in rules], str(ptot)))
print('pairs done', time.time() - t0, flush=True)
out = {'N': N, 'rules_evaluated': dict(nrules), 'structural_zeros': dict(structural), 'checks': dict(checks),
       'rowzero_T_constant': {str(k): str(v) for k, v in rowzero_T.items()},
       'nonzero_pairs_new': [(q, r, A[q]['kind'], A[r]['kind'], rk, p) for q, r, rk, p in nonzero_pairs]}
# literal union for nonzero pairs (only rule rowzero expected); confirm single rule on the pair
for q, r, rk, p in nonzero_pairs:
    assert rk == ['rowzero'], rk
# old-old pairs: from old_eval (exact): FMS events and binary sync
old_loss = Fr(0) if OLD['old_forbidden_events_nonzero'] == 0 else None
assert OLD['binary_projection_checks']['idempotent'] == 5592 and OLD['binary_projection_checks']['commutes_h'] == 5592 and 'sync_nonzero' not in OLD['binary_projection_checks']
total_event = sum((Fr(p) for _, _, _, p in nonzero_pairs), Fr(0)) + old_loss
Qn = NQ
out['sum_rejecting_event_probabilities'] = str(total_event)
out['uniform_game_loss'] = str(total_event / Qn**2)
tN = Fr(OLD['t_N'])
out['predicted_t_N_over_9600'] = str(tN / 9600)
out['matches_7_2'] = (total_event == tN / 9600)
out['seconds'] = time.time() - t0
json.dump(out, open(Path(__file__).parent / f'new-N{N}.json', 'w'), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
