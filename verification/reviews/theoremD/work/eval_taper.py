"""Refutation-first literal check of the Theorem D taper loss (paper Sec. 9, out7 Sec. 3).

Reuses ONLY the measurement builder `model()` of the cross-checked out5 producer evaluator
(copied verbatim to out5_evaluate_copy.py, SHA-256 24464738...).  The state, the
contraction, the aggregation by question class and the comparison are written here.

Word cutoff M: the model's operators coincide with an infinite canonical realization on every
J-eigenspace with J <= M-1 (integer J) and on every half-integer J <= M-1/2.  Hence any filter
with f(k/2)=0 for k=2J >= 2M is evaluated EXACTLY as the infinite canonical measurements
compressed to its support.  We assert that support condition.

Usage: python -B eval_taper.py M
"""
import os, sys, json, time, zipfile, hashlib, math, random
for t in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[t] = '1'
from pathlib import Path
from fractions import Fraction as Q
from collections import defaultdict
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import out5_evaluate_copy as ev

ARCH = HERE.parents[2] / 'out5' / 'functional-small.zip'
EXPECTED_SHA = 'bb5c0bf32e9a7175bef73d9b156a5c6f3e3a0487462c90a9138e82f809927aaf'


def catalan(n):
    return math.comb(2 * n, n) // (n + 1)


def taper(N):
    def f(k):  # k = 2J
        if k <= 0:
            return Q(1)
        v = 1 - Q(k, 2 * N) ** 2
        return v if v > 0 else Q(0)
    return f


def predicted(fk, M):
    """Analytic claim (3.3)-(3.6): loss = (12/25)(D1+2D2)/Z, with f evaluated on k=2J."""
    Z = Q(0); D1 = Q(0); D2 = Q(0)
    for n in range(0, M + 1):
        a = Q(catalan(n), 4 ** n)
        f0, f1, f2 = fk(2 * n), fk(2 * n - 1), fk(2 * n - 2)
        Z += a * (abs(f0) ** 2 + 8 * abs(f1) ** 2 + 16 * abs(f2) ** 2) / 50
        D1 += a * (f0 - f1) ** 2
        D2 += a * (f1 - f2) ** 2
    return Z, D1, D2, Q(12, 25) * (D1 + 2 * D2) / Z


def main(M, out_json):
    start = time.perf_counter()
    h = hashlib.sha256(ARCH.read_bytes()).hexdigest()
    assert h == EXPECTED_SHA, h
    with zipfile.ZipFile(ARCH) as z:
        meta = json.loads(z.read('format.json'))
        questions = json.loads(z.read('questions.json'))
        p = json.loads(z.read('core.json'))
        raw = z.read('coefficients.bin')
    amap, mats, amp_old, norm_old, ws = ev.model(M, p, questions)
    d = len(ws); n9 = 9 * d
    # Filters to test: tapers of several scales plus random rational filters (arbitrary f(J)).
    rng = random.Random(20261102 + M)
    filters = []
    for N in range(1, M + 1):
        filters.append((f'taper N={N}', taper(N)))
    for t in range(2):
        vals = {k: Q(rng.randint(-7, 9), rng.randint(1, 6)) for k in range(-2, 2 * M)}
        filters.append((f'random#{t} ' + ','.join(f'{k}:{vals[k]}' for k in sorted(vals)),
                        (lambda vals: (lambda k: vals.get(k, Q(0))))(vals)))
    # Classify literal records once.
    recs = np.frombuffer(raw, dtype=[('key', '<u8'), ('num', '<i8')])
    assert len(recs) == meta['nonzero_coefficients'] == 73769
    qa = (recs['key'] >> np.uint64(32)).astype(np.int64)
    qb = (recs['key'] & np.uint64(0xffffffff)).astype(np.int64)
    kindA = [q['kind'] for q in questions['A']]; kindB = [q['kind'] for q in questions['B']]

    def cls(ea, eb):
        A = questions['A'][ea // 16]; B = questions['B'][eb // 16]
        if A['kind'] in ('binary', 'anchor') and B['kind'] in ('binary', 'anchor'):
            return 'core/synchrony (binary x binary)'
        def sign(Qs, q, e):
            ans = q['answers'][e % 16]
            if q['kind'] == 'refined':
                c = Qs[q['coarse']]; s = ans[0]
            elif q['kind'] == 'fms':
                c = q; s = ans
            else:
                return None
            return (c['tag'][0], s) if s in ('+', '-') else (c['tag'][0], 'nonsign')
        sa = sign(questions['A'], A, ea); sb = sign(questions['B'], B, eb)
        if sa and sb and sa[0] == 'T' and sb[0] == 'U' and sa[1] != 'nonsign' and sb[1] != 'nonsign':
            return f'sign/sign T-U ({A["kind"]} x {B["kind"]})'
        return f'other event ({A["kind"]}:{A.get("tag", A["name"])[0] if A["kind"]=="fms" else A["kind"]} x {B["kind"]}:{B.get("tag", B["name"])[0] if B["kind"]=="fms" else B["kind"]})'
    classes = [cls(int(a), int(b)) for a, b in zip(qa, qb)]
    effA = amap['A'][qa]; effB = amap['B'][qb]
    assert np.all(effA >= 0) and np.all(effB >= 0)
    agg = defaultdict(dict); counts = defaultdict(int)
    for idx in range(len(recs)):
        c = classes[idx]; key = (int(effA[idx]), int(effB[idx]))
        coef, nrec = agg[key].get(c, (Q(0), 0))
        agg[key][c] = (coef + Q(int(recs['num'][idx]), meta['denominator']), nrec + 1)
        counts[c] += 1
    print('aggregated effect pairs', len(agg), dict(counts), flush=True)
    results = []
    for label, fk in filters:
        # Support condition for exactness of the truncated model.
        for k in range(2 * M, 2 * M + 200):
            assert fk(k) == 0, (label, k)
        # Integer amplitudes in Q(sqrt2): base * 2^(M-n) * (filter numerator over common denominator).
        den = 1
        for k in range(-2, 2 * M + 1):
            den = math.lcm(den, fk(k).denominator)
        amp = []
        for frame in range(3):
            for c in range(9):
                r = 0 if c == 0 else (1 if c < 5 else 2)
                for w in ws:
                    n = w.count('A'); k = 2 * n - r
                    fv = fk(k) * den
                    assert fv.denominator == 1
                    s = int(fv) * 2 ** (M - n)
                    v = [0] * 8
                    if r == 1:
                        v[1] = s          # sqrt2 * s
                    else:
                        v[0] = s * (1 if r == 0 else 2)
                    amp.append(tuple(v))
        norm = sum(Q(x[0] * x[0] + 2 * x[1] * x[1]) for x in amp)
        assert norm.denominator == 1 and norm > 0
        norm = int(norm)
        cache = {}

        def prob(a, b):
            key = (a, b)
            if key in cache:
                return cache[key]
            A = mats[a]; B = mats[b]; out = [0] * 8
            for (i, j), vij in A.items():
                wji = B.get((j, i))
                if wji is None:
                    continue
                if not any(amp[i]) or not any(amp[j]):
                    continue
                val = ev.fm(ev.fm(vij, wji), ev.fm(amp[i], amp[j]))
                for t, x in enumerate(val):
                    out[t] += x
            res = tuple(Q(x, ev.L * ev.L * norm) for x in out)
            cache[key] = res
            return res
        totals = defaultdict(lambda: [Q(0)] * 8)
        nonzero = defaultdict(int)
        for (a, b), byc in agg.items():
            pr = prob(a, b)
            if not any(pr):
                continue
            for c, (coef, nrec) in byc.items():
                nonzero[c] += nrec
                for t in range(8):
                    totals[c][t] += coef * pr[t]
        grand = [sum((totals[c][t] for c in totals), Q(0)) for t in range(8)]
        Z, D1, D2, loss = predicted(fk, M)
        # Our normalization equals 150*Z*(den^2)*4^M; check the analytic Z independently.
        assert Q(norm, den * den * 4 ** M) == 150 * Z, (norm, Z)
        ok = grand[0] == -loss and not any(grand[1:])
        row = {'filter': label, 'Z': str(Z), 'D1': str(D1), 'D2': str(D2), 'predicted_loss': str(loss),
               'literal_B': str(grand[0]), 'irrational_parts_zero': not any(grand[1:]), 'match': ok,
               'by_class': {c: {'records': counts[c], 'nonzero_records': nonzero[c],
                                'contribution': str(totals[c][0])} for c in sorted(counts)}}
        print(json.dumps(row, indent=1), flush=True)
        results.append(row)
    out = {'M': M, 'words': ws, 'local_dimension': 27 * d, 'archive_sha256': h,
           'evaluator_copy_sha256': hashlib.sha256((HERE / 'out5_evaluate_copy.py').read_bytes()).hexdigest(),
           'results': results, 'elapsed_seconds': time.perf_counter() - start,
           'peak_working_set_bytes': ev.peak_memory()}
    Path(out_json).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
    assert all(r['match'] for r in results)
    print('ALL MATCH', M, out['elapsed_seconds'], out['peak_working_set_bytes'], flush=True)


if __name__ == '__main__':
    M = int(sys.argv[1])
    main(M, HERE / f'eval-taper-M{M}.json')
