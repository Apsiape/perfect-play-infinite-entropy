import sys, json, numpy as np
from fractions import Fraction as Fr
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import field24 as F
from oldmodel import OldModel
W = Path(__file__).resolve().parent.parent
core = json.load(open(W/'unz'/'core.json'))
for N in [int(a) for a in sys.argv[1:]] or [0]:
    M = OldModel(N)
    bad = []
    for rel in core['polynomial_relations']:
        den = 1
        for t in rel['terms']: den = den * Fr(t['coefficient']).denominator // np.gcd(den, Fr(t['coefficient']).denominator)
        acc = np.zeros((8, M.m, M.m), dtype=np.int64)
        for t in rel['terms']:
            c = Fr(t['coefficient']) * den; assert c.denominator == 1
            acc += int(c) * M.word(t['word']).dense()
        if acc.any(): bad.append((rel['name'], den, acc))
    print('N', N, 'm', M.m, 'NW', M.NW, 'nonzero relations:', [b[0] for b in bad])
    for name, den, acc in bad:
        nz = np.argwhere(np.any(acc != 0, axis=0))
        print(name, 'den', den, 'entries', [(int(i), int(j), acc[:, i, j].tolist()) for i, j in nz][:12])
