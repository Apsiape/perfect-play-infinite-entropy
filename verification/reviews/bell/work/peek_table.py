import json, sys, struct, collections, hashlib
from fractions import Fraction
d = sys.argv[1]
q = json.load(open(d + '/questions.json'))
raw = open(d + '/coefficients.bin', 'rb').read()
print('sha', hashlib.sha256(raw).hexdigest(), len(raw) // 16)
recs = list(struct.iter_unpack('<Qq', raw))
A, B = q['A'], q['B']
def kind(qs, qid):
    return qs[qid]['kind']
cnt = collections.Counter(); tot = collections.Counter()
for key, num in recs:
    ea, eb = key >> 32, key & 0xffffffff
    xa, aa = divmod(ea, 16); xb, ab = divmod(eb, 16)
    k = (kind(A, xa), kind(B, xb))
    cnt[k] += 1; tot[k] += Fraction(num, 2048)
for k in sorted(cnt): print(k, cnt[k], tot[k])
# keys sorted?
keys = [r[0] for r in recs]
print('strictly increasing', all(keys[i] < keys[i+1] for i in range(len(keys)-1)))
print('zero coeffs', sum(1 for r in recs if r[1] == 0))
