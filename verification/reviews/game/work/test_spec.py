import sys, json, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gamespec as GS
G, Q, core = GS.load()
gad = GS.row_gadgets(G, Q)
print('row gadgets', len(gad), collections.Counter(g['d'] for g in gad.values()))
res = GS.check_rows_against_core(gad, core, Q)
print('rows encoded by predicate data:', sum(ok for ok, L in res.values()), 'of', len(res))
rows_seen = sorted(g['row'] for g in gad.values()); assert rows_seen == list(range(len(core['projection_linear_relations'])))
# rowzero masks
print(collections.Counter((g['d'], g['rowzero'][1]) for g in gad.values()))
# scalar pair metadata consistent with slot j
bad = 0
for dr, g in gad.items():
    for j, s in g['slots'].items():
        pq = Q['A'][s['pr']]
        if not (pq['j'] == j and 2 ** pq['n'] == g['d']): bad += 1
        if s['kmap'][0] != [0] or s['kmap'][1] != [1]: bad += 1
print('slot/pair-context mismatches', bad)
