"""Verifier smoke checks and a noncommuting rational-row example, exactly.
Uses the public verifier to exercise serialized event rules; the main checker
independently checks all mathematical row and binding specifications.
"""
import sys, json, zipfile, importlib.util, collections
from pathlib import Path
from fractions import Fraction as F
OUT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('verifier',OUT/'build_game.py'); g=importlib.util.module_from_spec(sp); sp.loader.exec_module(g)
with zipfile.ZipFile(OUT/'game.zip') as z:
 s=json.loads(z.read('game.json')); qs=json.loads(z.read('questions.json'))
# Supply one pair's rules at a time to keep memory low; exhaust all graph predicates.
count=0
for qa,qb,j,sgn in s['orthogonality']:
 if j:
  rule={ (qa,qb): [('graph',j,sgn)] }
  for t in range(len(qs['A'][qa]['answers'])):
   for b,(e,k) in enumerate(qs['B'][qb]['answers']):
    # Equivalent equation from the two-coordinate graph kernel:
    # scalar reference sign must be the opposite of chi*V's sign.
    expected=(k!=2 and (-1 if k else 1)*(1 if e else -1)*sgn == -((-1)**((t&j).bit_count()%2)))
    assert g.forbidden(s,qs,qa,t,qb,b,rule)==expected
    count+=1
 else:
  rule={(qa,qb):[('rowzero',sgn)]}
  d=next(r['d'] for r in s['rows'] if r['question']==qa)
  for t in (0,1):
   for b in range(256):
    assert g.forbidden(s,qs,qa,t,qb,b,rule)==(t==0 and b%d==0)
    count+=1
# Exercise the complete combined index, including cross-party orientation.
ix=g.make_index(s)
for q,m,r,mr in s['bindings'][::197]:
 m,mr=int(m),int(mr)
 for a in range(len(qs['A'][q]['answers'])):
  for b in range(len(qs['B'][r]['answers'])):
   if ((m>>a)&1)!=((mr>>b)&1):
    assert g.forbidden(s,qs,q,a,r,b,ix)
    assert g.forbidden(s,qs,r,b,q,a,ix)
for q in s['synchrony'][::97]:
 if len(qs['A'][q]['answers'])>1: assert g.forbidden(s,qs,q,0,q,1,ix)
# Noncommuting projections obey 25G-4I+7E-24F=0. Test the row event by
# summing all 64^2 actual 2x2 column-overlap blocks, without squaring a named row.
I=((F(1),F(0)),(F(0),F(1)))
E=((F(1),F(0)),(F(0),F(0)))
H=((F(1,2),F(1,2)),(F(1,2),F(1,2)))
G=((F(9,25),F(12,25)),(F(12,25),F(16,25)))
def add(a,b): return tuple(tuple(a[i][j]+b[i][j] for j in range(2)) for i in range(2))
def scale(a,c): return tuple(tuple(v*c for v in r) for r in a)
def mul(a,b): return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(2)) for j in range(2)) for i in range(2))
for P in (E,H,G): assert mul(P,P)==P
assert mul(E,H)!=mul(H,E)
cols=[I,scale(I,-1)]
for P,c in ((G,25),(E,7),(H,-24)):
 V=add(scale(P,2),scale(I,-1)); cols.extend([scale(V,1 if c>0 else -1)]*abs(c))
cols.extend([I,scale(I,-1)]*((64-len(cols))//2)); assert len(cols)==64
val=sum(sum(mul(a,b)[k][k] for k in range(2)) for a in cols for b in cols)/F(2*64**3)
assert val==0
bad=cols.copy(); bad[2]=scale(bad[2],-1)
badval=sum(sum(mul(a,b)[k][k] for k in range(2)) for a in bad for b in bad)/F(2*64**3)
assert badval==F(1,65536)
# Every valid answer pair not indexed by any rule is accepted.
assert not g.forbidden(s,qs,0,0,0,0,ix)
result={'public_graph_and_spectator_predicate_cases':count,'complete_pair_index_pairs':len(ix),'noncommuting_row_event':str(val),'altered_column_event':str(badval)}
import ctypes
class PMC(ctypes.Structure):
 _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('peak','working','qpeak','q','npeak','n','page','pagepeak')]
pm=PMC(); pm.cb=ctypes.sizeof(pm); ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb); assert pm.peak<4*1024**3; result['peak_working_set_bytes']=int(pm.peak)
(OUT/'verifier-checks.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))

