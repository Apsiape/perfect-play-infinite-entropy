"""Independent exact evaluator: reads the exported literal coefficient table.
No producer import, no F_alpha/template scoring. Matrices use Q(i,sqrt2,sqrt3).
Deduplicate identical canonical effects, then contract every table coefficient.
"""
import os
for t in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[t]='1'
import json,zipfile,hashlib,itertools,time,sys,ctypes,math,gc
from pathlib import Path
from fractions import Fraction as Q
from functools import lru_cache
from collections import defaultdict
import numpy as np
OUT=Path(__file__).resolve().parent
L=240; ZERO=(0,)*8
# Basis indexed by bit masks: sqrt2^a sqrt3^b i^c, a,b,c in {0,1}.
FACT=[[2**bool(a&b&1)*3**bool(a&b&2)*(-1)**bool(a&b&4) for b in range(8)] for a in range(8)]
@lru_cache(8192)
def fm(a,b):
 c=[0]*8
 for i,x in enumerate(a):
  if x:
   for j,y in enumerate(b):
    if y: c[i^j]+=x*y*FACT[i][j]
 return tuple(c)
def sc(v,c):
 c=Q(c); num,den=c.numerator,c.denominator
 if den==1: return tuple(x*num for x in v)
 assert all((x*num)%den==0 for x in v)
 return tuple(x*num//den for x in v)
def va(a,b): return tuple(x+y for x,y in zip(a,b))
def conj(a): return tuple(-x if i&4 else x for i,x in enumerate(a))
def real(c): return (c,0,0,0,0,0,0,0)
def clean(m): return {k:v for k,v in m.items() if any(v)}
def add(*ms):
 out={}
 for m,c in ms:
  for k,v in m.items(): out[k]=va(out.get(k,ZERO),sc(v,c))
 return clean(out)
def star(m): return {(j,i):conj(v) for (i,j),v in m.items()}
def prod(a,b):
 right=defaultdict(list)
 for (j,k),v in b.items(): right[j].append((k,v))
 out={}
 for (i,j),v in a.items():
  for k,w in right[j]:
   z=fm(v,w)
   key=(i,k); out[key]=va(out.get(key,ZERO),z)
 for key,v in out.items():
  assert all(x%L==0 for x in v)
  out[key]=tuple(x//L for x in v)
 return clean(out)
def diag(indices): return {(i,i):real(L) for i in indices}
# Integer phase vectors scaled by two, exp(i*pi*k/6).
PH=[(2,0,0,0,0,0,0,0),(0,0,1,0,1,0,0,0),(1,0,0,0,0,0,1,0),(0,0,0,0,2,0,0,0),(-1,0,0,0,0,0,1,0),(0,0,-1,0,1,0,0,0),(-2,0,0,0,0,0,0,0),(0,0,-1,0,-1,0,0,0),(-1,0,0,0,0,0,-1,0),(0,0,0,0,-2,0,0,0),(1,0,0,0,0,0,-1,0),(0,0,1,0,-1,0,0,0)]
def compose(a,b): return tuple(a[0][j] for j in b[0]),tuple((z+a[1][j])%12 for j,z in zip(b[0],b[1]))
def inverse(u):
 p=[0]*len(u[0]); z=p.copy()
 for j,r in enumerate(u[0]): p[r]=j; z[r]=-u[1][j]%12
 return tuple(p),tuple(z)

def model(N,p,questions):
 # Enumerate legal words independently by complete binary strings, N<=2.
 ws=[]
 for n in range(2*N+1):
  for t in itertools.product('AB',repeat=n):
   ht=0; valid=True
   for c in t:
    if c=='A': ht+=1
    elif ht<2: valid=False; break
    else: ht-=1
   if valid and t.count('A')<=N: ws.append(''.join(t))
 ws=sorted(ws,key=lambda w:(w.count('A'),len(w),w)); ix={w:i for i,w in enumerate(ws)}; d=len(ws); n=9*d; dim=3*n
 identity=(tuple(range(n)),(0,)*n); atoms={}; weights=(1,2,2,2,2,4,4,4,4); eig=[Q(weights[c],4**w.count('A')) for c in range(9) for w in ws]
 def monodiag(ph): return tuple(range(n)),tuple(ph)
 for name in p['parameters']:
  pp=list(range(n)); ph=[6]*n
  if name=='root': ph[:d]=[0]*d
  elif name.startswith('m'):
   corner=(1 if name[1]=='1' else 5)+2*int(name[3])+int(name[4]); ph[corner*d:(corner+1)*d]=[0]*d
  else:
   g=int(name[-1]); start=1 if g==1 else 5; axis=name.startswith('X1')
   for a,b in itertools.product((0,1),repeat=2):
    co=start+2*a+b; cc=start+2*(1-a)+b if axis else start+2*a+1-b
    for j in range(d): pp[co*d+j]=cc*d+j; ph[co*d+j]=0
  atoms['s_'+name]=(tuple(pp),tuple(ph))
 def complete(partial,scope):
  pp=list(range(n)); zp=[0]*n; zm=zp.copy()
  for val in sorted({eig[j] for j in scope}):
   cols=sorted(j for j in scope if eig[j]==val); used={partial[j] for j in cols if j in partial}
   holes=[j for j in cols if j not in partial]; free=[j for j in cols if j not in used]
   for j in cols:
    if j in partial: pp[j]=partial[j]; assert eig[j]==eig[pp[j]]
   for j,r in zip(holes,free): pp[j]=r; zp[j]=3; zm[j]=9
  return (tuple(pp),tuple(zp)),(tuple(pp),tuple(zm))
 for r,co in enumerate((0,1,5)):
  partial={co*d+j:co*d+ix[w+'B'] for j,w in enumerate(ws) if w+'B' in ix}
  atoms[f'u{r}_bplus'],atoms[f'u{r}_bminus']=complete(partial,set(range(co*d,(co+1)*d)))
  for k in ('p','q','c'):
   ph=[0]*n
   for j,w in enumerate(ws):
    ht=w.count('A')-w.count('B'); ph[co*d+j]=(0 if not w else 6) if k=='p' else ((0 if ht==1 else 6) if k=='q' else (0,3,4)[ht])
   atoms[f'u{r}_{k}']=monodiag(ph)
 partial={j:5*d+ix[w+'A'] for j,w in enumerate(ws) if w+'A' in ix}
 atoms['a_plus'],atoms['a_minus']=complete(partial,set(range(n)))
 for g in (1,2):
  start=1 if g==1 else 5; source=0 if g==1 else 1
  for k in ('bplus','bminus','p','q','c'):
   pp=list(range(n)); ph=[0]*n
   for co,rr in [(source,g-1)]+[(start+h,g) for h in range(4)]:
    ref=(0,1,5)[rr]; u=atoms[f'u{rr}_{k}']
    for j in range(d): pp[co*d+j]=co*d+u[0][ref*d+j]-ref*d; ph[co*d+j]=u[1][ref*d+j]
   atoms[f'v{g}_{k}']=(tuple(pp),tuple(ph))
  for axis in ('X','Z'):
   name=f'{"X1" if axis=="X" else "Z2"}plus{g}'; s=atoms['s_'+name]; pp=list(s[0]); ph=list(s[1])
   for co in range(9):
    if not start<=co<start+4:
     for j in range(d): pp[co*d+j]=co*d+j; ph[co*d+j]=0
   atoms[f'axis{axis}{g}']=(tuple(pp),tuple(ph))
 for rec in p['unitary_word_labels']:
  for t in rec['word']:
   if t.endswith('~'): atoms[t]=inverse(atoms[t[:-1]])
 for u in atoms.values():
  assert compose(u,inverse(u))==identity
  assert all(eig[j]==eig[u[0][j]] for j in range(n))
 words={():identity}; byid={}
 for rec in p['unitary_word_labels']:
  w=tuple(rec['word'])
  if w: words[w]=compose(words[w[:-1]],atoms[w[-1]])
  byid[rec['id']]=words[w]
 print('model words',N,len(byid),'distinct',len(set(byid.values())),flush=True)
 mats=[]; registry={}; coordinate_pool={}; vector_pool={}
 def reg(m):
  # Hash buckets are only an index: equality is checked exactly, so collisions are harmless.
  key=hash(tuple(sorted(m.items())))
  for cid in registry.get(key,()):
   if mats[cid]==m: return cid
  m={coordinate_pool.setdefault(k,k):vector_pool.setdefault(v,v) for k,v in m.items()}
  cid=len(mats); mats.append(m); registry.setdefault(key,[]).append(cid)
  return cid
 I=diag(range(dim)); zeroid=reg({}); oneid=reg(I)
 @lru_cache(None)
 def subtract(a,b): return reg(add((mats[a],1),(mats[b],-1)))
 @lru_cache(None)
 def graph(typ,ij,u):
  i,j=int(ij[0])-1,int(ij[1])-1
  if typ=='G': ci,cj,off=Q(1,2),Q(1,2),Q(1,2)
  else: ci,cj,off=Q(4,5),Q(1,5),Q(2,5)
  m=add((diag(range(i*n,(i+1)*n)),ci),(diag(range(j*n,(j+1)*n)),cj))
  for c,r in enumerate(u[0]):
   v=sc(PH[u[1][c]],L*off/2); m[i*n+r,j*n+c]=v; m[j*n+c,i*n+r]=conj(v)
  return reg(m)
 @lru_cache(None)
 def triangle(u,v,w):
  return reg(add((mats[graph('G','12',u)],Q(2,3)),(mats[graph('G','23',v)],Q(2,3)),(mats[graph('G','13',w)],Q(2,3)),(I,Q(-1,3))))
 drows={next(k for k in r['terms'] if k.startswith('D_')):r for r in p['projection_linear_relations'] if any(k.startswith('D_') for k in r['terms'])}
 @lru_cache(None)
 def symbol(name):
  if name=='I': return oneid
  if name.startswith('F') and len(name)==2: return reg(diag(range((int(name[1])-1)*n,int(name[1])*n)))
  if name.startswith('S') and len(name)==3: return reg(add(*[(mats[symbol('F'+i)],1) for i in name[1:]]))
  if name.startswith(('G','H')):
   typ=name[0]; ij,idx=name[1:].split('_'); return graph(typ,ij,byid[int(idx)])
  if name.startswith('K_'):
   other=name[2:]; return subtract(symbol('S'+other[1:3]),symbol(other))
  if name.startswith('D_'):
   gg={k[1:3]:byid[int(k.split('_')[1])] for k in drows[name]['terms'] if k.startswith('G')}
   assert compose(gg['12'],gg['23'])==gg['13']
   return triangle(gg['12'],gg['23'],gg['13'])
  if name.startswith('E_'):
   u=atoms['s_'+name[2:]]; m={}
   for frame in range(3):
    for j,r in enumerate(u[0]): m[frame*n+r,frame*n+j]=sc(PH[u[1][j]],Q(L,4))
   return reg(add((I,Q(1,2)),(m,1)))
  if name.startswith(('EF_','NF_')):
   par,fi=name[3:].rsplit('_',1); e=mats[symbol('E_'+par)]; f=mats[symbol('F'+fi)]
   return reg(prod(e if name.startswith('EF_') else add((I,1),(e,-1)),f))
  if name.startswith('J_'):
   par,ij,ab=name[2:].rsplit('_',2); e=mats[symbol('E_'+par)]; g=mats[graph('G',ij,identity)]; su=mats[symbol('S'+ij)]
   return reg(prod(e if ab[0]=='1' else add((I,1),(e,-1)),g if ab[1]=='1' else add((su,1),(g,-1))))
  if name.startswith('active'):
   g=int(name[-1]); co=[0,1,2,3,4] if g==1 else [1,5,6,7,8]
   return reg(diag([f*n+c*d+j for f in range(3) for c in co for j in range(d)]))
  raise ValueError(name)
 maps={s:np.full(16*len(questions[s]),-1,dtype=np.int32) for s in ('A','B')}
 # All core binary effects are synchronized; Bob stores its transpose-back matrix.
 for side in ('A','B'):
  for rec in questions[side]:
   q=rec['id']
   if side=='B' and rec['kind']=='binary':
    maps[side][16*q:16*q+2]=maps['A'][16*q:16*q+2]; continue
   if rec['kind']=='anchor': maps[side][16*q]=oneid
   elif rec['kind']=='binary':
    cid=symbol(rec['name']); maps[side][16*q+1]=cid; maps[side][16*q]=subtract(oneid,cid)
 # MS in column-0 basis: XI->ZI, ZI->XI. All square cells are real.
 def kron(a,b): return np.kron(a,b)
 z=np.diag([1,-1]); x=np.array([[0,1],[1,0]],dtype=int); eye=np.eye(2,dtype=int); yy=np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]],dtype=int)
 MS=[[kron(z,eye),kron(eye,x),kron(z,x)],[kron(eye,z),kron(x,eye),kron(x,z)],[kron(z,z),kron(x,x),-yy]]
 def emb4(g,mat):
  start=1 if g==1 else 5; out={}
  for f in range(3):
   for a,b in itertools.product(range(4),repeat=2):
    if mat[a,b]:
     for j in range(d): out[f*n+(start+a)*d+j,f*n+(start+b)*d+j]=real(int(mat[a,b]))
  return out
 def flag(g): return diag([f*n+(0 if g==1 else 1)*d+j for f in range(3) for j in range(d)])
 def original(rec):
  g=rec['g']; k,i=rec['tag']; ans=rec['answers']; es=[]
  for a in ans:
   if a=='inactive': e=add((I,1),(mats[symbol('active'+str(g))],-1))
   elif a=='f': e=flag(g)
   elif isinstance(a,list):
    ops=MS[i] if k=='R' else ([MS[h][i] for h in range(3)] if k=='C' else MS[0])
    # T answers are column-0 outcomes, U answers are row-0 outcomes.
    if k=='T': ops=[MS[h][0] for h in range(3)]
    mat=np.eye(4,dtype=np.int64)*L
    for op,sgn in zip(ops,a):
     raw=mat@(np.eye(4,dtype=np.int64)+sgn*op); assert not np.any(raw%2); mat=raw//2
    e=emb4(g,mat)
   else:
    sgn=1 if a=='+' else -1; e=add((flag(g),Q(1,2))); start=1 if g==1 else 5; fc=0 if g==1 else 1
    # sign projection (f +/- c_i)/sqrt2 or (f +/- (c0+(-1)^i c1)/sqrt2)/sqrt2.
    for f in range(3):
     for j in range(d):
      ff=f*n+fc*d+j
      if k=='T':
       cc=f*n+(start+i)*d+j; e[cc,cc]=real(L//2); e[ff,cc]=real(sgn*L//2); e[cc,ff]=real(sgn*L//2)
      else:
       cs=[f*n+start*d+j,f*n+(start+1)*d+j]; signs=[1,(-1)**i]
       for h,c in enumerate(cs):
        vec=[0]*8; vec[1]=sgn*signs[h]*L//4; e[ff,c]=tuple(vec); e[c,ff]=tuple(vec)
        for l,c2 in enumerate(cs): e[c,c2]=real(signs[h]*signs[l]*L//4)
   es.append(reg(e))
  return es
 for side in ('A','B'):
  for rec in questions[side]:
   if rec['kind']=='fms':
    es=original(rec)
    for ai,cid in enumerate(es): maps[side][16*rec['id']+ai]=cid
   elif rec['kind']=='refined':
    z=mats[symbol(rec['label'])]
    for ai,(a,bit) in enumerate(rec['answers']):
     orig=questions[side][rec['coarse']]; idx=orig['answers'].index(a); e=mats[int(maps[side][16*orig['id']+idx])]
     assert prod(e,z)==prod(z,e)
     maps[side][16*rec['id']+ai]=reg(prod(e,z if bit else add((I,1),(z,-1))))
 # Prove the instantiated questions are PVMs by exact multiplication and sums.
 print('effect classes',N,len(mats),'checking projectivity',flush=True)
 for m in mats:
  assert m==star(m)
  assert prod(m,m)==m
 for side in ('A','B'):
  for rec in questions[side]:
   if rec['kind'] in ('fms','refined'):
    assert add(*[(mats[int(maps[side][16*rec['id']+ai])],1) for ai in range(len(rec['answers']))])==I
 # K unnormalized; use Q-field vectors with integer sqrt2 for the middle sector.
 amp=[]
 for f in range(3):
  for c in range(9):
   for w in ws:
    v=[0]*8; v[1 if c in range(1,5) else 0]=(1 if c<5 else 2)*2**(N-w.count('A')); amp.append(tuple(v))
 zsum=sum((Q(1,4**w.count('A')) for w in ws),Q(0)); norm=75*zsum*4**N
 return maps,mats,amp,norm,ws

def run(variant,N):
 gc.collect()
 start=time.perf_counter(); path=OUT/f'functional-{variant}.zip'
 with zipfile.ZipFile(path) as z:
  meta=json.loads(z.read('format.json')); questions=json.loads(z.read('questions.json')); p=json.loads(z.read('core.json'))
  amap,mats,amp,norm,ws=model(N,p,questions)
  # Aggregate only identical actual effects, never rows or penalty templates.
  agg={}; external={}; records=0; h=hashlib.sha256(); previous=-1
  binary_count=sum(rec['kind']=='binary' for rec in questions['A'])
  with z.open('coefficients.bin') as f:
   while True:
    raw=f.read(16*524288)
    if not raw: break
    assert len(raw)%16==0; h.update(raw); a=np.frombuffer(raw,dtype=[('key','<u8'),('num','<i8')]); keys=a['key']; vals=a['num']; assert np.all(vals!=0)
    assert int(keys[0])>previous and np.all(keys[1:]>keys[:-1]); previous=int(keys[-1])
    aa=amap['A'][(keys>>32).astype(np.int64)]; bb=amap['B'][(keys&0xffffffff).astype(np.int64)]; assert np.all(aa>=0) and np.all(bb>=0)
    kk=(aa.astype(np.uint64)<<32)|bb.astype(np.uint64); order=np.argsort(kk); kk=kk[order]; vv=vals[order]; starts=np.r_[0,np.flatnonzero(kk[1:]!=kk[:-1])+1]; sums=np.add.reduceat(vv,starts)
    # Integer range bound, checked before any numpy reduction could overflow.
    assert sum(abs(int(x)) for x in vals)<2**62
    for k,v in zip(kk[starts],sums):
     if v: ik=int(k); agg[ik]=agg.get(ik,0)+int(v)
    mask=((keys>>32)//16>binary_count)|((keys&0xffffffff)//16>binary_count)
    assert np.all(vals[mask]<0)
    for k,v in zip((((aa[mask].astype(np.uint64))<<32)|bb[mask].astype(np.uint64)),vals[mask]):
     ik=int(k); external[ik]=external.get(ik,0)+int(v)
    records+=len(a)
  assert records==meta['nonzero_coefficients']; assert h.hexdigest()==meta['coefficient_sha256']
  agg={k:v for k,v in agg.items() if v}; print('aggregated literal table',variant,N,records,'effect pairs',len(agg),flush=True)
  # Exact inner product, in the unnormalized coefficient matrix coordinates.
  QZERO=Q(0); total=[QZERO]*8; nonzero_pairs=0
  assert norm.denominator==1
  @lru_cache(8192)
  def probability(a,b):
   A=mats[a]; B=mats[b]; small=A if len(A)<len(B) else B; out=[0]*8
   for i,j in small:
    if (i,j) not in A or (j,i) not in B: continue
    value=fm(fm(A[i,j],B[j,i]),fm(amp[i],amp[j]))
    for k,v in enumerate(value): out[k]+=v
   return tuple(Q(x,L*L*int(norm)) if x else QZERO for x in out)
  for key,v in agg.items():
   a,b=key>>32,key&0xffffffff; pr=probability(a,b)
   if any(pr): nonzero_pairs+=1
   for k,t in enumerate(pr):
    if t: total[k]+=v*t/Q(meta['denominator'])
  ext=[Q(0)]*8
  for key,v in external.items():
   pr=probability(key>>32,key&0xffffffff)
   for k,t in enumerate(pr): ext[k]+=v*t/Q(meta['denominator'])
  assert not any(ext),ext
  # Identify the boundary directly as the matrix of its exported G3 row.
  rec=next(r for r in p['projection_linear_relations'] if r['name']=='R_isometry')
  qbyname={r['name']:r['id'] for r in questions['A']}
  def rowmat(r):
   den=math.lcm(*(Q(v).denominator for v in r['terms'].values()))
   return add(*[(mats[int(amap['A'][16*qbyname[k]+1])] if k!='I' else mats[int(amap['A'][0])],Q(v)*den) for k,v in r['terms'].items()])
  residual=rowmat(rec); expected={}; iso_den=math.lcm(*(Q(v).denominator for v in rec['terms'].values()))
  for j,w in enumerate(ws):
   if w.count('A')==N: expected[j,9*len(ws)+j]=real(-L*iso_den); expected[9*len(ws)+j,j]=real(-L*iso_den)
  assert residual==expected
  failed=[]; polynames={r['name'] for r in p['polynomial_relations']}
  for r in p['projection_linear_relations']:
   if r['name'] in polynames and rowmat(r): failed.append(r['name'])
  assert failed==['R_isometry'],failed
  top=sum((Q(1,4**w.count('A')) for w in ws if w.count('A')==N),Q(0))/sum((Q(1,4**w.count('A')) for w in ws),Q(0))
  predicted=-2*top/75
  assert total==[predicted]+[Q(0)]*7,(variant,N,total,predicted)
  result={'variant':variant,'N':N,'tree_dimension':len(ws),'local_dimension':27*len(ws),'records_evaluated':records,'coefficient_sha256':h.hexdigest(),'effect_classes':len(mats),'aggregated_effect_pairs':len(agg),'top_shell_mass':str(top),'exact_score':str(total[0]),'predicted_boundary_score':str(predicted),'irrational_components':[str(t) for t in total[1:]],'all_PVMs_exact':True,'external_event_penalty':str(ext[0]),'only_failed_polynomial_row':failed,'peak_working_set_bytes':peak_memory(),'elapsed_seconds':time.perf_counter()-start}
  (OUT/f'evaluation-{variant}-N{N}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
def peak_memory():
 class PMC(ctypes.Structure):
  _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ('peak','working','qpeak','q','npeak','n','page','pagepeak')]
 pm=PMC(); pm.cb=ctypes.sizeof(pm); ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb)
 assert pm.peak<4*1024**3
 return int(pm.peak)
if __name__=='__main__':
 for variant in sys.argv[1:] or ['small','full']:
  if ':' in variant:
   variant,only=variant.split(':'); cutoffs=(int(only),)
  else: cutoffs=(0,1,2)
  for N in cutoffs: run(variant,N)








