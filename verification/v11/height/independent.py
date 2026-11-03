"""Independent infinite-word action engine. No import of either prior evaluator.
Only compress *effects*, never atoms; all word actions are on the infinite tree.
Writes only beside this file. Run with a process timeout, one BLAS thread.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json,zipfile,math,struct,time,hashlib,sys,gc,threading,ctypes,psutil
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict,Counter
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix,eye,kron
def verification_root():
    if 'ENTROPY_VERIFICATION_ROOT' in os.environ:
        return Path(os.environ['ENTROPY_VERIFICATION_ROOT']).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent/'out5/functional-small.zip').is_file() and (parent/'out6/game.zip').is_file():
            return parent
    raise FileNotFoundError('Pass ENTROPY_VERIFICATION_ROOT for the verification archives')
ROOT=verification_root()
OUT=Path(os.environ.get('ENTROPY_V11_OUTPUT', str(Path(__file__).resolve().parent/'results')))
OUT.mkdir(parents=True,exist_ok=True)
watchdog=threading.Timer(900,lambda:os._exit(124));watchdog.daemon=True;watchdog.start()
def memory_watch():
    proc=psutil.Process()
    while True:
        if proc.memory_info().rss>5*1024**3:os._exit(137)
        time.sleep(.5)
threading.Thread(target=memory_watch,daemon=True).start()
with zipfile.ZipFile(ROOT/'out5/functional-small.zip') as z:
    CORE=json.loads(z.read('core.json')); QUESTIONS=json.loads(z.read('questions.json'))
    RAW=z.read('coefficients.bin'); META=json.loads(z.read('format.json'))
assert hashlib.sha256(RAW).hexdigest()==META['coefficient_sha256']
LABELS={r['id']:tuple(r['word']) for r in CORE['unitary_word_labels']}
DROWS={k:r for r in CORE['projection_linear_relations'] for k in r['terms'] if k.startswith('D_')}
WEIGHTS=(1,2,2,2,2,4,4,4,4)

def height(w): return w.count('A')-w.count('B')

@lru_cache(None)
def atom(token,c,w):
    inverse=token.endswith('~'); t=token.rstrip('~'); phase=1+0j
    if t.startswith('s_'):
        p=t[2:]
        if p=='root': return c,w,1 if c==0 else -1
        if p.startswith('m'):
            co=(1 if p[1]=='1' else 5)+2*int(p[3])+int(p[4])
            return c,w,1 if c==co else -1
        g=int(p[-1]); start=1 if g==1 else 5
        if start<=c<start+4:
            return start+((c-start)^(2 if p.startswith('X1') else 1)),w,1
        return c,w,-1
    if t.startswith('axis'):
        g=int(t[-1]); start=1 if g==1 else 5
        return (start+((c-start)^(2 if t[4]=='X' else 1)),w,1) if start<=c<start+4 else (c,w,1)
    if t.startswith('a_'):
        sign=1 if t=='a_plus' else -1
        if not inverse:
            if c==0: return 5,w+'A',1
            if c==5 and w.endswith('A'): return 0,w[:-1],sign*1j
            return c,w,sign*1j
        if c==5 and w.endswith('A'): return 0,w[:-1],1
        if c==0: return 5,w+'A',-sign*1j
        return c,w,-sign*1j
    if t.startswith('u'):
        r=int(t[1]); kind=t[3:]; active=c==(0,1,5)[r]
    elif t.startswith('v'):
        g=int(t[1]); kind=t[3:]; start=1 if g==1 else 5
        active=c==(0 if g==1 else 1) or start<=c<start+4
    else: raise ValueError(t)
    if not active: return c,w,1
    h=height(w)
    if kind.startswith('b'):
        sign=1 if kind=='bplus' else -1
        if not w: return c,w,sign*(-1j if inverse else 1j)
        if not inverse:
            if h>=2: return c,w+'B',1
            return c,w.rstrip('B'),sign*1j
        if w.endswith('B'): return c,w[:-1],1
        return c,w+'B'*(h-1),-sign*1j
    if kind=='p': phase=1 if not w else -1
    elif kind=='q': phase=1 if h==1 else -1
    elif kind=='c':
        x=2.0**(-h); phase=2*x-1+2j*math.sqrt(x*(1-x))
    else: raise ValueError(kind)
    return c,w,phase.conjugate() if inverse else phase

@lru_cache(None)
def word_action(tokens,c,w):
    ph=1+0j
    for t in reversed(tokens):
        c,w,z=atom(t,c,w); ph*=z
    return c,w,ph

def invword(w): return tuple(t[:-1] if t.endswith('~') else t+'~' for t in reversed(w))

def symbol_column(name,b):
    """Full uncompressed action of a projection on one frame/corner/history."""
    f,c,w=b
    if name=='I': return {b:1}
    if name.startswith('F') and len(name)==2: return {b:1} if f==int(name[1])-1 else {}
    if name.startswith('S') and len(name)==3: return {b:1} if str(f+1) in name[1:] else {}
    if name.startswith('K_'):
        g=name[2:]; a=symbol_column('S'+g[1:3],b); return combine((a,1),(symbol_column(g,b),-1))
    if name[0] in 'GH':
        pair,idx=name[1:].split('_'); i,j=int(pair[0])-1,int(pair[1])-1
        if f not in (i,j): return {}
        a,beta,off=(.5,.5,.5) if name[0]=='G' else (.8,.2,.4)
        t=LABELS[int(idx)]
        if f==j:
            co,v,z=word_action(t,c,w); return combine(({(f,c,w):beta},1),({(i,co,v):z*off},1))
        co,v,z=word_action(invword(t),c,w); return combine(({(f,c,w):a},1),({(j,co,v):z*off},1))
    if name.startswith('D_'):
        r=DROWS[name]; return combine(*[(symbol_column(k,(f,c,w)),-float(F(v))/float(F(r['terms'][name]))) for k,v in r['terms'].items() if k!=name])
    if name.startswith('E_'):
        co,v,z=atom('s_'+name[2:],c,w); return combine(({(f,c,w):.5},1),({(f,co,v):z*.5},1))
    if name.startswith(('EF_','NF_')):
        par,fi=name[3:].rsplit('_',1)
        if f!=int(fi)-1: return {}
        e=symbol_column('E_'+par,(f,c,w))
        return e if name.startswith('EF_') else combine(({(f,c,w):1},1),(e,-1))
    if name.startswith('J_'):
        par,ij,bits=name[2:].rsplit('_',2)
        g=symbol_column('G'+ij+'_0',(f,c,w))
        if bits[1]=='0': g=combine((symbol_column('S'+ij,(f,c,w)),1),(g,-1))
        o=defaultdict(complex)
        for b1,z in g.items():
            e=symbol_column('E_'+par,b1)
            if bits[0]=='0': e=combine(({b1:1},1),(e,-1))
            for b2,x in e.items(): o[b2]+=z*x
        return dict(o)
    if name.startswith('active'):
        cs=(0,1,2,3,4) if name[-1]=='1' else (1,5,6,7,8)
        return {(f,c,w):1} if c in cs else {}
    raise ValueError(name)

def combine(*parts):
    out=defaultdict(complex)
    for d,s in parts:
        for k,v in d.items(): out[k]+=s*v
    return {k:v for k,v in out.items() if abs(v)>1e-14}

class Engine:
    def __init__(self,ws):
        self.ws=sorted(ws,key=lambda w:(w.count('A'),len(w),w))
        self.basis=[(f,c,w) for f in range(3) for c in range(9) for w in self.ws]
        self.ix={b:i for i,b in enumerate(self.basis)}; self.n=len(self.basis)
        self.Z=sum(4.**(-w.count('A')) for w in self.ws)
        self.amp=np.array([math.sqrt(WEIGHTS[c])*2.**(-w.count('A')) for f,c,w in self.basis])/math.sqrt(75*self.Z)
        self.I=eye(self.n,format='csr',dtype=complex); self.effects={}; self.registry={}
    def reg(self,m):
        m=m.astype(complex).tocsr(); m.sum_duplicates(); m.eliminate_zeros(); m.sort_indices()
        # Equality up to roundoff merely deduplicates matrices, never scores them.
        m.data.real[np.abs(m.data.real)<1e-14]=0; m.data.imag[np.abs(m.data.imag)<1e-14]=0
        key=(m.indptr.tobytes(),m.indices.tobytes(),np.round(m.data,13).tobytes())
        if key not in self.registry: self.registry[key]=m
        return self.registry[key]
    def effect(self,name):
        if name not in self.effects:
            rr=[];cc=[];vv=[]
            for j,b in enumerate(self.basis):
                for out,z in symbol_column(name,b).items():
                    if out in self.ix: rr.append(self.ix[out]);cc.append(j);vv.append(z)
            self.effects[name]=self.reg(csr_matrix((vv,(rr,cc)),shape=(self.n,self.n)))
        return self.effects[name]
    def prob(self,a,b):
        m=a.multiply(b.transpose()).tocoo()
        return float(np.real(np.sum(m.data*self.amp[m.row]*self.amp[m.col])))
    def sync(self,name):
        e=self.effect(name)
        return max(0.,2*(np.dot(self.amp**2,e.diagonal()).real-self.prob(e,e)))
    def original(self,r):
        g=r['g']; start=1 if g==1 else 5; flag=0 if g==1 else 1
        typ,i=r['tag']; answers=r['answers']; out=[]
        x=np.array([[0,1],[1,0]]); z=np.diag([1,-1]); id2=np.eye(2)
        yy=np.kron(np.array([[0,-1j],[1j,0]]),np.array([[0,-1j],[1j,0]]))
        ms=[[np.kron(z,id2),np.kron(id2,x),np.kron(z,x)],[np.kron(id2,z),np.kron(x,id2),np.kron(x,z)],[np.kron(z,z),np.kron(x,x),-yy]]
        def embed(base): return self.reg(kron(eye(3),kron(csr_matrix(base),eye(len(self.ws))),format='csr'))
        for a in answers:
            b=np.zeros((9,9),complex)
            if a=='inactive':
                active=(flag,)+tuple(range(start,start+4)); b=np.diag([int(c not in active) for c in range(9)])
            elif a=='f': b[flag,flag]=1
            elif isinstance(a,list):
                ops=ms[i] if typ=='R' else (ms[0] if typ=='U' else [ms[k][i] for k in range(3)])
                if typ=='T': ops=[ms[k][0] for k in range(3)]
                v=np.eye(4)
                for sg,op in zip(a,ops): v=v@(np.eye(4)+sg*op)/2
                b[start:start+4,start:start+4]=v
            else:
                sg=1 if a=='+' else -1; vec=np.zeros(9); vec[flag]=1/math.sqrt(2)
                if typ=='T': vec[start+i]=sg/math.sqrt(2)
                else: vec[start]=sg/2;vec[start+1]=sg*((-1)**i)/2
                b=np.outer(vec,vec)
            out.append(embed(b))
        return out
    def all_measurements(self):
        maps={}; cache={}
        for side in ('A','B'):
            for r in QUESTIONS[side]:
                if r['kind']=='anchor': es=[self.I]
                elif r['kind']=='binary':
                    e=self.effect(r['name']);es=[self.reg(self.I-e),e]
                elif r['kind']=='fms': es=self.original(r)
                else:
                    z=self.effect(r['label']); coarse=QUESTIONS[side][r['coarse']]
                    es=[self.reg(maps[side,16*r['coarse']+coarse['answers'].index(a)]@(z if bit else self.I-z)) for a,bit in r['answers']]
                    # Labels preserve the history set, hence compression commutes.
                for a,e in enumerate(es): maps[side,16*r['id']+a]=e
        return maps
    def bell(self):
        maps=self.all_measurements(); ps={}; vals=[]; external=[]
        for key,num in struct.iter_unpack('<Qq',RAW):
            a,b=key>>32,key&0xffffffff; ea,eb=maps['A',a],maps['B',b]
            k=id(ea),id(eb)
            if k not in ps: ps[k]=self.prob(ea,eb)
            term=num/2048*ps[k]; vals.append(term)
            if QUESTIONS['A'][a//16]['kind'] in ('fms','refined') or QUESTIONS['B'][b//16]['kind'] in ('fms','refined'): external.append(term)
        return -math.fsum(vals),-math.fsum(external)

def validate_atoms():
    words=make_words(5,7)
    tokens={t.rstrip('~') for w in LABELS.values() for t in w}
    err=0
    for w in words:
      for c in range(9):
       for t in tokens:
        co,v,z=atom(t,c,w); cc,vv,zz=atom(t+'~',co,v)
        assert (cc,vv)==(c,w);err=max(err,abs(z*zz-1))
        assert F(WEIGHTS[c],4**w.count('A'))==F(WEIGHTS[co],4**v.count('A'))
    return {'words':len(words),'tokens':len(tokens),'max_inverse_error':err}

def make_words(H,N):
    out={''};todo=['']
    while todo:
        w=todo.pop(); h=height(w)
        for a in 'AB':
            if (a=='A' and h<H and w.count('A')<N) or (a=='B' and h>=2):
                v=w+a;out.add(v);todo.append(v)
    return out

def mu(ws): return sum((F(1,4**w.count('A')) for w in ws),F(0))
def boundary(ws): return {w for w in ws if w+'A' not in ws}

def main_sets():
    cases=[('height1',make_words(1,2)),('height2_n2',make_words(2,2)),('height2_n4',make_words(2,4)),('height3_n3',make_words(3,3)),('height3_n4',make_words(3,4))]
    cases += [('comb'+str(k),{''}|{'A'*n+'B'*j for n in range(1,k+1) for j in range(n)}) for k in (2,3,5,7,10,12)]
    if len(sys.argv)>2:cases=[x for x in cases if x[0] in sys.argv[2:]]
    results=[]
    for name,ws in cases:
        t=time.perf_counter(); e=Engine(ws); loss,ext=e.bell();ratio=mu(boundary(ws))/mu(ws)
        pred=F(50808529,540000)*ratio
        result={'set':name,'words':len(ws),'boundary_max_height':max(map(height,boundary(ws))),'boundary_ratio':str(ratio),'literal_B_loss':loss,'prediction':float(pred),'error':loss-float(pred),'external_loss':ext,'peak_working_set_bytes':psutil.Process().memory_info().peak_wset,'seconds':time.perf_counter()-t}
        print(json.dumps(result),flush=True);results.append(result);del e;gc.collect()
    target=OUT/('sets_profile.json' if len(sys.argv)>2 else 'sets.json')
    target.write_text(json.dumps(results,indent=2))

def constants():
    # All states reachable from empty histories under these atoms lie in this
    # finite invariant union of exact density eigenspaces (13 base coordinates).
    basis=[(f,c,'') for f in range(3) for c in range(9)]+[(f,c,'A') for f in range(3) for c in range(5,9)]
    ix={b:i for i,b in enumerate(basis)};n=len(basis); scale=60
    mask=np.array([w=='' for f,c,w in basis]); wt=np.array([WEIGHTS[c] if w=='' else 0 for f,c,w in basis],dtype=np.int64)
    symbols=sorted({r['name'] for r in QUESTIONS['A'] if r['kind']=='binary'})
    mats={}; sig={}; fullmats={}
    for name in symbols:
        m=np.zeros((n,n),complex)
        for j,b in enumerate(basis):
            for v,z in symbol_column(name,b).items():
                assert v in ix,(name,b,v)
                m[ix[v],j]+=z
        assert np.max(abs(m@m-m))<1e-13 and np.max(abs(m-m.conj().T))<1e-13
        real=np.rint(m.real*scale).astype(np.int64);imag=np.rint(m.imag*scale).astype(np.int64)
        assert np.max(abs(m-(real+1j*imag)/scale))<1e-13,name
        # Only crossing entries contribute; no square roots remain.
        numerator=int(np.sum(((real**2+imag**2)*wt[None,:])[~mask,:]))
        sig[name]=F(2*numerator,75*scale**2)
        fullmats[name]=(real,imag)
    kappas={s:F(1) for s in symbols}
    maxrow=0
    for row in CORE['projection_linear_relations']:
        a={k:F(v) for k,v in row['terms'].items()};k=1+sum(v*v for s,v in a.items() if s!='I')/2
        for s in a:
            if s!='I': kappas[s]+=k
        rm=sum((float(v)*(np.eye(n) if s=='I' else (fullmats[s][0]+1j*fullmats[s][1])/scale) for s,v in a.items()),np.zeros((n,n),complex))
        maxrow=max(maxrow,float(np.max(abs(rm))))
    lb=sum((kappas[s]*sig[s] for s in symbols),F(0))
    with zipfile.ZipFile(ROOT/'out6/game.zip') as z: game=json.loads(z.read('game.json'))
    eyei=np.eye(n,dtype=np.int64)*scale
    # Deduplicate signed reflections up to sign: signs cancel in leakage norms.
    refs={};refs_ids={};rkeys={}
    for s in game['core_variables']:
        re,im=fullmats[s];re=2*re-eyei;im=2*im
        key=(re.tobytes(),im.tobytes())
        if key not in refs_ids:
            idx=len(refs_ids);refs_ids[key]=idx;refs[idx]=(re,im)
        rkeys[s]=refs_ids[key]
    ident=len(refs);refs[ident]=(eyei,np.zeros_like(eyei));rkeys[None]=ident
    pair_cache={}
    def leakage(i,j):
        # Product of *full*, uncompressed reflections. Integer arithmetic.
        key=(i,j)
        if key not in pair_cache:
            a,b=refs[i];c,d=refs[j]; re=a@c-b@d;im=a@d+b@c
            pair_cache[key]=int(np.sum(((re**2+im**2)*wt[None,:])[~mask,:]))
        return pair_cache[key]
    row_sync=F(0);graph=F(0);slots=F(0);slot_count=Counter()
    for r in game['rows']:
        d=r['d'];counts=Counter(rkeys[s] for s,sg in r['slots'])
        num=sum(ci*cj*leakage(i,j) for i,ci in counts.items() for j,cj in counts.items())
        row_sync+=F(num,75*scale**4*d*d)
        for s,sg in r['slots'][1:]:
            if s is not None:
                slot_count[s]+=1;slots+=3*sig[s];graph+=2*sig[s]/d
    old=sum(sig.values(),F(0));commute=48*sum((sig[s] for s in game['core_variables']),F(0))
    lg=old+commute+slots+row_sync+graph
    active=[s for s in symbols if sig[s]]
    result={'Lambda_B':str(lb),'Lambda_G':str(lg),'Lambda_G_float':float(lg),'active':len(active),'active_types':dict(Counter(s.split('_')[0] if not s.startswith('K_') else 'K' for s in active)),'kappa_min':str(min(kappas[s] for s in active)),'kappa_max':str(max(kappas[s] for s in active)),'max_full_row_residual':maxrow,'game_terms':{k:str(v) for k,v in [('old_sync',old),('commute_old',commute),('slot_sync_binding',slots),('row_columns_sync',row_sync),('graph_orthogonality',graph)]},'pair_cache':len(pair_cache),'distinct_reflections':len(refs),'Q':game['questions_per_party'],'H1_B':str(lb/5),'H2_B':str(lb/17),'H1_G':str(lg/(5*game['questions_per_party']**2)),'H2_G':str(lg/(17*game['questions_per_party']**2))}
    assert lb==F(50808529,540000),(lb,result)
    print(json.dumps(result,indent=2),flush=True)
    (OUT/'constants.json').write_text(json.dumps(result,indent=2))
    (OUT/'symbol_constants.json').write_text(json.dumps({s:{'sigma_per_boundary':str(sig[s]),'kappa':str(kappas[s]),'slot_count':slot_count[s]} for s in symbols},indent=2))

def game_sets():
    with zipfile.ZipFile(ROOT/'out6/game.zip') as z: g=json.loads(z.read('game.json'))
    sc=json.loads((OUT/'symbol_constants.json').read_text())
    active=[s for s,v in sc.items() if F(v['sigma_per_boundary'])]
    results=[]
    cases=[('height1',make_words(1,1)),('height3_n3',make_words(3,3)),('comb10',{''}|{'A'*n+'B'*j for n in range(1,11) for j in range(n)})]
    for name,ws in cases:
        start=time.perf_counter();e=Engine(ws); universe=dict(e.ix);columns={};mats={};reg={};index={};sigma={}
        for s in active:
            cc=[]
            for j,b in enumerate(e.basis):
                col=combine((symbol_column(s,b),2),({b:1},-1))
                for out in col:
                    if out not in universe: universe[out]=len(universe)
                cc.append(col)
            columns[s]=cc
        for s,cols in columns.items():
            rr=[];cc=[];vv=[]
            for j,col in enumerate(cols):
                for b,v in col.items():rr.append(universe[b]);cc.append(j);vv.append(v)
            m=csr_matrix((vv,(rr,cc)),shape=(len(universe),e.n));m.sort_indices();m.eliminate_zeros()
            key=(m.indptr.tobytes(),m.indices.tobytes(),np.round(m.data,13).tobytes())
            if key not in reg:
                idx=len(reg);reg[key]=idx;mats[idx]=m
                outside=m[e.n:,:].tocoo();sigma[idx]=float(np.sum(abs(outside.data)**2*e.amp[outside.col]**2))/2
            index[s]=reg[key]
        del columns
        def sig(s):return sigma[index[s]] if s in index else 0.
        cache={}
        def pair(i,j):
            if i is None:return 2*sigma[j] if j is not None else 0.
            if j is None:return 2*sigma[i]
            if i==j:return 0.
            k=tuple(sorted((i,j)))
            if k not in cache:
                a=(mats[i].conj().T@mats[j]).tocoo()
                cache[k]=1-float(np.sum(abs(a.data)**2*e.amp[a.col]**2))
            return cache[k]
        old=sum(sig(s) for s in sc); comm=48*sum(sig(s) for s in g['core_variables'])
        slots=[];graph=[];sync=[]
        for r in g['rows']:
            d=r['d'];cnt=Counter(index.get(s) for s,sg in r['slots'])
            sync.append(math.fsum(ci*cj*pair(i,j) for i,ci in cnt.items() for j,cj in cnt.items())/d**2)
            for s,sg in r['slots'][1:]:slots.append(3*sig(s));graph.append(2*sig(s)/d)
        terms={'old':old,'commute':comm,'slots':math.fsum(slots),'columns':math.fsum(sync),'graph':math.fsum(graph)}
        ratio=mu(boundary(ws))/mu(ws);total=math.fsum(terms.values());pred=float(F(14844029,54000)*ratio)
        result={'set':name,'words':len(ws),'unnormalized_game_loss':total,'prediction':pred,'error':total-pred,'terms':terms,'full_action_rows':len(universe),'distinct_active_reflections':len(mats),'pair_cache':len(cache),'seconds':time.perf_counter()-start}
        print(json.dumps(result),flush=True);results.append(result);del e,mats;gc.collect()
    (OUT/'game_sets.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'sets'
    if mode=='constants': constants()
    elif mode=='game_sets': game_sets()
    elif mode=='atoms': print(json.dumps(validate_atoms(),indent=2))
    else: main_sets()
