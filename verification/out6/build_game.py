"""Build a finite 0/1 game by a Hadamard-column compiler. Writes only out6.
The question distribution is uniform on the full Cartesian product.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math, zipfile, struct, hashlib, collections, sys
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'out5'/'functional-small.zip'
def bitsmask(answers, pred):
    return sum(1<<i for i,a in enumerate(answers) if pred(a))
def compile_game():
    with zipfile.ZipFile(SOURCE) as z:
        old=json.loads(z.read('questions.json')); core=json.loads(z.read('core.json'))
        raw=z.read('coefficients.bin')
    rows=core['projection_linear_relations']
    variables=sorted({k for r in rows for k in r['terms']}-{'I'})
    oldid={q['name']:q['id'] for q in old['A']}; oldQ=len(old['A'])
    external=[]
    for key,num in struct.iter_unpack('<Qq',raw):
        ea,eb=key>>32,key&0xffffffff; qa,a=divmod(ea,16); qb,b=divmod(eb,16)
        if old['A'][qa]['kind'] in ('fms','refined') or old['B'][qb]['kind'] in ('fms','refined'):
            assert num<0; external.append([qa,a,qb,b])
    plans=[]
    for row in rows:
        al={k:F(v) for k,v in row['terms'].items()}
        den=math.lcm(*(v.denominator for v in al.values()))
        c0=int(2*den*al.get('I',0)+sum(den*v for k,v in al.items() if k!='I'))
        slots=[[None,1],[None,-1]]
        for name,c in [(None,c0)]+[(k,int(den*v)) for k,v in sorted(al.items()) if k!='I']:
            slots.extend([[name,1 if c>0 else -1] for _ in range(abs(c))])
        assert len(slots)%2==0
        d=4
        while d<len(slots): d*=4
        slots.extend([[None,1],[None,-1]]*((d-len(slots))//2))
        plans.append({'name':row['name'],'den':den,'d':d,'slots':slots})
    maxd=max(p['d'] for p in plans); n=maxd.bit_length()-1
    assert n%2==0 and maxd==256
    queries=[]; binds=[]; orth=[]
    def query(kind,answers,**meta):
        q=oldQ+len(queries); queries.append(dict(id=q,kind=kind,answers=answers,**meta)); return q
    def answers(q): return old['A'][q]['answers'] if q<oldQ else queries[q-oldQ]['answers']
    def bind(q,mq,r,mr): binds.append([q,str(mq),r,str(mr)])
    def event(q,pred): return bitsmask(answers(q),pred)
    def positive(q): return event(q,lambda a:a==1)
    # Four synchronized MS algebras; positive cell bit 1.
    axes=[]; cells=[]
    from itertools import product
    for g in range(n//2):
        cc=[[query('scalar_cell',[0,1],g=g,i=i,j=j) for j in range(3)] for i in range(3)]
        cells.append(cc)
        for orientation in ('R','C'):
            for i in range(3):
                parity=-1 if orientation=='C' and i==2 else 1
                an=[list(t) for t in product((1,-1),repeat=3) if math.prod(t)==parity]
                q=query('scalar_ms',an,g=g,orientation=orientation,index=i)
                for j in range(3):
                    b=cc[i][j] if orientation=='R' else cc[j][i]
                    bind(q,event(q,lambda a,j=j:a[j]==1),b,2)
        axes.extend([{'Z':cc[0][0],'X':cc[1][1]},{'Z':cc[0][1],'X':cc[1][0]}])
    def joint2(p,q,kind,**meta):
        r=query(kind,[[a,b] for a in (0,1) for b in (0,1)],p=p,q=q,**meta)
        bind(r,event(r,lambda a:a[0]==1),p,2)
        bind(r,event(r,lambda a:a[1]==1),q,2)
        return r
    for i in range(n):
        for j in range(i+1,n):
            if i//2!=j//2:
                for a,b in product(('X','Z'),repeat=2): joint2(axes[i][a],axes[j][b],'cross_axes')
    for name in variables:
        for i in range(n):
            for a in ('X','Z'): joint2(oldid[name],axes[i][a],'commute_old',variable=name,axis=i,letter=a)
    globals={}
    for a in ('X','Z'):
        q=query('scalar_global',list(range(maxd)),axis=a,n=n); globals[a]=q
        for i in range(n): bind(q,event(q,lambda z,i=i:not ((z>>i)&1)),axes[i][a],2)
    pair={}
    for dn in sorted({p['d'] for p in plans}):
        nn=dn.bit_length()-1
        for j in range(1,dn):
            pivot=(j&-j).bit_length()-1
            kernel=[(1<<k)^((1<<pivot) if (j>>k)&1 else 0) for k in range(nn) if k!=pivot]
            # Answer low bit: X^j negative; other bits: independent Z parity negatives.
            q=query('scalar_pair',list(range(dn)),n=nn,j=j,kernel=kernel)
            bind(q,event(q,lambda z:not (z&1)),globals['X'],event(globals['X'],lambda z,j=j:(z&j).bit_count()%2==0))
            for k,l in enumerate(kernel):
                bind(q,event(q,lambda z,k=k:not ((z>>(k+1))&1)),globals['Z'],event(globals['Z'],lambda z,l=l:(z&l).bit_count()%2==0))
            pair[dn,j]=q
    for ri,plan in enumerate(plans):
        dn=plan['d']; nn=dn.bit_length()-1
        dr=query('row_columns',list(range(dn)),row=ri)
        plan['question']=dr
        for j in range(1,dn):
            name,sgn=plan['slots'][j]
            jr=query('row_joint',[[e,k] for e in (0,1) for k in (0,1,2)],row=ri,slot=j,variable=name,sign=sgn)
            plan.setdefault('joint_questions',[]).append(jr)
            eq=oldid[name] if name is not None else 0
            bind(jr,event(jr,lambda a:a[0]==1),eq,2 if name is not None else 1)
            pr=pair[dn,j]
            # Scalar three-way coarse outcomes: answer 0 -> R+, 1 -> R-, others -> outside.
            for k in range(3):
                bind(jr,event(jr,lambda a,k=k:a[1]==k),pr,event(pr,lambda a,k=k:(0 if a==0 else 1 if a==1 else 2)==k))
            orth.append([dr,jr,j,sgn])
        # This is the only new test that can fail in a canonical truncation.
        orth.append([dr,globals['X'],0,dn-1])
    allq={s:old[s]+queries for s in ('A','B')}
    sync=[q['id'] for q in old['A'] if q['kind']=='binary']+[q['id'] for q in queries]
    spec={'format':'hadamard-game-v1','distribution':'uniform-full-cartesian','old_questions':oldQ,'questions_per_party':len(allq['A']),'max_answers':max(len(q['answers']) for q in allq['A']),'scalar_qubits':n,'scalar_dimension':maxd,'core_variables':variables,'rows':plans,'axes':axes,'globals':globals,'bindings':binds,'synchrony':sync,'orthogonality':orth,'old_forbidden':external}
    assert len(allq['A'])==len(allq['B'])==169460
    return spec,allq,core

def forbidden(spec,questions,qa,a,qb,b,index=None):
    """Literal 0/1 verifier. Optional precomputed pair index avoids a linear scan."""
    if not(0<=a<len(questions['A'][qa]['answers']) and 0<=b<len(questions['B'][qb]['answers'])): return True
    if index is None: index=make_index(spec)
    for rule in index.get((qa,qb),()):
        if rule[0]=='sync' and a!=b: return True
        if rule[0]=='old' and (a,b)==tuple(rule[1:]): return True
        if rule[0]=='bind' and ((rule[1]>>a)&1)!=((rule[2]>>b)&1): return True
        if rule[0]=='graph':
            j,sgn=rule[1:]; e,k=questions['B'][qb]['answers'][b]
            chi=1 if (a&j).bit_count()%2==0 else -1
            if k<2 and (1 if k==0 else -1)==-chi*sgn*(2*e-1): return True
        if rule[0]=='rowzero' and a==0 and b&rule[1]==0: return True
    return False

def make_index(spec):
    ix=collections.defaultdict(list)
    for q in spec['synchrony']: ix[q,q].append(('sync',))
    for q,a,r,b in spec['old_forbidden']: ix[q,r].append(('old',a,b))
    for q,mq,r,mr in spec['bindings']:
        mq,mr=int(mq),int(mr)
        ix[q,r].append(('bind',mq,mr)); ix[r,q].append(('bind',mr,mq))
    for q,r,j,sgn in spec['orthogonality']:
        ix[q,r].append(('graph',j,sgn) if j else ('rowzero',sgn))
    return ix

if __name__=='__main__':
    spec,questions,core=compile_game()
    with zipfile.ZipFile(OUT/'game.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,obj in [('game.json',spec),('questions.json',questions),('core.json',core)]:
            z.writestr(name,json.dumps(obj,separators=(',',':')))
        z.write(__file__,'build_game.py')
    counts={'questions_per_party':spec['questions_per_party'],'max_answers':spec['max_answers'],'new_questions':spec['questions_per_party']-spec['old_questions'],'question_histogram':dict(collections.Counter(len(q['answers']) for q in questions['A'])),'new_kinds':dict(collections.Counter(q['kind'] for q in questions['A'][spec['old_questions']:])), 'rows':len(spec['rows']),'row_dimensions':dict(collections.Counter(r['d'] for r in spec['rows'])),'bindings':len(spec['bindings']),'synchrony_tests':len(spec['synchrony']),'graph_tests':sum(bool(r[2]) for r in spec['orthogonality']),'rowzero_tests':sum(not r[2] for r in spec['orthogonality']),'old_forbidden_events':len(spec['old_forbidden']),'sha256':hashlib.file_digest((OUT/'game.zip').open('rb'),'sha256').hexdigest(),'bytes':(OUT/'game.zip').stat().st_size}
    (OUT/'counts.json').write_text(json.dumps(counts,indent=2))
    print(json.dumps(counts,indent=2))

