"""Exact checker for the serialized 0/1 game. No producer import.
Canonical old effects are reconstructed by the prior independent matrix evaluator.
New tests are contracted from the serialized signed columns, not F_alpha penalties.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import sys, json, zipfile, math, collections, importlib.util, time, hashlib, gc
from fractions import Fraction as F
from pathlib import Path
OUT=Path(__file__).resolve().parent

def read_game():
    with zipfile.ZipFile(OUT/'game.zip') as z:
        spec=json.loads(z.read('game.json')); qs=json.loads(z.read('questions.json')); core=json.loads(z.read('core.json'))
    return spec,qs,core

def audit(s,qs,core):
    Q=s['questions_per_party']; old=s['old_questions']; qq=qs['A']
    assert len(qq)==len(qs['B'])==Q==169460
    assert all(q['id']==i for side in qs.values() for i,q in enumerate(side))
    assert qq[old:]==qs['B'][old:]
    assert max(len(q['answers']) for q in qq)==256
    assert set(s['synchrony'])=={q['id'] for q in qq if q['kind']=='binary' or q['id']>=old}
    bactual=collections.Counter(tuple(x) for x in s['bindings']); bexpect=collections.Counter()
    def mask(q,pred): return sum(1<<i for i,a in enumerate(qq[q]['answers']) if pred(a))
    def binding(q,m,r,mr): bexpect[(q,str(m),r,str(mr))]+=1
    axes=s['axes']; globals=s['globals']; pairs={}; kinds=collections.Counter()
    oldid={q['name']:q['id'] for q in qq[:old]}; cell={}
    for q in qq[old:]:
        k=q['kind']; kinds[k]+=1; qi=q['id']; an=q['answers']
        if k=='scalar_cell': cell[q['g'],q['i'],q['j']]=qi
    for q in qq[old:]:
        k=q['kind']; qi=q['id']; an=q['answers']
        if k=='scalar_ms':
            i=q['index']; orient=q['orientation']; par=-1 if orient=='C' and i==2 else 1
            assert len(an)==4 and all(math.prod(a)==par for a in an)
            for j in range(3):
                c=cell[q['g'],i,j] if orient=='R' else cell[q['g'],j,i]
                binding(qi,mask(qi,lambda a,j=j:a[j]==1),c,2)
        elif k in ('cross_axes','commute_old'):
            assert an==[[0,0],[0,1],[1,0],[1,1]]
            binding(qi,mask(qi,lambda a:a[0]==1),q['p'],2)
            binding(qi,mask(qi,lambda a:a[1]==1),q['q'],2)
        elif k=='scalar_global':
            assert an==list(range(256))
            for i in range(8): binding(qi,mask(qi,lambda a,i=i:not (a>>i)&1),axes[i][q['axis']],2)
        elif k=='scalar_pair':
            n=q['n']; j=q['j']; ker=q['kernel']; dn=1<<n; pairs[dn,j]=qi
            assert an==list(range(dn)) and len(ker)==n-1
            assert {z for z in range(dn) if all((z&l).bit_count()%2==0 for l in ker)}=={0,j}
            assert all((j&l).bit_count()%2==0 for l in ker)
            binding(qi,mask(qi,lambda a:not a&1),globals['X'],mask(globals['X'],lambda a,j=j:(a&j).bit_count()%2==0))
            for i,l in enumerate(ker): binding(qi,mask(qi,lambda a,i=i:not (a>>(i+1))&1),globals['Z'],mask(globals['Z'],lambda a,l=l:(a&l).bit_count()%2==0))
    # Audit all scalar generators and all old/scalar commutation contexts.
    for g in range(4):
        assert axes[2*g]=={'Z':cell[g,0,0],'X':cell[g,1,1]}
        assert axes[2*g+1]=={'Z':cell[g,0,1],'X':cell[g,1,0]}
    expected_cross={tuple(sorted((axes[i][a],axes[j][b]))) for i in range(8) for j in range(i+1,8) if i//2!=j//2 for a in ('X','Z') for b in ('X','Z')}
    assert {tuple(sorted((q['p'],q['q']))) for q in qq[old:] if q['kind']=='cross_axes'}==expected_cross
    assert {(q['p'],q['q']) for q in qq[old:] if q['kind']=='commute_old'}=={(oldid[v],axes[i][a]) for v in s['core_variables'] for i in range(8) for a in ('X','Z')}
    expectedorth=[]; slots_count=0
    for ri,(r,source) in enumerate(zip(s['rows'],core['projection_linear_relations'])):
        assert r['name']==source['name']; d=r['d']; slots=r['slots']; dr=r['question']; assert len(slots)==d and slots[0]==[None,1]
        assert qq[dr]['kind']=='row_columns' and qq[dr]['row']==ri and qq[dr]['answers']==list(range(d))
        al={k:F(v) for k,v in source['terms'].items()}; den=math.lcm(*(v.denominator for v in al.values())); assert den==r['den']
        ci=collections.Counter()
        for name,sgn in slots:
            assert sgn in (-1,1)
            if name is None: ci['I']+=sgn
            else: ci[name]+=2*sgn; ci['I']-=sgn
        assert {k:F(v) for k,v in ci.items() if v}=={k:2*den*v for k,v in al.items() if v}
        for j,jr in enumerate(r['joint_questions'],1):
            name,sgn=slots[j]; q=qq[jr]
            assert q['kind']=='row_joint' and (q['row'],q['slot'],q['variable'],q['sign'])==(ri,j,name,sgn)
            assert q['answers']==[[e,k] for e in (0,1) for k in (0,1,2)]
            binding(jr,mask(jr,lambda a:a[0]==1),oldid[name] if name else 0,2 if name else 1)
            pr=pairs[d,j]
            for k in range(3): binding(jr,mask(jr,lambda a,k=k:a[1]==k),pr,mask(pr,lambda a,k=k:(0 if a==0 else 1 if a==1 else 2)==k))
            expectedorth.append([dr,jr,j,sgn]); slots_count+=1
        expectedorth.append([dr,globals['X'],0,d-1])
    assert expectedorth==s['orthogonality']
    assert bactual==bexpect
    # Exhaust the scalar character orthogonality; this verifies the normalization
    # of every column partition without allocating giant block matrices.
    for d in sorted({r['d'] for r in s['rows']}):
        assert all(sum((-1)**((t&j).bit_count()%2) for j in range(d))==(d if t==0 else 0) for t in range(d))
    for q,m,r,mr in s['bindings']:
        assert 0<=int(m)<1<<len(qq[q]['answers']) and 0<=int(mr)<1<<len(qq[r]['answers'])
    # Every old binary-to-binary direct test is synchrony; old external tests
    # must reproduce the retained negative event records exactly.
    with zipfile.ZipFile(OUT.parent/'out5'/'functional-small.zip') as z:
        import struct
        ev=[]
        for key,num in struct.iter_unpack('<Qq',z.read('coefficients.bin')):
            qa,a=divmod(key>>32,16); qb,b=divmod(key&0xffffffff,16)
            if qs['A'][qa]['kind'] in ('fms','refined') or qs['B'][qb]['kind'] in ('fms','refined'):
                assert num<0; ev.append([qa,a,qb,b])
    assert ev==s['old_forbidden']
    return {'all_serialized_bindings_checked':len(s['bindings']),'all_column_graph_tests_checked':slots_count,'all_rational_row_expansions_checked':len(s['rows']),'scalar_pair_contexts_checked':len(pairs),'spectator_masks_checked':True,'character_orthogonality_exact':True,'old_forbidden_events_checked':len(ev),'questions':Q,'max_answers':256}

def matrices():
    spec=importlib.util.spec_from_file_location('prior_exact_matrix_evaluator',OUT.parent/'out5'/'evaluate.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def cutoff(N,s,qs,core,m):
    start=time.perf_counter(); old=s['old_questions']; oldqs={side:a[:old] for side,a in qs.items()}
    maps,mats,amp,norm,ws=m.model(N,core,oldqs)
    oldid={q['name']:q['id'] for q in oldqs['A']}; I=mats[int(maps['A'][0])]
    def effect(name): return I if name=='I' else mats[int(maps['A'][16*oldid[name]+1])]
    def probability(A,B):
        v=[0]*8
        for i,j in (A if len(A)<len(B) else B):
            if (i,j) in A and (j,i) in B:
                z=m.fm(m.fm(A[i,j],B[j,i]),m.fm(amp[i],amp[j]))
                v=[a+b for a,b in zip(v,z)]
        return tuple(F(a,m.L*m.L*int(norm)) for a in v)
    for qa,a,qb,b in s['old_forbidden']:
        assert not any(probability(mats[int(maps['A'][16*qa+a])],mats[int(maps['B'][16*qb+b])]))
    binary_sync=0
    for q in oldqs['A']:
        if q['kind']=='binary':
            qid=q['id']; E=mats[int(maps['A'][16*qid+1])]; C=mats[int(maps['B'][16*qid])]; assert not any(probability(E,C)); assert not any(probability(mats[int(maps['A'][16*qid])],mats[int(maps['B'][16*qid+1])]))
            binary_sync+=2
    # Reconstruct each row's literal signed-column sum, then contract its
    # D_0/R event by expanded matrix entries of the actual columns.
    failed=[]; total=F(0); iso=None
    for row in s['rows']:
        coeff=collections.Counter()
        for name,sgn in row['slots']:
            coeff['I']+=sgn if name is None else -sgn
            if name is not None: coeff[name]+=2*sgn
        V=m.add(*[(effect(k),v) for k,v in coeff.items() if v])
        if not V: continue
        failed.append(row['name'])
        vv=[0]*8
        for (i,j),v in V.items():
            z=m.fm(m.fm(m.conj(v),v),m.fm(amp[j],amp[j]))
            vv=[a+b for a,b in zip(vv,z)]
        event=tuple(F(x,m.L*m.L*int(norm)*row['d']**3) for x in vv)
        assert not any(event[1:]); total+=event[0]
        if row['name']=='R_isometry': iso=(row,V,event[0])
    assert failed==['R_isometry']; row,V,event=iso
    iso_den=row['den']; expected={}
    for j,w in enumerate(ws):
        if w.count('A')==N:
            expected[j,9*len(ws)+j]=m.real(-m.L*2*iso_den)
            expected[9*len(ws)+j,j]=m.real(-m.L*2*iso_den)
    assert V==expected
    top=sum((F(1,4**w.count('A')) for w in ws if w.count('A')==N),F(0))/sum((F(1,4**w.count('A')) for w in ws),F(0))
    assert event==total==top/F(9600)
    Q=s['questions_per_party']; loss=total/(Q*Q)
    result={'N':N,'old_local_dimension':len(amp),'local_dimension':256*len(amp),'top_shell_mass':str(top),'failed_rows':failed,'old_forbidden_probabilities_exactly_zero':len(s['old_forbidden']),'old_binary_mismatch_events_exactly_zero':binary_sync,'rows_contracted':len(s['rows']),'unnormalized_event_loss':str(total),'uniform_game_loss':str(loss),'uniform_game_score':str(1-loss),'row_ancilla_dimension':row['d'],'row_denominator':row['den'],'peak_working_set_bytes':m.peak_memory(),'elapsed_seconds':time.perf_counter()-start}
    (OUT/f'check-N{N}.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    s,qs,core=read_game(); audit_result=audit(s,qs,core)
    (OUT/'audit.json').write_text(json.dumps(audit_result,indent=2)); print(json.dumps(audit_result),flush=True)
    m=matrices()
    for N in map(int,sys.argv[1:] or ['0','1','2']):
        cutoff(N,s,qs,core,m); gc.collect()

