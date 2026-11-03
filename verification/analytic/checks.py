"""Turn 7 checks: constant-memory scalar sums and a literal-table audit.
No imports of prior producer/evaluator code. Run python -B checks.py.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json, math, struct, zipfile, time, hashlib
from pathlib import Path
from fractions import Fraction as Q
from collections import defaultdict, Counter
HERE=Path(__file__).resolve().parent
BASE=HERE.parent

def audit_table():
    with zipfile.ZipFile(BASE/'out5/functional-small.zip') as z:
        qs=json.loads(z.read('questions.json'))
        data=z.read('coefficients.bin')
    direct=[]; pairs=defaultdict(dict); groups=Counter()
    for key,num in struct.iter_unpack('<Qq',data):
        ea,eb=key>>32,key&0xffffffff
        qa,qb=qs['A'][ea//16],qs['B'][eb//16]
        if qa['kind'] not in ('fms','refined') and qb['kind'] not in ('fms','refined'):
            continue
        assert num==-2048
        direct.append((ea,eb))
        aa,ab=qa['answers'][ea%16],qb['answers'][eb%16]
        ca=qs['A'][qa['coarse']] if qa['kind']=='refined' else qa
        cb=qs['B'][qb['coarse']] if qb['kind']=='refined' else qb
        sa=aa[0] if qa['kind']=='refined' else aa
        sb=ab[0] if qb['kind']=='refined' else ab
        if ca['kind']=='fms' and cb['kind']=='fms' and ca['tag'][0]=='T' and cb['tag'][0]=='U' and sa in ('+','-') and sb in ('+','-'):
            assert ca['g']==cb['g']
            x,y=ca['tag'][1],cb['tag'][1]
            assert (1 if sa=='+' else -1)*(1 if sb=='+' else -1)==-(-1)**(x*y)
            key2=(qa['id'],qb['id'],sa,sb)
            pairs[key2][(aa[1] if qa['kind']=='refined' else None,ab[1] if qb['kind']=='refined' else None)]=1
            groups[(ca['g'],qa['kind'],qb['kind'])]+=1
    assert len(direct)==6916
    for (a,b,sa,sb),bits in pairs.items():
        ka,kb=qs['A'][a]['kind'],qs['B'][b]['kind']
        expected={(i,j) for i in ((0,1) if ka=='refined' else (None,)) for j in ((0,1) if kb=='refined' else (None,))}
        assert set(bits)==expected
    coarse_counts=Counter(qs['A'][a]['g'] for a,b,sa,sb in pairs)
    assert coarse_counts=={1:96,2:96} # 48 contexts, two wrong signs
    assert sum(groups.values())==368
    return {'direct_records':len(direct),'potentially_nonzero_sign_records':sum(groups.values()),'coarse_sign_events_after_summing_labels':dict(coarse_counts),'groups':{str(k):v for k,v in groups.items()},'coefficient_sha256':hashlib.sha256(data).hexdigest()}

def f(j,N):
    if j<=0: return 1.0
    return max(0.0,1.0-(j/N)**2)

def fq(j,N):
    return Q(1) if j<=0 else max(Q(0),Q(1)-(j/N)**2)

def exact_loss(N):
    a=Q(1); z=Q(0); d1=Q(0); d2=Q(0)
    for n in range(2*N+2):
        fs=[fq(Q(2*n-r,2),N) for r in range(3)]
        z+=a*(fs[0]**2+8*fs[1]**2+16*fs[2]**2)/50
        d1+=a*(fs[0]-fs[1])**2
        d2+=a*(fs[1]-fs[2])**2
        a*=Q(2*n+1,2*n+4)
    loss=Q(12,25)*(d1+2*d2)/z
    # Independently sum the 48 actual two-sign contexts per bridge.
    direct=Q(0); a=Q(1)
    for n in range(2*N+2):
        for g,w in ((1,1),(2,2)):
            diff=fq(Q(2*n-g+1,2),N)-fq(Q(2*n-g,2),N)
            direct+=48*2*Q(w,200)*a*diff**2/z
        a*=Q(2*n+1,2*n+4)
    assert direct==loss
    return {'N':N,'normalization':str(z),'D1':str(d1),'D2':str(d2),'loss':str(loss)}

def statistics(N):
    a=1.0; zn=0.0; d1=0.0; d2=0.0; energy_entropy=0.0; hard_z=0.0; hard_m=0.0
    for n in range(2*N+2):
        fs=[f(n-r/2,N) for r in range(3)]
        for r,mult in ((0,1),(1,4),(2,4)):
            v=fs[r]
            if v:
                mass=a*mult*2**r*v*v/50
                zn+=mass
                energy_entropy+=mass*(2*n-r-2*math.log2(v))
        d1+=a*(fs[0]-fs[1])**2
        d2+=a*(fs[1]-fs[2])**2
        if n<=N:
            hard_z+=a; hard_m+=n*a
            if n==N: hard_a=a
        a*=(2*n+1)/(2*n+4)
    entropy=math.log2(150*zn)+energy_entropy/zn
    loss=(12/25)*(d1+2*d2)/zn
    hard_t=hard_a/hard_z
    hard_s=math.log2(hard_z)+2*hard_m/hard_z
    return {'N':N,'Z':zn,'entropy_bits':entropy,'bell_loss':loss,'scaled_loss':loss*N**2.5,'scaled_entropy':entropy/math.sqrt(N),'S_times_loss_to_one_fifth':entropy*loss**.2,'sharp_root_S_times_t_to_one_third':hard_s*hard_t**(1/3)}

def constants():
    cs=64/(45*math.sqrt(math.pi))
    cl=24/(25*math.sqrt(math.pi))
    q=169460
    return {'entropy_sqrtN':cs,'loss_N_to_minus_5_over_2':cl,'bell_entropy_loss_constant':cs*cl**.2,'game_entropy_loss_constant_if_review_holds':cs*(cl/q**2)**.2,'old_sharp_bell_constant':(2/math.sqrt(math.pi))*(1/(75*math.sqrt(math.pi)))**(1/3)}

def rank_one_check():
    # Exact Q(sqrt(2)) rank-one Born contractions, independently of (3.5).
    def add(z,w): return (z[0]+w[0],z[1]+w[1])
    def mul(z,w): return (z[0]*w[0]+2*z[1]*w[1],z[0]*w[1]+z[1]*w[0])
    cases=0
    for ff,fm in ((Q(1),Q(1)),(Q(3,4),Q(5,8)),(Q(1),Q(0)),(Q(0),Q(2,3))):
        for x in (0,1):
            for y in (0,1):
                for a in (-1,1):
                    for b in (-1,1):
                        if a*b != -(-1)**(x*y): continue
                        va=[(Q(0),Q(0)) for _ in range(5)]
                        vb=va.copy()
                        va[0]=(Q(0),Q(1,2)); va[1+x]=(Q(0),Q(a,2))
                        vb[0]=(Q(0),Q(1,2)); vb[1]=(Q(b,2),Q(0)); vb[2]=(Q(b*(-1)**y,2),Q(0))
                        kk=[(ff,Q(0))]+[(Q(0),fm)]*4
                        amp=(Q(0),Q(0))
                        for i in range(5): amp=add(amp,mul(mul(va[i],kk[i]),vb[i]))
                        prob=mul(amp,amp)
                        assert prob==(Q((ff-fm)**2,4),Q(0))
                        cases+=1
    return {'exact_rank_one_Born_contractions':cases,'field':'Q(sqrt(2))'}

def peak_memory():
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
    api=ctypes.WinDLL('psapi'); api.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.c_void_p,wintypes.DWORD]
    assert api.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
    assert counters.PeakWorkingSetSize < 4*1024**3
    return int(counters.PeakWorkingSetSize)

start=time.perf_counter()
result={'rank_one_check':rank_one_check(),'table_audit':audit_table(),'exact_checks':[exact_loss(n) for n in (1,2,3,5)],'constants':constants(),'numerics':[statistics(n) for n in (10,100,1000,4000,10000,100000,1000000)]}
result['elapsed_seconds']=time.perf_counter()-start
result['peak_working_set_bytes']=peak_memory()
(HERE/'checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
