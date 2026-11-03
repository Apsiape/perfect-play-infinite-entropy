"""Exact local-boundary phase audit and exact transfer/entropy formulas."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,zipfile,math,time,threading
from collections import Counter,defaultdict
from fractions import Fraction as F
import sympy as s
import independent as m
z,t=s.symbols('z t',nonzero=True);I=s.I
R=s.Rational
active=[k for k,v in json.loads((m.OUT/'symbol_constants.json').read_text()).items() if F(v['sigma_per_boundary'])]
sc=json.loads((m.OUT/'symbol_constants.json').read_text())
with zipfile.ZipFile(m.ROOT/'out6/game.zip') as f:g=json.loads(f.read('game.json'))

def conjugate(x):return s.expand(s.conjugate(x).subs({s.conjugate(z):1/z,s.conjugate(t):1/t}))

def atom(token):
    inv=token.endswith('~');k=token.rstrip('~')
    if k=='a_plus':a=s.Matrix([[0,I],[1,0]])
    elif k=='a_minus':a=s.Matrix([[0,-I],[1,0]])
    elif k=='s_root':a=s.diag(1,-1)
    elif k=='s_m2_00':a=s.diag(-1,1)
    elif k=='u0_c':a=s.diag(z,1)
    elif k=='u2_c':a=s.diag(1,t)
    else:raise ValueError(k)
    return a.applyfunc(conjugate).T if inv else a

from functools import lru_cache
@lru_cache(None)
def word(idx):
    a=s.eye(2)
    for token in m.LABELS[idx]:a=a*atom(token)
    return a

@lru_cache(None)
def projection(name):
    if name=='I':return s.eye(6)
    if name.startswith('F') and len(name)==2:
        a=s.zeros(6);i=2*(int(name[1])-1);a[i,i]=a[i+1,i+1]=1;return a
    if name.startswith('S') and len(name)==3:return projection('F'+name[1])+projection('F'+name[2])
    if name.startswith('K_'):
        other=name[2:];return projection('S'+other[1:3])-projection(other)
    if name[0] in 'GH':
        pair,idx=name[1:].split('_');i,j=2*(int(pair[0])-1),2*(int(pair[1])-1)
        u=word(int(idx));a=s.zeros(6)
        x,y,b=(R(1,2),R(1,2),R(1,2)) if name[0]=='G' else (R(4,5),R(1,5),R(2,5))
        a[i:i+2,i:i+2]=s.eye(2)*x;a[j:j+2,j:j+2]=s.eye(2)*y
        a[i:i+2,j:j+2]=u*b;a[j:j+2,i:i+2]=u.applyfunc(conjugate).T*b
        return a
    if name.startswith('D_'):
        r=m.DROWS[name];return sum((-R(v)*projection(k)/R(r['terms'][name]) for k,v in r['terms'].items() if k!=name),s.zeros(6))
    raise ValueError(name)

def phases():
    start=time.perf_counter();inside=[0,2,4];outside=[1,3,5]
    vs={a:(2*projection(a)-s.eye(6)).applyfunc(s.expand) for a in active}
    # First certify every individual synchrony constant at arbitrary unit phases.
    for a in active:
        p=projection(a);sq=sum((s.expand(p[i,j]*conjugate(p[i,j])) for i in outside for j in inside),s.S(0))
        assert s.expand(2*sq/75-R(sc[a]['sigma_per_boundary']))==0,a
    weights=defaultdict(F);shortcut=F(0)
    aset=set(active)
    for row in g['rows']:
        counts=Counter(a for a,sg in row['slots']);d=row['d']
        for a,ca in counts.items():
          for b,cb in counts.items():
            weight=F(ca*cb,d*d)
            if a in aset and b in aset:
                if a!=b:weights[tuple(sorted((a,b)))]+=weight
            elif (a in aset)!=(b in aset):shortcut+=weight*2*F(sc[a if a in aset else b]['sigma_per_boundary'])
    total=R(shortcut.numerator,shortcut.denominator);nonconstant=0
    for (a,b),w in weights.items():
        v=vs[a]*vs[b]
        p=s.expand(sum((v[i,j]*conjugate(v[i,j]) for i in outside for j in inside),s.S(0))/75)
        if p.free_symbols:nonconstant+=1
        total+=R(w.numerator,w.denominator)*p
    total=s.expand(total)
    assert total==R(259423,90000),total
    result={'individual_sigma_constants_phase_independent':len(active),'weighted_active_pairs':len(weights),'pairs_with_nonconstant_leakage':nonconstant,'row_columns_constant_exact':str(total),'arbitrary_phases':['z','t'],'seconds':time.perf_counter()-start}
    print(json.dumps(result,indent=2),flush=True);(m.OUT/'phase_audit.json').write_text(json.dumps(result,indent=2))

def entropy():
    rows=[]
    for H in range(1,101):
        q=F(1,2**H);Z=(2*H+q)/(H+1)
        en=(F(2,3)*H*(H-1)+q*H*(H+5)/6)/(2*H+q)
        # Solve the independent finite transfer recurrences, exactly.
        if H<=20:
            a=s.eye(H);rhs=s.ones(H,1)
            for h in range(H):
                if h:a[h,h-1]=-1
                if h<H-1:a[h,h+1]=-R(1,4)
            cont=a.inv()*rhs
            moment=a.inv()*s.Matrix([cont[h+1]/4 if h<H-1 else 0 for h in range(H)])
            assert 1+cont[0]/4==R(Z.numerator,Z.denominator)
            assert (cont[0]+moment[0])/4/(1+cont[0]/4)==R(en.numerator,en.denominator)
        S=math.log2(75)-1.6+math.log2(float(Z))+2*float(en)
        lb=F(50808529,540000)/(H*2**(H+1)+1);lg=F(14844029,54000)/(169460**2*(H*2**(H+1)+1))
        rows.append({'H':H,'Z':str(Z),'mean_n':str(en),'S_B':S,'loss_B':str(lb),'loss_G':str(lg)})
    result={'rows':rows,'S_bound_constant':math.log2(150)-1.6-2/3,'B_rounding_H2':rows[1]['S_B']+2/3*math.log2(float(F(rows[0]['loss_B']))),'G_rounding_H2':rows[1]['S_B']+8+2/3*math.log2(float(F(rows[0]['loss_G']))),'H_ge_3_B_bound':math.log2(150)-1.6-2/3+2/3*math.log2(float(F(50808529,1080000))),'H_ge_3_G_bound':math.log2(150)-1.6-2/3+8+2/3*math.log2(float(F(14844029,108000*169460**2)))}
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2),flush=True)
    (m.OUT/'entropy.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':phases();entropy()
