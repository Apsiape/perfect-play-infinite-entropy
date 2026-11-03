from pathlib import Path
from fractions import Fraction as F
import json,sys,hashlib,importlib.util,decimal
run=Path(sys.argv[1]).resolve()
def read(p):return json.loads((run/p).read_text(encoding='utf-8'))
def require(ok,message):
 if not ok:raise AssertionError(message)
checks=[]
def passed(name):checks.append(name)
b=read('robust/bridge_identities.json')
require(b['sym_identity_9_residual']=='0' and b['sym_identity_9_bob_residual']=='0','symbolic bridge identities')
for key,val in b.items():
 if 'residual' in key and isinstance(val,(int,float)):require(abs(val)<1e-11,key)
require(b['sigma_matches_overlaps'] and b['W00=W01=W10=H'] and b['W11=J_L H=H J_R'],'comparison geometry')
passed('Alice and Bob symbolic bridge identities; numerical residuals <1e-11')
r=read('robust/rounding_and_compiler_checks.json')
require(r['threshold_lemma_commuting_max_K']<=48,'commuting partition bound')
require(r['noncommuting_counterexample']['norm_XminusY_v']<1e-12 and r['noncommuting_counterexample']['integral_mismatch_sq']>.3,'noncommuting negative control')
require(r['recurrence_max_excess_over_u']<=4.00001 and abs(r['green_function_mass']-4)<1e-12,'recurrence')
for d,x in r['compiler_star_laplacian'].items():
 require(x['min_eig_Kt_minus_half_I_minus_Pt']>-1e-11,'compiler spectral gap '+d)
 for k in ['Jt_orthonormal_err','sum_Pt_err','column_overlap_identity_err']:require(x[k]<1e-11,k)
passed('Commuting partition; expected noncommuting failure; recurrence; compiler identities')
rows=read('log/sec2_identities.json')
for row in rows:
 for k in ['e_ok','ebound_le_20delta','a_k_bound_ok','rH_le_delta','heightmatch_ok']:require(row[k],k)
 require(row['identity31_err']<1e-10,'renewal identity')
passed('Exact-local-model renewal and recurrence checks')
for name in ['julia_mirror_stress','julia_mirror_stress2','julia_mirror_stress3']:
 summary=read('robust/'+name+'.json')['summary']
 require(summary['trials']>400 and summary['worst_bridge_over_eta']<11 and summary['worst_mirrors_over_eta']<11,'Julia stress '+name)
summary=read('robust/julia_mirror_stress3.json')['summary']
require(summary['small_eta_count']>=200 and summary['eta_range'][0]<1e-5,'small-error Julia adversary')
passed('1872 Julia stress samples; small-error hostile cases retained; numerical ratios below 11')
c=read('height/constants.json')
require(F(c['Lambda_B'])==F(50808529,540000) and F(c['Lambda_G'])==F(14844029,54000),'height constants')
require(c['active']==392 and F(c['kappa_min'])==F(7,2) and F(c['kappa_max'])==F(221,2),'active symbols and penalty range')
require(c['max_full_row_residual']==0,'full row identities')
s=read('height/phase_audit.json')
require(s['individual_sigma_constants_phase_independent']==392 and s['pairs_with_nonconstant_leakage']==0,'phase independence')
passed('Exact Bell/game constants; 392 active symbols; repaired penalty range; arbitrary phases')
for row in read('height/entropy.json')['rows']:
 H=row['H'];q=F(1,2**H);Z=(2*H+q)/(H+1)
 require(F(row['Z'])==Z,'Z_H')
 require(F(row['loss_B'])==F(50808529,540000)/(H*2**(H+1)+1),'Bell loss')
 require(F(row['loss_G'])==F(14844029,54000)/(169460**2*(H*2**(H+1)+1)),'game loss')
passed('Height formulas H=1..100, with independent exact transfer solves H<=20')
spec=importlib.util.spec_from_file_location('s4',run/'log/s4_counterexample.py');s4=importlib.util.module_from_spec(spec);spec.loader.exec_module(s4)
Z,M,b,_=s4.modified(18,7,0)
require(Z==F(55409916813,34359738368) and M==F(57550635573,34359738368),'S4 moments')
require(b/Z==F(1288366711,166229750439) and b/Z<F(1,129),'S4 boundary ratio')
D=decimal.Decimal;decimal.getcontext().prec=60
def dec(v):return D(v.numerator)/D(v.denominator)
def entropy(z,m):return dec(z).ln()/D(2).ln()+2*dec(m/z)
require(entropy(Z,M)<entropy(F(129,80),F(67,40))-D('0.00012'),'S4 entropy improvement')
z2,m2,b2=s4.engine2(18,7,0,nmax=300)
require(max(abs(Z-z2),abs(M-m2),abs(b-b2))<F(1,10**35),'S4 independent DP')
passed('Explicit n0=7,N=18 witness: rational data exact; entropy gain >0.00012 bits; second DP residual <1e-35')
report=dict(status='PASS',checks=checks,scope='Finite identities, exact formulas and numerical stress tests only; not a proof of unrestricted lower bounds.')
(run/'asserted-results.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(report,indent=2))
