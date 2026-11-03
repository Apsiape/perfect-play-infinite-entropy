"""Portable review suite, with fresh outputs and optional independent-group concurrency."""
from pathlib import Path
import argparse,subprocess,sys,os,json,time,tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output-parent',type=Path,default=BASE/'runs');parser.add_argument('--workers',type=int,default=1)
 args=parser.parse_args();assert 1<=args.workers<=5
 args.output_parent.mkdir(parents=True,exist_ok=True)
 target=Path(tempfile.mkdtemp(prefix=time.strftime('%Y%m%dT%H%M%S')+'-',dir=args.output_parent))
 groups=('robust','log','height','decoding','rounding')
 for group in groups:
  dest=target/group;dest.mkdir()
  for p in (BASE/group).glob('*.py'):(dest/p.name).write_text(p.read_text(encoding='utf-8-sig'),encoding='utf-8',newline='\n')
 jobs={
 'robust':[(x,[]) for x in ('bridge_identities.py','julia_mirror_stress.py','julia_mirror_stress2.py','julia_mirror_stress3.py','rounding_and_compiler_checks.py')],
 'log':[(x,[]) for x in ('sec1_exact.py','sec2_identities.py','inverse_bound.py','refined_upper.py','s4_witness.py','smoothing.py')],
 'height':[('independent.py',['atoms']),('independent.py',['constants']),('phase_and_entropy.py',[]),('audit_game.py',[])],
 'decoding':[(x,[]) for x in ('magic_rigidity_check.py','sign_mirror_check.py','sign_mirror_pure.py','repair_inequalities_check.py')],
 'rounding':[(x,[]) for x in ('check1_paths.py','check1b_exact.py','check2_banding.py','check3_arith.py')]}
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def run_group(group):
  rows=[];local_env=dict(env,ENTROPY_VERIFICATION_ROOT=str(ROOT/'verification'),ENTROPY_V11_OUTPUT=str(target/group))
  for name,script_args in jobs[group]:
   start=time.monotonic();out=target/group/(Path(name).stem+('_'+'_'.join(script_args) if script_args else '')+'.stdout.txt')
   print('RUN',group,name,*script_args,flush=True)
   with out.open('w',encoding='utf-8') as stream:
    try:code=subprocess.run([sys.executable,'-B',str(target/group/name),*script_args],cwd=target/group,env=local_env,stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
    except subprocess.TimeoutExpired:code=124
   row=dict(group=group,script=name,args=script_args,exit_code=code,seconds=round(time.monotonic()-start,3),stdout=out.relative_to(target).as_posix());rows.append(row);print(json.dumps(row),flush=True)
  return rows
 results=[]
 with ThreadPoolExecutor(max_workers=args.workers) as executor:
  pending=[executor.submit(run_group,group) for group in groups]
  for future in as_completed(pending):
   results.extend(future.result());(target/'run-summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8',newline='\n')
 if not any(row['exit_code'] for row in results):subprocess.run([sys.executable,'-B',str(BASE/'assert_results.py'),str(target)],check=True,env=env)
 print('RESULTS',target,flush=True);sys.exit(any(row['exit_code'] for row in results))
if __name__=='__main__':main()
