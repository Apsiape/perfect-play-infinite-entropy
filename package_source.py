"""Package version 1.1.0 and verify all checksums after fresh extraction."""
from pathlib import Path
import argparse,hashlib,json,zipfile,subprocess,sys,tempfile
R=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8',newline='\n')
def eligible(p):
 if not p.is_file():return False
 rel=p.relative_to(R)
 return not any(x in rel.parts for x in ('.git','__pycache__')) and rel.parts[0]!='work' and rel.parts[:2]!=('paper','build') and rel.as_posix() not in ('source.zip','SHA256SUMS','source-manifest.json')

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--report-dir',type=Path,default=R/'work/package-checks')
 args=parser.parse_args();out=args.report_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 v=R/'verification';vf=sorted(p for p in v.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts)
 (v/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(v).as_posix()+'\n' for p in vf),encoding='ascii',newline='\n')
 subprocess.run([sys.executable,'-B',str(v/'verify_hashes.py')],check=True)
 payload=sorted((p for p in R.rglob('*') if eligible(p)),key=lambda p:p.relative_to(R).as_posix())
 manifest=dict(version='1.1.0',description='Released manuscript source and finite verification bundle.',theorem='Both-party normal-spatial entropy price Theta(log(1/epsilon)); full actual-state decoding and rounding proofs.',contents=['Manuscript, LaTeX source and bibliography.','Portable finite verification scripts and one passing execution.','Original functional and game archives, evaluators and retained finite results.'],files=[dict(path=p.relative_to(R).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in payload],excludes=['Version-control history','Disposable manuscript build caches','Python bytecode caches','Packaging intermediates','Root archive, manifest and checksums from their own file list'],archive_checksums='Internal SHA256SUMS covers every payload file and manifest. External root SHA256SUMS additionally hashes source.zip.')
 mp=R/'source-manifest.json';dump(mp,manifest);payload.append(mp);payload.sort(key=lambda p:p.relative_to(R).as_posix())
 internal=''.join(sha(p)+'  '+p.relative_to(R).as_posix()+'\n' for p in payload)
 with zipfile.ZipFile(R/'source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
  for p in payload:
   i=zipfile.ZipInfo(p.relative_to(R).as_posix(),date_time=(2026,11,3,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o100644<<16;z.writestr(i,p.read_bytes())
  i=zipfile.ZipInfo('SHA256SUMS',date_time=(2026,11,3,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;i.external_attr=0o100644<<16;z.writestr(i,internal.encode('ascii'))
 (R/'SHA256SUMS').write_text(internal+sha(R/'source.zip')+'  source.zip\n',encoding='ascii',newline='\n')
 extracted=Path(tempfile.mkdtemp(prefix='source-unpack-',dir=out)).resolve()
 with zipfile.ZipFile(R/'source.zip') as z:
  assert z.testzip() is None
  for name in z.namelist():assert (extracted/name).resolve().is_relative_to(extracted),name
  z.extractall(extracted)
 for line in (extracted/'SHA256SUMS').read_text().splitlines():
  digest,name=line.split('  ',1);assert sha(extracted/name)==digest,name
 for row in json.loads((extracted/'source-manifest.json').read_text())['files']:
  p=extracted/row['path'];assert p.stat().st_size==row['size'] and sha(p)==row['sha256'],row['path']
 subprocess.run([sys.executable,'-B',str(extracted/'verification/verify_hashes.py')],check=True)
 assert b'\r' not in (R/'SHA256SUMS').read_bytes() and b'\r' not in (v/'SHA256SUMS').read_bytes()
 result=dict(status='PASS',version='1.1.0',archive_files=len(payload)+1,archive_size=(R/'source.zip').stat().st_size,archive_sha256=sha(R/'source.zip'),pdf_sha256=sha(R/'paper/main.pdf'),verification_hashes=len(vf),root_hashes=len(payload)+1,manifest_hashes=len(manifest['files']),lf_only_checksums=True,zip_crc_and_all_internal_hashes=True,fresh_extraction=str(extracted))
 dump(out/'package-check-3.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
