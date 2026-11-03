from pathlib import Path
import hashlib,sys
r=Path(__file__).resolve().parent
n=0
for line in (r/'SHA256SUMS').read_text().splitlines():
 digest,name=line.split('  ',1)
 f=r/name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
  print('FAIL',name);sys.exit(1)
 n+=1
print('PASS',n,'SHA-256 hashes')
