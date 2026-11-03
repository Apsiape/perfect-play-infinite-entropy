"""Build with pdflatex and BibTeX; fail on final warnings or box defects."""
from pathlib import Path
import subprocess,shutil,re,sys,tempfile
p=Path(__file__).resolve().parent
cache=p/'build';cache.mkdir(exist_ok=True)
b=Path(tempfile.mkdtemp(prefix='clean-',dir=cache))
relative=b.relative_to(p).as_posix()
def run(args):
 r=subprocess.run(args,cwd=p,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (b/(args[0]+'.stdout')).write_text(r.stdout,encoding='utf-8')
 if r.returncode: print(r.stdout[-6000:]);sys.exit(r.returncode)
run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory='+relative,'main.tex'])
run(['bibtex',relative+'/main'])
for _ in range(3): run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','-output-directory='+relative,'main.tex'])
log=(b/'main.log').read_text(errors='replace')
bad=[l for l in log.splitlines() if re.search(r'Warning:|Overfull|Underfull|undefined references|undefined citations',l,re.I)]
if bad: print('FINAL BUILD ISSUES:\n'+'\n'.join(bad));sys.exit(2)
shutil.copy2(b/'main.pdf',p/'main.pdf')
shutil.copy2(b/'main.log',cache/'main.log')
(cache/'latest-build.txt').write_text(relative+'\n',encoding='utf-8')
print('Clean pdflatex + BibTeX build.')
