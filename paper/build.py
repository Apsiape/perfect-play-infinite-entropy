"""Build with pdflatex and BibTeX; fail on final warnings or box defects."""
from pathlib import Path
import subprocess,shutil,re,sys
p=Path(__file__).resolve().parent
b=p/'build';b.mkdir(exist_ok=True)
def run(args):
 r=subprocess.run(args,cwd=p,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (b/(args[0]+'.stdout')).write_text(r.stdout,encoding='utf-8')
 if r.returncode: print(r.stdout[-6000:]);sys.exit(r.returncode)
run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'])
run(['bibtex','build/main'])
for _ in range(3): run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','main.tex'])
log=(b/'main.log').read_text(errors='replace')
bad=[l for l in log.splitlines() if re.search(r'Warning:|Overfull|Underfull|undefined references|undefined citations',l,re.I)]
if bad: print('FINAL BUILD ISSUES:\n'+'\n'.join(bad));sys.exit(2)
shutil.copy2(b/'main.pdf',p/'main.pdf')
print('Clean pdflatex + BibTeX build.')
