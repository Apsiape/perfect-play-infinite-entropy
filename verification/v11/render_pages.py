"""Render every manuscript page and contact sheets; optionally identify changed pages."""
from pathlib import Path
import argparse,json,subprocess,hashlib
from PIL import Image,ImageDraw
from pypdf import PdfReader
parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--baseline',type=Path);args=parser.parse_args()
root=Path(__file__).resolve().parents[2];pdf=root/'paper/main.pdf';qa=args.output;qa.mkdir(parents=True,exist_ok=True)
reader=PdfReader(pdf);subprocess.run(['pdftoppm','-r','85','-png',str(pdf),str(qa/'page')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
imgs=sorted(qa.glob('page-*.png'));assert len(imgs)==len(reader.pages)
for i in range(0,len(imgs),4):
 sheet=Image.new('RGB',(2000,2860),'#b8b8b8');draw=ImageDraw.Draw(sheet)
 for j,f in enumerate(imgs[i:i+4]):
  im=Image.open(f).convert('RGB');im.thumbnail((980,1395));x=(j%2)*1000+(1000-im.width)//2;y=(j//2)*1430+28
  sheet.paste(im,(x,y));draw.text(((j%2)*1000+20,(j//2)*1430+8),f'PAGE {i+j+1}',fill='black')
 sheet.save(qa/f'contact-{i//4+1:02}.jpg',quality=92)
changed=[]
if args.baseline:
 for i,f in enumerate(imgs,1):
  old=args.baseline/f.name
  if not old.exists() or hashlib.sha256(f.read_bytes()).digest()!=hashlib.sha256(old.read_bytes()).digest():changed.append(i)
index=[dict(page=n,start=page.extract_text()[:190]) for n,page in enumerate(reader.pages,1)]
(qa/'page-text-index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(pages=len(imgs),contact_sheets=(len(imgs)+3)//4,changed_pages=changed)))
