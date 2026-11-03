"""Search the extracted PDF text for visibly corrupted macro fragments."""
from pathlib import Path
import argparse,hashlib,json,re
from pypdf import PdfReader

def main():
 root=Path(__file__).resolve().parents[2]
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--pdf',type=Path,default=root/'paper/main.pdf')
 parser.add_argument('--output',type=Path,default=Path(__file__).with_name('visual-qa.json'))
 args=parser.parse_args();pages=[p.extract_text() or '' for p in PdfReader(args.pdf).pages]
 symptoms=('ext{','o0','heta','ewline','abla','ightarrow')
 hits=[]
 for number,content in enumerate(pages,1):
  for symptom in symptoms:
   for m in re.finditer(re.escape(symptom),content):
    hits.append(dict(page=number,symptom=symptom,context=content[max(0,m.start()-50):m.end()+70]))
 data=json.loads(args.output.read_text()) if args.output.exists() else {}
 data.update(pages=len(pages),pdf_sha256=hashlib.sha256(args.pdf.read_bytes()).hexdigest(),text_layer_search=dict(engine='pypdf PdfReader page.extract_text',pages_searched=len(pages),literal_symptoms=list(symptoms),hits=hits,cell_disagreement_pages=[i for i,t in enumerate(pages,1) if 'cell disagreement' in t],bottom_edge_pages=[i for i,t in enumerate(pages,1) if re.search(r'B\s*1\s*:\s*1\s*→\s*0',t)]))
 args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
 assert not hits,hits
 assert data['text_layer_search']['cell_disagreement_pages'],'Missing corrected cell-disagreement text'
 assert data['text_layer_search']['bottom_edge_pages'],'Missing corrected bottom edge arrow'
 print(json.dumps(data['text_layer_search'],indent=2))
if __name__=='__main__':main()
