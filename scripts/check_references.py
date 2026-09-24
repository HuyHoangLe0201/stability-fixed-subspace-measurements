"""Read-only DOI metadata audit against Crossref; saves source metadata."""
from pathlib import Path
import re,json,urllib.request,concurrent.futures
root=Path(__file__).resolve().parents[1]
text=(root/'paper'/'main.tex').read_text(encoding='utf-8')
dois=re.findall(r'doi: ([^\s]+)\.',text)
def get(doi):
    try:
        req=urllib.request.Request('https://api.crossref.org/works/'+doi,headers={'User-Agent':'ManuscriptReferenceAudit/1.0'})
        m=json.load(urllib.request.urlopen(req,timeout=30))['message']
        return dict(doi=doi,title=m.get('title'),container=m.get('container-title'),
                    volume=m.get('volume'),issue=m.get('issue'),page=m.get('page'),date=m.get('published'),
                    source='https://api.crossref.org/works/'+doi,status='resolved')
    except Exception as e:return dict(doi=doi,status='unresolved',error=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(get,dois))
(root/'validation'/'reference_metadata.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for row in results:print(json.dumps(row,ensure_ascii=True))
