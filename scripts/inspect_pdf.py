from pathlib import Path
import fitz
from PIL import Image, ImageOps, ImageDraw
root=Path(__file__).resolve().parents[1]
out=root/'validation'/'precision_revision_pages'
out.mkdir(exist_ok=True)
pdf=fitz.open(root/'paper'/'main.pdf')
thumbs=[]
for i,page in enumerate(pdf):
    pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
    pix.save(out/f'page-{i+1:02d}.png')
    im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    im.thumbnail((459,594))
    tile=Image.new('RGB',(479,624),'#cccccc')
    tile.paste(im,((479-im.width)//2,20))
    ImageDraw.Draw(tile).text((10,5),f'Page {i+1}',fill='black')
    thumbs.append(tile)
for batch in range(0,len(thumbs),4):
    panel=Image.new('RGB',(958,1248),'white')
    for n,tile in enumerate(thumbs[batch:batch+4]):panel.paste(tile,((n%2)*479,(n//2)*624))
    panel.save(out/f'contact-{batch//4+1}.png')
print('Rendered',len(pdf),'pages to',out)
