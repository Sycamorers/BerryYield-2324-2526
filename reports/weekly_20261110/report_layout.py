"""Retain the supplied laboratory Word styling and insert undistorted figures."""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import sys
from lxml import etree
import struct
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from reports import build_weekly_20261027 as base

class ReportWriter(base.TemplateWriter):
    def __init__(self, destination):
        packaged_template=Path(__file__).parent/"format_template.docx"
        if packaged_template.exists():base.TEMPLATE=packaged_template
        super().__init__()
        self.destination=Path(destination)

    def picture(self,path,width=6.5):
        path=Path(path)
        if path.suffix.lower() not in {".png"}:
            converted=Path(__file__).parent/"figures"/(path.stem+".png")
            Image.open(path).save(converted)
            path=converted
        super().picture(path)
        data=path.read_bytes()
        pixel_width,pixel_height=struct.unpack('>II',data[16:24])
        cx=round(width*914400);cy=round(cx*pixel_height/pixel_width)
        for extent in self.nodes[-1].xpath('.//wp:extent | .//a:xfrm/a:ext',namespaces=base.NS):
            extent.set('cx',str(cx));extent.set('cy',str(cy))
        prop=base.property_node(self.nodes[-1])
        base.flag(prop,'keepNext')

    def paragraph(self,text,kind='body',new_page=False):
        p=super().paragraph(text,kind,new_page)
        if kind=='caption':
            for sz in p.xpath('w:r/w:rPr/w:sz',namespaces=base.NS):sz.set(base.q('val'),'18')
        return p

    def table(self, headers, rows, widths=None):
        # Separate adjacent OOXML tables so Word preserves each table header.
        if self.nodes and self.nodes[-1].tag == base.q('tbl'):
            p=etree.Element(base.q('p'));prop=etree.SubElement(p,base.q('pPr'))
            etree.SubElement(prop,base.q('spacing'),{base.q('before'):'0',base.q('after'):'0',base.q('line'):'20',base.q('lineRule'):'exact'})
            self.nodes.append(p)
        super().table(headers,rows,widths)
        for p in self.nodes[-1].xpath('.//w:tc/w:p',namespaces=base.NS):
            prop=base.property_node(p)
            spacing=prop.find('w:spacing',base.NS)
            if spacing is None:spacing=etree.SubElement(prop,base.q('spacing'))
            for name,value in {'before':'0','after':'0','line':'264','lineRule':'auto'}.items():spacing.set(base.q(name),value)

    def cover(self):
        super().cover()
        cells=self.nodes[4].xpath('.//w:tc',namespaces=base.NS)
        base.set_text(cells[3].find('w:p',base.NS),'From: 10.28.2026    To: 11.10.2026')
        base.set_text(cells[7].find('w:p',base.NS),'11.10.2026')

    def save(self):
        base.DEST=self.destination
        return super().save()


def update_page_count(path,pages):
    with ZipFile(path) as archive:parts={n:archive.read(n) for n in archive.namelist()}
    root=etree.fromstring(parts['docProps/app.xml'])
    item=root.find('{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Pages')
    if item is not None:item.text=str(pages)
    parts['docProps/app.xml']=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(path,'w',ZIP_DEFLATED) as archive:
        for name,data in parts.items():archive.writestr(name,data)
