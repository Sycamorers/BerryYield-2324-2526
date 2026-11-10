"""Verify that every new report figure preserves its original aspect ratio."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json

from lxml import etree
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "Jing-2026-1110-yield prediction new data-revised.docx"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def main():
    results = []
    with ZipFile(REPORT) as archive:
        document = etree.fromstring(archive.read("word/document.xml"))
        rels = etree.fromstring(archive.read("word/_rels/document.xml.rels"))
        targets = {node.get("Id"): node.get("Target") for node in rels}
        for blip in document.xpath(".//a:blip", namespaces=NS):
            rid = blip.get(f"{{{NS['r']}}}embed")
            if rid not in {f"rId{i}" for i in range(101, 107)}:
                continue
            inline = blip.xpath("ancestor::wp:inline", namespaces=NS)[0]
            extent = inline.xpath(".//wp:extent", namespaces=NS)[0]
            width_emu, height_emu = int(extent.get("cx")), int(extent.get("cy"))
            image_path = "word/" + targets[rid]
            with Image.open(BytesIO(archive.read(image_path))) as image:
                width_px, height_px = image.size
            error = abs(width_emu / height_emu - width_px / height_px)
            results.append({"image": image_path, "display_ratio": width_emu / height_emu,
                            "source_ratio": width_px / height_px, "ratio_error": error})
    assert len(results) == 6
    assert max(row["ratio_error"] for row in results) < 1e-5
    print(json.dumps({"passed": True, "figures": results}, indent=2))


if __name__ == "__main__":
    main()
