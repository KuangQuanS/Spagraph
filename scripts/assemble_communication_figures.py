"""Replace communication panels in submission copies; preserve other panels."""
import argparse
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas


def bounds(page, region):
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    x0, y0, x1, y1 = region  # normalized, top-origin
    return x0*w, (1-y1)*h, x1*w, (1-y0)*h


def preserve(target, original, region):
    part = deepcopy(original)
    part.cropbox = RectangleObject(bounds(original, region))
    target.merge_page(part)


def panel(target, path, region):
    source = PdfReader(path).pages[0]
    x0, y0, x1, y1 = bounds(target, region)
    w, h = float(source.mediabox.width), float(source.mediabox.height)
    scale = min((x1-x0)/w, (y1-y0)/h)
    target.merge_transformed_page(source, Transformation().scale(scale).translate(
        x0+(x1-x0-w*scale)/2, y0+(y1-y0-h*scale)/2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--panel-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        'Fig3.pdf': ([(0,0,1,.276),(.455,.501,1,.728)],
                     [('fig3c.pdf',(0.012,.286,.445,.495)),
                      ('figureF_multimethod_rank_strip_preview.pdf',(.46,.289,.998,.495)),
                      ('fig3e.pdf',(.015,.520,.452,.714)),
                      ('fig3g.pdf',(.01,.737,.998,.996))],
                     [('c',.008,.283),('d',.450,.283),('e',.008,.520),('g',.008,.744)]),
        'FigS1.pdf': ([(0,0,1,.742)], [('figs1de.pdf',(.015,.754,.995,.992))], [])}
    for name, (regions, panels, labels) in specs.items():
        original = PdfReader(a.source_dir/name).pages[0]
        target = PageObject.create_blank_page(width=original.mediabox.width,
                                             height=original.mediabox.height)
        for region in regions:
            preserve(target, original, region)
        for filename, region in panels:
            panel(target, a.panel_dir/filename, region)
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=(float(target.mediabox.width),float(target.mediabox.height)))
        c.setFont('Times-Bold', 16)
        for label, x, y in labels:
            c.drawString(x*float(target.mediabox.width),(1-y)*float(target.mediabox.height),label)
        c.save()
        target.merge_page(PdfReader(buffer).pages[0])
        writer = PdfWriter()
        writer.add_page(target)
        with (a.output_dir/name).open('wb') as stream:
            writer.write(stream)


if __name__ == '__main__':
    main()
