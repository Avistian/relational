"""Rebuild portable notebook figures from lesson SVGs and measured JSON.

Run from repo root: .venv/bin/python labs/_figures_l047.py
Authoring dependencies: Node, system librsvg/cairo, matplotlib (no browser).
Students need none of the render tooling: PNGs are embedded as inline markdown images.
"""
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from xml.etree import ElementTree as ET
from relkit.saint_report import save_reference_figures

HERE = Path(__file__).resolve().parent
DEST = HERE / 'figures/l047'
STATES = {'feature-axis': 'axes-640-1', 'row-axis': 'axes-640-2',
          'architecture': 'architecture-640-4', 'companion': 'context-640-3',
          'views': 'views-640-true-0.2', 'contrast': 'contrast-640-0.7'}


def render(source, dest):
    r, ca, g = (c.CDLL(name) for name in ('librsvg-2.so.2', 'libcairo.so.2', 'libgobject-2.0.so.0'))
    class Rect(c.Structure):
        _fields_ = [(k, c.c_double) for k in ('x', 'y', 'width', 'height')]
    r.rsvg_handle_new_from_data.argtypes = [c.c_char_p, c.c_size_t, c.c_void_p]
    r.rsvg_handle_new_from_data.restype = c.c_void_p
    r.rsvg_handle_render_document.argtypes = [c.c_void_p, c.c_void_p, c.POINTER(Rect), c.c_void_p]
    r.rsvg_handle_render_document.restype = c.c_int
    ca.cairo_image_surface_create.argtypes = [c.c_int, c.c_int, c.c_int]
    ca.cairo_image_surface_create.restype = c.c_void_p
    ca.cairo_create.argtypes = [c.c_void_p]
    ca.cairo_create.restype = c.c_void_p
    ca.cairo_scale.argtypes = [c.c_void_p, c.c_double, c.c_double]
    ca.cairo_set_source_rgb.argtypes = [c.c_void_p, c.c_double, c.c_double, c.c_double]
    ca.cairo_paint.argtypes = [c.c_void_p]
    ca.cairo_surface_write_to_png.argtypes = [c.c_void_p, c.c_char_p]
    ca.cairo_destroy.argtypes = [c.c_void_p]
    ca.cairo_surface_destroy.argtypes = [c.c_void_p]
    g.g_object_unref.argtypes = [c.c_void_p]
    data = source.read_bytes()
    _, _, w, h = map(float, ET.fromstring(data).attrib['viewBox'].split())
    handle = r.rsvg_handle_new_from_data(data, len(data), None)
    assert handle, source
    surface = ca.cairo_image_surface_create(0, int(w * 2), int(h * 2))
    ctx = ca.cairo_create(surface)
    try:
        ca.cairo_set_source_rgb(ctx, 245/255, 248/255, 250/255)
        ca.cairo_paint(ctx)
        ca.cairo_scale(ctx, 2., 2.)
        assert r.rsvg_handle_render_document(handle, ctx, c.byref(Rect(0, 0, w, h)), None)
        assert ca.cairo_surface_write_to_png(surface, str(dest).encode()) == 0
    finally:
        ca.cairo_destroy(ctx)
        ca.cairo_surface_destroy(surface)
        g.g_object_unref(handle)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='l047-figures-') as directory:
        subprocess.run(['node', str(HERE / '_viz_check_l047.js'), directory], check=True)
        for name, state in STATES.items():
            render(Path(directory) / f'{state}.svg', DEST / f'{name}.png')
    save_reference_figures(json.loads((HERE / '_verify_l047_results.json').read_text()), DEST)
    sources = ['../assets/saint-viz.js', '_viz_check_l047.js', '_verify_l047_results.json',
               'relkit/saint_report.py', '_figures_l047.py']
    manifest = {'command': '.venv/bin/python labs/_figures_l047.py', 'diagram_states': STATES,
                'note': 'Synthetic diagrams and separately labeled measured author reference plots.',
                'source_sha256': {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in sources},
                'png_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DEST.glob('*.png'))}}
    (DEST / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Wrote {len(STATES) + 3} portable figures to {DEST}')


if __name__ == '__main__':
    main()
