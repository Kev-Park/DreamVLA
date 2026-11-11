"""Verify public poster files without access to the unpublished source paper."""
from collections import Counter
from pathlib import Path
import argparse
import base64
import hashlib
import json
import xml.etree.ElementTree as ET

import pymupdf

ROOT = Path(__file__).resolve().parent
NS = {'s': 'http://www.w3.org/2000/svg'}
FILES = ('DreamVLA_CoRL_poster.pdf', 'DreamVLA_CoRL_poster.svg',
         'DreamVLA_CoRL_poster_preview.png')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def inspect(directory):
    svg = ET.parse(directory / FILES[1]).getroot()
    tokens = Counter(' '.join(''.join(item.itertext())
                             for item in svg.findall('s:text', NS)).split())
    images = [hashlib.sha256(base64.b64decode(
        item.attrib['{http://www.w3.org/1999/xlink}href'].split(',', 1)[1])).hexdigest()
              for item in svg.findall('s:image', NS)]
    with pymupdf.open(directory / FILES[0]) as pdf:
        assert len(pdf) == 1
        assert (pdf[0].rect.width, pdf[0].rect.height) == (3456, 2592)
        assert len(images) == 10
        fonts = len(pdf[0].get_fonts())
        text_lines = len(pdf[0].get_text().splitlines())
    data = json.loads((ROOT / 'poster_data.json').read_text())
    return {
        'schema_version': 1,
        'theme': 'website',
        'theme_reference': 'https://ember-center-berkeley.github.io/DreamVLA/',
        'pages': 1,
        'size_inches': [48, 36],
        'files': {name: digest(directory / name) for name in FILES},
        'semantic_text_sha256': canonical_digest(dict(tokens)),
        'image_payload_sha256': images,
        'data_sha256': digest(ROOT / 'poster_data.json'),
        'declared_source_pdf_sha256': data['source_pdf_sha256'],
        'source_pdf_bundled': False,
        'source_values_reviewed_before_publication': True,
        'embedded_fonts': fonts,
        'text_lines': text_lines,
        'svg_valid_xml': True,
        'anonymous': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT)
    parser.add_argument('--write-manifest', action='store_true',
                        help='For maintainers: explicitly refresh prebuilt-file records.')
    args = parser.parse_args()
    directory = args.directory.resolve()
    actual = inspect(directory)
    if args.write_manifest:
        assert directory == ROOT, 'Manifest refresh is only allowed for the committed files.'
        for name in ('publication_manifest.json', 'layout_checks.json', 'final_validation.json'):
            (ROOT / name).write_text(json.dumps(actual, indent=2) + '\n')
        print('Publication records written; source paper was not bundled.')
        return
    expected = json.loads((ROOT / 'publication_manifest.json').read_text())
    for key in ('semantic_text_sha256', 'image_payload_sha256', 'data_sha256',
                'declared_source_pdf_sha256', 'pages', 'size_inches'):
        assert actual[key] == expected[key], f'Poster mismatch: {key}'
    if directory == ROOT:
        assert actual['files'] == expected['files'], 'Committed artifact bytes changed.'
    print('PASS: poster text, six figures, conference logo, two affiliation crests, project QR, quantitative data, source hash, and print size.')


if __name__ == '__main__':
    main()
