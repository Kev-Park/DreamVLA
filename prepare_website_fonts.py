"""Freeze the website's open-license variable fonts into printable static fonts."""
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parent / 'website_theme' / 'fonts'
VARIANTS = [
    ('DM_Sans_variable.ttf', 'DMSans-Regular.ttf', {'wght': 400, 'opsz': 14}),
    ('DM_Sans_variable.ttf', 'DMSans-Medium.ttf', {'wght': 500, 'opsz': 14}),
    ('Manrope_variable.ttf', 'Manrope-Semibold.ttf', {'wght': 650}),
    ('Manrope_variable.ttf', 'Manrope-ExtraBold.ttf', {'wght': 800}),
]

for source, destination, axes in VARIANTS:
    with TTFont(ROOT / source) as font:
        static = instantiateVariableFont(font, axes, inplace=False)
        assert 'fvar' not in static, f'Unpinned axes in {destination}'
        static.save(ROOT / destination)
    print(destination)
