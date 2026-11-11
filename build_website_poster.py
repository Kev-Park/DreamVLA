"""Print-size DreamVLA poster, using original paper figures and vector text/charts."""
from pathlib import Path
import argparse
import base64
import hashlib
import html
import json
import math
import pymupdf as fitz

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--theme', choices=('website',), default='website')
parser.add_argument('--data', type=Path)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--source-pdf', type=Path, help='Optional paper PDF for hash verification; not bundled.')
args = parser.parse_args()
WEBSITE = args.theme == 'website'
DATA_PATH = (args.data or ROOT / 'poster_data.json').resolve()
DATA = json.loads(DATA_PATH.read_text(encoding='utf-8'))
if DATA.get('expert_ablation'):
    assert all(len(item['success']) == len(DATA['series']) for item in DATA['expert_ablation']['rows'])
OUTPUT = (args.output_dir or ROOT / 'build').resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
ASSETS = ROOT / 'assets'
round_count = DATA['dagger_rounds']
assert round_count == 3, 'This layout currently supports rounds 0 through 3.'
assert len(DATA['series']) == 4
assert len(DATA['scaling']) == 3, 'This layout currently supports three data budgets.'
for item in DATA['series']:
    assert len(item['values']) == round_count + 1
    assert all(0 <= value <= 100 for value in item['values'])
    assert 0 <= item['offline_control'] <= 100
for row in DATA['scaling']:
    assert len(row['success']) == len(DATA['series'])
    assert all(0 <= value <= 100 for value in row['success'])
SOURCE_PDF = None
SOURCE_SHA256 = None
if args.source_pdf or DATA.get('source_pdf'):
    SOURCE_PDF = args.source_pdf or Path(DATA['source_pdf'])
    if not SOURCE_PDF.is_absolute():
        SOURCE_PDF = DATA_PATH.parent / SOURCE_PDF
    SOURCE_PDF = SOURCE_PDF.resolve(strict=True)
    SOURCE_SHA256 = hashlib.sha256(SOURCE_PDF.read_bytes()).hexdigest()
    if DATA.get('source_pdf_sha256') and SOURCE_SHA256 != DATA['source_pdf_sha256']:
        raise ValueError('The source PDF changed. Review its contents before reusing these result values.')

def percentage(value):
    return f'{value:g}%'

W, H = 3456, 2592  # 48 x 36 inches, at 72 points/inch
NAVY = '#14263e'
INK = '#213349'
MUTED = '#536477'
TEAL = '#008e88'
ORANGE = '#d66b32'
BLUE = '#2674b5'
PURPLE = '#8055a7'
PALE = '#edf6f5'
GRID = '#dce4e9'
WHITE = '#ffffff'
PAPER = WHITE
ROW_FILL = '#f4f7f9'
if WEBSITE:
    NAVY = INK = '#202c30'
    MUTED = '#687378'
    TEAL = '#25776c'
    ORANGE, BLUE, PURPLE = '#ce8338', '#377db0', '#8970ab'
    PAPER, PALE, GRID, ROW_FILL = '#fafaf8', '#f0f2ee', '#dfe4e0', '#f7f9f5'
    font_root = ROOT / 'website_theme' / 'fonts'
    FONT_FILES = {'regular': font_root / 'DMSans-Regular.ttf',
                  'medium': font_root / 'DMSans-Medium.ttf',
                  'bold': font_root / 'Manrope-Semibold.ttf',
                  'display': font_root / 'Manrope-ExtraBold.ttf',
                  'fallback': font_root / 'DMSans-Regular.ttf'}
elif Path('C:/Windows/Fonts/arial.ttf').is_file():
    FONT_FILES = {'regular': Path('C:/Windows/Fonts/arial.ttf'),
                  'bold': Path('C:/Windows/Fonts/arialbd.ttf')}
else:
    FONT_FILES = {'regular': Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),
                  'bold': Path('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf')}
FONTS = {k: fitz.Font(fontfile=str(v)) for k, v in FONT_FILES.items()}
FONT_WEIGHTS = {'regular': 400, 'bold': 650 if WEBSITE else 700,
                'display': 800, 'medium': 500, 'fallback': 400}
STEP = TEAL if WEBSITE else ORANGE
doc = fitz.open()
page = doc.new_page(width=W, height=H)
for key, name in FONT_FILES.items():
    page.insert_font(fontname=key, fontfile=str(name))
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="48in" height="36in" viewBox="0 0 {W} {H}">']
svg.append('<defs><style>')
for key, path in FONT_FILES.items():
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    weight = FONT_WEIGHTS[key]
    family = f'DreamVLAPoster-{key}' if WEBSITE else 'DreamVLAPoster'
    svg.append(f'@font-face {{font-family: {family}; font-weight: {weight}; src: url(data:font/ttf;base64,{encoded}) format("truetype");}}')
svg.append('</style></defs>')
text_boxes = []
image_records = []

def rgb(color):
    return tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))

def rect(x, y, w, h, fill, stroke=None, sw=1, radius=0):
    if radius:
        r = min(radius, w / 2, h / 2)
        k = .5522847498
        shape = page.new_shape()
        shape.draw_line((x+r,y),(x+w-r,y))
        shape.draw_bezier((x+w-r,y),(x+w-r+k*r,y),(x+w,y+r-k*r),(x+w,y+r))
        shape.draw_line((x+w,y+r),(x+w,y+h-r))
        shape.draw_bezier((x+w,y+h-r),(x+w,y+h-r+k*r),(x+w-r+k*r,y+h),(x+w-r,y+h))
        shape.draw_line((x+w-r,y+h),(x+r,y+h))
        shape.draw_bezier((x+r,y+h),(x+r-k*r,y+h),(x,y+h-r+k*r),(x,y+h-r))
        shape.draw_line((x,y+h-r),(x,y+r))
        shape.draw_bezier((x,y+r),(x,y+r-k*r),(x+r-k*r,y),(x+r,y))
        shape.finish(color=rgb(stroke) if stroke else None, fill=rgb(fill) if fill else None, width=sw)
        shape.commit()
    else:
        page.draw_rect(fitz.Rect(x, y, x+w, y+h), color=rgb(stroke) if stroke else None,
                       fill=rgb(fill) if fill else None, width=sw)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def line(x1, y1, x2, y2, color=GRID, sw=2, dash=None):
    page.draw_line((x1,y1),(x2,y2), color=rgb(color),width=sw,
                   dashes='[10 7] 0' if dash else None)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="10 7"' if dash else '')+'/>')

def circle(x,y,r,fill):
    page.draw_circle((x,y),r,color=None,fill=rgb(fill))
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

def font_runs(s, font):
    runs = []
    for char in s:
        actual = font
        if WEBSITE and not FONTS[font].has_glyph(ord(char)):
            actual = 'fallback'
        if not FONTS[actual].has_glyph(ord(char)):
            raise ValueError(f'Missing glyph {char!r} in font {actual}')
        if runs and runs[-1][0] == actual:
            runs[-1] = (actual, runs[-1][1] + char)
        else:
            runs.append((actual, char))
    return runs

def measure(s, font, size):
    if not WEBSITE:
        return FONTS[font].text_length(s,fontsize=size)
    return sum(FONTS[key].text_length(part,fontsize=size) for key,part in font_runs(s,font))

def text(x,y,s,size=30,color=INK,bold=False,align='left',maxw=None,font=None):
    font = font or ('bold' if bold else 'regular')
    width = measure(s,font,size)
    if maxw is not None:
        assert width <= maxw + .5, (s,width,maxw)
    if align == 'center': x -= width/2
    if align == 'right': x -= width
    baseline = y + size * .91
    if WEBSITE:
        # Keep one semantic text node even when font fallback or brand color splits a run.
        chunks = [(s, color)]
        if s == 'DreamVLA':
            chunks = [('Dream', color), ('VLA', TEAL)]
        text_node = [f'<text x="{x:.3f}" y="{baseline:.3f}" font-size="{size}" fill="{color}">']
        cursor = x
        for chunk, chunk_color in chunks:
            for key, part in font_runs(chunk, font):
                page.insert_text((cursor,baseline),part,fontname=key,fontsize=size,color=rgb(chunk_color))
                text_node.append(f'<tspan font-family="DreamVLAPoster-{key}" font-weight="{FONT_WEIGHTS[key]}" fill="{chunk_color}">{html.escape(part)}</tspan>')
                cursor += FONTS[key].text_length(part,fontsize=size)
        svg.append(''.join(text_node) + '</text>')
    else:
        page.insert_text((x,baseline),s,fontname=font,fontsize=size,color=rgb(color))
        svg.append(f'<text x="{x:.3f}" y="{baseline:.3f}" font-family="DreamVLAPoster, Liberation Sans, Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
    text_boxes.append(dict(text=s,x=x,y=y,width=width,height=size*1.16,font=font,size=size))
    return width

def para(x,y,s,width,size=30,color=INK,bold=False,leading=1.32):
    key='bold' if bold else 'regular'
    yy=y
    for paragraph in s.split('\n'):
        words=paragraph.split(); current=''
        for word in words:
            trial=(current+' '+word).strip()
            if current and measure(trial,key,size)>width:
                text(x,yy,current,size,color,bold,maxw=width); yy += size*leading;current=word
            else: current=trial
        if current:
            text(x,yy,current,size,color,bold,maxw=width);yy+=size*leading
    return yy

def image(path,x,y,w,h):
    path=Path(path)
    page.insert_image(fitz.Rect(x,y,x+w,y+h),filename=str(path),keep_proportion=True)
    mime='image/jpeg' if path.suffix.lower() in ('.jpg','.jpeg') else 'image/png'
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"/>')
    image_records.append({'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})

def heading(x,y,num,s):
    if WEBSITE:
        text(x,y+8,num,24,TEAL,True)
        text(x+65,y,s,35,NAVY,True,maxw=991)
        line(x,y+59,x+1056,y+59,GRID,1.5)
    else:
        text(x,y,num,37,TEAL,True)
        text(x+75,y,s,37,NAVY,True,maxw=981)
        line(x,y+59,x+1056,y+59,TEAL,3)

def arrow_down(x,y):
    line(x,y,x,y+24,STEP,4)
    line(x,y+24,x-10,y+13,STEP,4)
    line(x,y+24,x+10,y+13,STEP,4)

# Header: author order, affiliations, and equal contribution match the final paper.
rect(0,0,W,H,PAPER)
if WEBSITE:
    # Website-style navigation rule, quiet centered hero, and green VLA wordmark.
    line(84,68,W-84,68,GRID,1.5)
    text(84,22,'CoRL 2026  |  Workshop poster',23,MUTED,font='medium')
    text(W-84,22,'ember-center-berkeley.github.io/DreamVLA',23,MUTED,align='right')
    text(W/2,86,'DreamVLA',108,INK,True,align='center',font='display')
    text(W/2,214,'From Generated Human Motions to Humanoid Vision-Language-Action Policies',47,INK,True,align='center',maxw=3288)
    text(W/2,284,'Kevin Park¹*   Izat Rasyad²*   Dvij Kalaria²   Shankar Sastry²',29,TEAL,align='center',font='medium')
    text(W/2,328,'¹ Princeton University     ² University of California, Berkeley     * Equal contribution',23,MUTED,align='center')
    for card_x in (60,1176,2292):
        rect(card_x,500,1104,1938,WHITE,GRID,1.2,radius=20)
else:
    rect(0,0,W,350,NAVY)
    text(84,48,'DreamVLA',110,WHITE,True)
    text(84,189,'From Generated Human Motions to Humanoid Vision-Language-Action Policies',53,WHITE,maxw=3288)
    text(84,267,'Kevin Park¹*   Izat Rasyad²*   Dvij Kalaria²   Shankar Sastry²',35,'#ffffff')
    text(84,313,'¹ Princeton University     ² University of California, Berkeley     * Equal contribution',26,'#dce8ef')
    text(W-84,110,'ember-center-berkeley.github.io/DreamVLA',31,'#dce8ef',align='right')
    text(W-84,278,'CoRL 2026  |  Workshop poster',28,'#dce8ef',align='right')
rect(84,369,3288,93,PALE,radius=14 if WEBSITE else 0)
text(W/2 if WEBSITE else 120,393,'Generated motion → residual expert → corrective recovery → vision-language-action policy',37,NAVY,True,align='center' if WEBSITE else 'left',maxw=3216)

X1,X2,X3 = 84,1200,2316
heading(X1,522,'01','Generate demonstrations')
heading(X2,522,'02','Recover from learner errors')
heading(X3,522,'03','Measure closed-loop success')

# Left column: expert data pipeline and four scenes.
para(X1,610,'Replace repeated task-specific teleoperation with generated motion and physically executed expert demonstrations.',1056,32)
stages=[
    (750,'Generate and retarget','Task text + contact constraints → OmniControl human motion → OmniRetarget G1 references. Augment with HumanoidMimicGen.'),
    (935,'Adapt contact with residual RL','SONIC supplies a whole-body motion prior. A privileged residual expert co-tracks the robot and task object under randomized physics.'),
    (1120,'Keep successful expert rollouts','Record egocentric RGB, proprioception, executed tokens, hand commands, and language in LeRobot format. Initialize GR00T on accepted episodes.'),
]
for i,(yy,title,body) in enumerate(stages):
    circle(X1+23,yy+22,22,TEAL)
    text(X1+23,yy+3,str(i+1),28,WHITE,True,align='center')
    text(X1+64,yy,title,35,NAVY,True)
    para(X1+64,yy+53,body,982,29)
rect(X1,1310,1056,75,PALE,radius=12 if WEBSITE else 0)
text(X1+26,1330,'Executed token = SONIC token + learned residual',33,TEAL,True,maxw=1004)
text(X1,1412,'Four simulated G1 interactions',35,NAVY,True)
tasks=['Pick from platform','Open drawer','Press button','Walk and pick']
for i,title in enumerate(tasks):
    row,col=divmod(i,2); xx=X1+40+col*530; yy=1480+row*480
    path=next(ASSETS.glob(f'task_{i}.*'))
    image(path,xx+10,yy,420,420)
    text(xx+220,yy+427,title,29,NAVY,True,align='center')

# Middle column: explicit trajectory-level DAgger, not pointwise relabeling.
para(X2,610,'Offline demonstrations cover expert states. Learner errors alter contacts and move the VLA away from that distribution.',1056,32)
steps=[
    ('Roll out the current VLA','Execute images + proprioception + language. Save simulator states along the learner trajectory.'),
    ('Restore a failed learner state','Sample a time uniformly from the failed rollout. Restore that state and resume the same reference suffix.'),
    ('Collect an expert continuation','The privileged expert attempts recovery. Keep the corrective trajectory only when it completes the task.'),
    ('Aggregate and fine-tune',DATA.get('aggregation_description', 'Add accepted continuations to the dataset; fine-tune GR00T and repeat for three DAgger rounds.')),
]
for i,(title,body) in enumerate(steps):
    yy=766+i*171
    circle(X2+26,yy+22,24,STEP)
    text(X2+26,yy+3,str(i+1),30,WHITE,True,align='center')
    text(X2+75,yy,title,34,NAVY,True)
    para(X2+75,yy+52,body,971,29)
    if i<3:arrow_down(X2+26,yy+110)
rect(X2,1468,1056,133,PALE,radius=12 if WEBSITE else 0)
text(X2+28,1487,'Student deployment',30,TEAL,True)
para(X2+28,1530,'RGB + robot state + instruction → token / hand actions → frozen SONIC decoder. No privileged object state or reference input.',1000,27)
text(X2,1645,'Residual adaptation at contact',35,NAVY,True)
text(X2+260,1710,'Nominal SONIC',30,NAVY,True,align='center')
text(X2+796,1710,'Residual SONIC',30,TEAL,True,align='center')
image(next(ASSETS.glob('comparison_2.*')),X2+30,1758,460,526)
image(next(ASSETS.glob('comparison_5.*')),X2+566,1758,460,526)
para(X2,2310,'Matched final frames from the paper: nominal tracking loses the bottle; residual execution retains it.',1056,29)

# Right column: native vector chart.
px,py,pw,ph=X3+83,658,874,470
chart_max = max(80, math.ceil(max(max(item['values']) for item in DATA['series']) / 20) * 20)
for value in range(0,chart_max+1,20):
    yy=py+ph-(value/chart_max)*ph
    line(px,yy,px+pw,yy,GRID,1.4)
    text(px-20,yy-15,f'{value}',26,MUTED,align='right')
text(px,605,'Task success (%)',28,MUTED)
series=[(item['name'], item['values'], color)
        for item, color in zip(DATA['series'], [ORANGE, BLUE, TEAL, PURPLE])]
for k in range(4):
    xx=px+k*(pw-65)/3
    text(xx,py+ph+20,str(k),28,MUTED,align='center')
for name,values,color in series:
    points=[(px+k*(pw-65)/3,py+ph-v/chart_max*ph) for k,v in enumerate(values)]
    for a,b in zip(points,points[1:]):line(*a,*b,color,5)
    for xx,yy in points:circle(xx,yy,7,color)
    text(points[-1][0]+22,points[-1][1]-17,percentage(values[-1]),29,color,True)
text(px+pw/2,1177,'DAgger round',28,MUTED,align='center')
for i,(name,_,color) in enumerate(series):
    xx=X3+i*264
    line(xx,1240,xx+34,1240,color,5)
    text(xx+44,1223,name,27,INK)
text(X3,1283,DATA.get('evaluation_note', f'{DATA["evaluation_rollouts"]} evaluation rollouts per task; paper Table 2.'),26,MUTED)

text(X3,1350,'Corrective data vs. more offline data',34,NAVY,True)
columns=[X3+8,X3+458,X3+704,X3+1038]
headers=['Task','Offline','DAgger-3','Offline control']
for j,h in enumerate(headers):text(columns[j],1418,h,27,MUTED,j==0,align='left' if j==0 else 'right')
line(X3,1464,X3+1056,1464,NAVY,2)
rows=[(item['name'], percentage(item['values'][0]), percentage(item['values'][-1]),
       percentage(item['offline_control'])) for item in DATA['series']]
for i,row in enumerate(rows):
    yy=1483+i*63
    if i%2==0:rect(X3,yy-2,1056,60,ROW_FILL)
    for j,s in enumerate(row):text(columns[j],yy+8,s,31,TEAL if j==2 else INK,j==2,align='left' if j==0 else 'right')
line(X3,1736,X3+1056,1736,GRID,2)
para(X3,1754,DATA['retained_count_note'],1056,24,MUTED)

text(X3,1862,'Generated-data scale: offline only',34,NAVY,True)
cols=[X3+8,X3+460,X3+650,X3+830,X3+1040]
budget_label = 'Refs / episodes' if all('references' in row for row in DATA['scaling']) else 'Episodes'
for j,h in enumerate([budget_label,'Pick','Drawer','Button','Walk + pick']):text(cols[j],1926,h,25,MUTED,align='left' if j==0 else 'right')
line(X3,1966,X3+1056,1966,NAVY,2)
scale_rows=[[(f'{item["references"]} / {item["episodes"]}' if 'references' in item else str(item['episodes'])), *map(percentage, item['success'])]
            for item in DATA['scaling']]
for i,row in enumerate(scale_rows):
    yy=1988+i*67
    for j,s in enumerate(row):text(cols[j],yy,s,30,INK,align='left' if j==0 else 'right')
line(X3,2190,X3+1056,2190,GRID,2)
rect(X3,2221,1056,184,PALE,radius=12 if WEBSITE else 0)
text(X3+25,2242,'Main finding',31,TEAL,True)
para(X3+25,2291,DATA['main_finding'],1004,30,NAVY)

# Footer: scope and short citations, no unsupported hardware claims or fake links.
line(84,2467,W-84,2467,NAVY,2)
text(84,2487,'Scope: simulation only; pretrained motion, control, and VLA backbones remain in use.',25,NAVY,True)
text(84,2530,'Built on DreamControl / OmniControl · OmniRetarget · HumanoidMimicGen · SONIC · GR00T. DAgger: Ross et al., AISTATS 2011.',22,MUTED)
text(W-84,2530,DATA.get('results_footer', 'Results: supplied paper, Tables 2–4'),22,MUTED,align='right')

svg.append('</svg>')
for box in text_boxes:
    assert box['x']>=0 and box['y']>=0
    assert box['x']+box['width']<=W+.5, box
    assert box['y']+box['height']<=H+.5, box
doc.set_metadata({'title':'DreamVLA — CoRL workshop poster','author':'Kevin Park; Izat Rasyad; Dvij Kalaria; Shankar Sastry',
                  'subject':'Generated-motion demonstrations and reference-conditioned recovery DAgger',
                  'keywords':'DreamVLA, humanoid, VLA, DAgger, CoRL, simulation'})
doc.save(OUTPUT/'DreamVLA_CoRL_poster.pdf',deflate=True)
(OUTPUT/'DreamVLA_CoRL_poster.svg').write_text('\n'.join(svg),encoding='utf-8')
page.get_pixmap(matrix=fitz.Matrix(.7,.7)).save(OUTPUT/'DreamVLA_CoRL_poster_preview.png')
(OUTPUT/'layout_checks.json').write_text(json.dumps({'poster_inches':[48,36],'page_points':[W,H],
    'text_elements':len(text_boxes),'text_within_page':True,'anonymous':False,
    'source_pdf':str(SOURCE_PDF) if SOURCE_PDF else None,
    'source_pdf_sha256':SOURCE_SHA256,
    'declared_source_pdf_sha256':DATA.get('source_pdf_sha256'),
    'source_pdf_verified_at_build':SOURCE_SHA256 is not None,
    'content_status':DATA['content_status'],
    'upstream_commit':'3d105f24814017c208ede64dbb25b660d0c327d7',
    'data_file':str(DATA_PATH),
    'data_sha256':hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
    'pymupdf_version':fitz.VersionBind,
    'font_files':{key:str(path) for key,path in FONT_FILES.items()},
    'font_sha256':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in FONT_FILES.items()},
    'theme':args.theme,
    'theme_reference':'https://ember-center-berkeley.github.io/DreamVLA/' if WEBSITE else None,
    'theme_reference_sha256':{name:hashlib.sha256((ROOT/'website_theme'/name).read_bytes()).hexdigest()
                              for name in ('reference.html','reference.css')} if WEBSITE else None,
    'text_boxes':text_boxes,
    'images':image_records,
    'data':DATA,
    'notes':DATA.get('render_notes', ['No common per-round retained trajectory counts asserted across all tasks.',
             'No aggregate MPJPE or unsupported absolute-token comparison included.'])},indent=2))
print(f'Created PDF, editable SVG, and preview in {OUTPUT}')
print(f'Content status: {DATA["content_status"]}')
