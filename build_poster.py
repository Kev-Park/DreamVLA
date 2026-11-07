"""Print-size DreamVLA poster, using original paper figures and vector text/charts."""
from pathlib import Path
import base64
import html
import json
import math
import fitz

ROOT = Path(__file__).resolve().parent
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
FONT_FILES = {'regular': 'arial.ttf', 'bold': 'arialbd.ttf'}
FONTS = {k: fitz.Font(fontfile=f'C:/Windows/Fonts/{v}') for k, v in FONT_FILES.items()}
doc = fitz.open()
page = doc.new_page(width=W, height=H)
for key, name in FONT_FILES.items():
    page.insert_font(fontname=key, fontfile=f'C:/Windows/Fonts/{name}')
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="48in" height="36in" viewBox="0 0 {W} {H}">']
text_boxes = []

def rgb(color):
    return tuple(int(color[i:i+2], 16) / 255 for i in (1, 3, 5))

def rect(x, y, w, h, fill, stroke=None, sw=1):
    page.draw_rect(fitz.Rect(x, y, x+w, y+h), color=rgb(stroke) if stroke else None,
                   fill=rgb(fill) if fill else None, width=sw)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def line(x1, y1, x2, y2, color=GRID, sw=2, dash=None):
    page.draw_line((x1,y1),(x2,y2), color=rgb(color),width=sw,
                   dashes='[10 7] 0' if dash else None)
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"'+(' stroke-dasharray="10 7"' if dash else '')+'/>')

def circle(x,y,r,fill):
    page.draw_circle((x,y),r,color=None,fill=rgb(fill))
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>')

def text(x,y,s,size=30,color=INK,bold=False,align='left',maxw=None):
    font = 'bold' if bold else 'regular'
    width = FONTS[font].text_length(s,fontsize=size)
    if maxw is not None:
        assert width <= maxw + .5, (s,width,maxw)
    if align == 'center': x -= width/2
    if align == 'right': x -= width
    baseline = y + size * .91
    page.insert_text((x,baseline),s,fontname=font,fontsize=size,color=rgb(color))
    svg.append(f'<text x="{x:.3f}" y="{baseline:.3f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(s)}</text>')
    text_boxes.append(dict(text=s,x=x,y=y,width=width,height=size*1.16))
    return width

def para(x,y,s,width,size=30,color=INK,bold=False,leading=1.32):
    key='bold' if bold else 'regular'
    yy=y
    for paragraph in s.split('\n'):
        words=paragraph.split(); current=''
        for word in words:
            trial=(current+' '+word).strip()
            if current and FONTS[key].text_length(trial,fontsize=size)>width:
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

def heading(x,y,num,s):
    text(x,y,num,37,TEAL,True)
    text(x+75,y,s,37,NAVY,True,maxw=981)
    line(x,y+59,x+1056,y+59,TEAL,3)

def arrow_down(x,y):
    line(x,y,x,y+24,ORANGE,4)
    line(x,y+24,x-10,y+13,ORANGE,4)
    line(x,y+24,x+10,y+13,ORANGE,4)

# Header: author order, affiliations, and equal contribution match the final paper.
rect(0,0,W,H,WHITE)
rect(0,0,W,350,NAVY)
text(84,48,'DreamVLA',110,WHITE,True)
text(84,189,'From Generated Human Motions to Humanoid Vision-Language-Action Policies',53,WHITE,maxw=3288)
text(84,267,'Kevin Park¹*   Izat Rasyad²*   Dvij Kalaria²   Shankar Sastry²',35,'#ffffff')
text(84,313,'¹ Princeton University     ² University of California, Berkeley     * Equal contribution',26,'#dce8ef')
text(W-84,110,'ember-center-berkeley.github.io/DreamVLA',31,'#dce8ef',align='right')
text(W-84,278,'CoRL 2026  |  Workshop poster',28,'#dce8ef',align='right')
rect(84,369,3288,93,PALE)
text(120,393,'Generated motion → residual expert → corrective recovery → vision-language-action policy',37,NAVY,True,maxw=3216)

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
rect(X1,1310,1056,75,PALE)
text(X1+26,1330,'Executed token = SONIC token + learned residual',33,TEAL,True,maxw=1004)
text(X1,1412,'Four simulated G1 interactions',35,NAVY,True)
tasks=['Pick from platform','Open drawer','Press button','Walk and pick']
for i,title in enumerate(tasks):
    row,col=divmod(i,2); xx=X1+40+col*530; yy=1480+row*480
    path=next((ROOT/'assets').glob(f'task_{i}.*'))
    image(path,xx+10,yy,420,420)
    text(xx+220,yy+427,title,29,NAVY,True,align='center')

# Middle column: explicit trajectory-level DAgger, not pointwise relabeling.
para(X2,610,'Offline demonstrations cover expert states. Learner errors alter contacts and move the VLA away from that distribution.',1056,32)
steps=[
    ('Roll out the current VLA','Execute images + proprioception + language. Save simulator states along the learner trajectory.'),
    ('Restore a failed learner state','Sample a time uniformly from the failed rollout. Restore that state and resume the same reference suffix.'),
    ('Collect an expert continuation','The privileged expert attempts recovery. Keep the corrective trajectory only when it completes the task.'),
    ('Aggregate and fine-tune','Add accepted continuations to the dataset; fine-tune GR00T and repeat for three DAgger rounds.'),
]
for i,(title,body) in enumerate(steps):
    yy=766+i*171
    circle(X2+26,yy+22,24,ORANGE)
    text(X2+26,yy+3,str(i+1),30,WHITE,True,align='center')
    text(X2+75,yy,title,34,NAVY,True)
    para(X2+75,yy+52,body,971,29)
    if i<3:arrow_down(X2+26,yy+110)
rect(X2,1468,1056,133,PALE)
text(X2+28,1487,'Student deployment',30,TEAL,True)
para(X2+28,1530,'RGB + robot state + instruction → token / hand actions → frozen SONIC decoder. No privileged object state or reference input.',1000,27)
text(X2,1645,'Residual adaptation at contact',35,NAVY,True)
text(X2+260,1710,'Nominal SONIC',30,NAVY,True,align='center')
text(X2+796,1710,'Residual SONIC',30,TEAL,True,align='center')
image(next((ROOT/'assets').glob('comparison_2.*')),X2+30,1758,460,526)
image(next((ROOT/'assets').glob('comparison_5.*')),X2+566,1758,460,526)
para(X2,2310,'Matched final frames from the paper: nominal tracking loses the bottle; residual execution retains it.',1056,29)

# Right column: native vector chart.
px,py,pw,ph=X3+83,658,874,470
for value in range(0,81,20):
    yy=py+ph-(value/80)*ph
    line(px,yy,px+pw,yy,GRID,1.4)
    text(px-20,yy-15,f'{value}',26,MUTED,align='right')
text(px,605,'Task success (%)',28,MUTED)
series=[('Pick',[0,17,30,54],ORANGE),('Drawer',[55,57,54,62],BLUE),
        ('Button',[50,55,61,72],TEAL),('Walk + pick',[0,1,7,18],PURPLE)]
for k in range(4):
    xx=px+k*(pw-65)/3
    text(xx,py+ph+20,str(k),28,MUTED,align='center')
for name,values,color in series:
    points=[(px+k*(pw-65)/3,py+ph-v/80*ph) for k,v in enumerate(values)]
    for a,b in zip(points,points[1:]):line(*a,*b,color,5)
    for xx,yy in points:circle(xx,yy,7,color)
    text(points[-1][0]+22,points[-1][1]-17,f'{values[-1]}%',29,color,True)
text(px+pw/2,1177,'DAgger round',28,MUTED,align='center')
for i,(name,_,color) in enumerate(series):
    xx=X3+i*264
    line(xx,1240,xx+34,1240,color,5)
    text(xx+44,1223,name,27,INK)
text(X3,1283,'100 evaluation rollouts per task; paper Table 2.',26,MUTED)

text(X3,1350,'Corrective data vs. more offline data',34,NAVY,True)
columns=[X3+8,X3+458,X3+704,X3+1038]
headers=['Task','Offline','DAgger-3','Offline control']
for j,h in enumerate(headers):text(columns[j],1418,h,27,MUTED,j==0,align='left' if j==0 else 'right')
line(X3,1464,X3+1056,1464,NAVY,2)
rows=[('Pick','0%','54%','0%'),('Drawer','55%','62%','54%'),('Button','50%','72%','52%'),('Walk + pick','0%','18%','0%')]
for i,row in enumerate(rows):
    yy=1483+i*63
    if i%2==0:rect(X3,yy-2,1056,60,'#f4f7f9')
    for j,s in enumerate(row):text(columns[j],yy+8,s,31,TEAL if j==2 else INK,j==2,align='left' if j==0 else 'right')
line(X3,1736,X3+1056,1736,GRID,2)
para(X3,1754,'Walking retained episodes, base → R3: 200 → 314 → 416 → 535. Successful corrections are filtered after rollout attempts.',1056,24,MUTED)

text(X3,1862,'Generated-data scale: offline only',34,NAVY,True)
cols=[X3+8,X3+460,X3+650,X3+830,X3+1040]
for j,h in enumerate(['Episodes','Pick','Drawer','Button','Walk + pick']):text(cols[j],1926,h,25,MUTED,align='left' if j==0 else 'right')
line(X3,1966,X3+1056,1966,NAVY,2)
scale_rows=[['10','0%','4%','0%','0%'],['50','0%','15%','15%','0%'],['200','0%','55%','50%','0%']]
for i,row in enumerate(scale_rows):
    yy=1988+i*67
    for j,s in enumerate(row):text(cols[j],yy,s,30,INK,align='left' if j==0 else 'right')
line(X3,2190,X3+1056,2190,GRID,2)
rect(X3,2221,1056,184,PALE)
text(X3+25,2242,'Main finding',31,TEAL,True)
para(X3+25,2291,'More offline data improves drawer and button success. DAgger raises pick success from 0% to 54% in this study.',1004,30,NAVY)

# Footer: scope and short citations, no unsupported hardware claims or fake links.
line(84,2467,W-84,2467,NAVY,2)
text(84,2487,'Scope: simulation only; pretrained motion, control, and VLA backbones remain in use.',25,NAVY,True)
text(84,2530,'Built on DreamControl / OmniControl · OmniRetarget · HumanoidMimicGen · SONIC · GR00T. DAgger: Ross et al., AISTATS 2011.',22,MUTED)
text(W-84,2530,'Results: supplied paper, Tables 2–4',22,MUTED,align='right')

svg.append('</svg>')
for box in text_boxes:
    assert box['x']>=0 and box['y']>=0
    assert box['x']+box['width']<=W+.5, box
    assert box['y']+box['height']<=H+.5, box
doc.set_metadata({'title':'DreamVLA — CoRL workshop poster','author':'Kevin Park; Izat Rasyad; Dvij Kalaria; Shankar Sastry',
                  'subject':'Generated-motion demonstrations and reference-conditioned recovery DAgger',
                  'keywords':'DreamVLA, humanoid, VLA, DAgger, CoRL, simulation'})
doc.save(ROOT/'DreamVLA_CoRL_poster.pdf',deflate=True)
(ROOT/'DreamVLA_CoRL_poster.svg').write_text('\n'.join(svg),encoding='utf-8')
page.get_pixmap(matrix=fitz.Matrix(.7,.7)).save(ROOT/'DreamVLA_CoRL_poster_preview.png')
(ROOT/'layout_checks.json').write_text(json.dumps({'poster_inches':[48,36],'page_points':[W,H],
    'text_elements':len(text_boxes),'text_within_page':True,'anonymous':False,
    'source_pdf':'K:/Downloads/DreamVLA_overleaf.pdf',
    'walking_retained_counts_source':'out/walking_all12_legacy_results_20261022.json',
    'data':{'pick':[0,17,30,54],'drawer':[55,57,54,62],'button':[50,55,61,72],'walk_and_pick':[0,1,7,18]},
    'notes':['No common per-round retained trajectory counts asserted across all tasks.',
             'No aggregate MPJPE or unsupported absolute-token comparison included.']},indent=2))
print('Created PDF, editable SVG, and preview.')
