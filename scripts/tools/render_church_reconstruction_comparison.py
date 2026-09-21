#!/usr/bin/env python3
"""Render the measured Church reconstructions and paired metrics for review."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    out = args.directory.resolve()
    rows = json.loads((out/'per-image.json').read_text())
    result = json.loads((out/'result.json').read_text())
    names = {'native_laser':'Native LASER', 'compact_laser':'Compact LASER (active)',
             'released_rqvae':'Released RQVAE', 'continuous_greedy_control':'Continuous greedy control'}
    cells = ''.join(f'<tr><td>{label}</td><td>{result["summary"][name]["lpips"]:.5f}</td>'
                    f'<td>{result["summary"][name]["psnr_db"]:.3f}</td>'
                    f'<td>{result["summary"][name]["mse"]:.6f}</td></tr>' for name,label in names.items())
    template = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Church: stage-1 reconstruction comparison</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:24px auto;padding:0 20px;max-width:1450px;color:#17212d;background:#f7f9fc}
h1{font-size:27px}p{max-width:1000px}table{border-collapse:collapse;background:white;margin:20px 0}th,td{padding:8px 18px;text-align:left;border-bottom:1px solid #d9e0e8}th{font-size:14px}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center;padding:16px;background:#eaf0f7;border-radius:8px;margin:24px 0}
input[type=range]{min-width:220px}input[type=number]{width:65px}button,select,input{font:inherit}button{padding:4px 12px}#grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}article{background:white;padding:10px;border:1px solid #d9e0e8;border-radius:6px}h2{font-size:16px;margin:0 0 8px}img{width:100%;height:auto;display:block}article p{font-size:13px;margin:8px 0 0}.muted{font-size:14px;color:#526174}a{color:#1754ad}@media(max-width:800px){#grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
<h1>Church: where does reconstruction distortion enter?</h1>
<p>All 300 official validation images, with identical original RQVAE preprocessing. The LASER paths share the same frozen three-epoch encoder and decoder. Metrics were measured on full-precision reconstructions before PNG saving.</p>
<table><thead><tr><th>Reconstruction path</th><th>LPIPS ↓</th><th>Mean PSNR (dB) ↑</th><th>Pixel MSE ↓</th></tr></thead><tbody>__TABLE__</tbody></table>
<p><strong>Finding:</strong> native LASER has the lowest average perceptual and pixel error. Compact conversion adds perceptual error on every validation image, but compact LASER and released RQVAE have similar mean LPIPS. This does not establish that reconstruction quality explains the generation gap.</p>
<div class="controls"><button id="prev">← Previous</button><button id="next">Next →</button><label>Image <input id="number" type="number" min="0" max="299" value="0"></label><input id="slider" aria-label="Image index" type="range" min="0" max="299" value="0"><label>Browse <select id="order"><option value="original">Original dataset order</option><option value="degradation">Largest compact LPIPS increase first</option></select></label><label><input id="control" type="checkbox"> Show continuous greedy control</label></div>
<p id="detail" class="muted"></p><div id="grid"></div>
<p class="muted">Native LASER uses its historical OMP implementation with continuous coefficients. Compact LASER uses the exact active frozen tokenizer. The optional greedy control retains continuous coefficients while using residual greedy selection; it helps separate algorithm and coefficient effects, but is not a fixed-support coefficient ablation.</p>
<p><a href="result.json">Full measurements and provenance</a> · <a href="per-image.json">Per-image measurements</a> · <a href="representative-grid.png">Representative comparison grid</a></p>
<script>const rows=__ROWS__, names=__NAMES__;let sequence=rows.map(r=>r.index),position=0;
const slider=document.querySelector('#slider'),number=document.querySelector('#number'),grid=document.querySelector('#grid');
function show(){const r=rows[sequence[position]],methods=['original','native_laser','compact_laser','released_rqvae'];if(document.querySelector('#control').checked)methods.push('continuous_greedy_control');
grid.style.gridTemplateColumns=window.innerWidth<=800?'repeat(2,minmax(0,1fr))':`repeat(${methods.length},minmax(0,1fr))`;slider.value=r.index;number.value=r.index;
document.querySelector('#detail').textContent=`Validation image ${r.index} · compact minus native LPIPS: ${(r.metrics.compact_laser.lpips-r.metrics.native_laser.lpips).toFixed(5)} · click an image for its native PNG`;
grid.innerHTML=methods.map(m=>{const path=`${m}/${String(r.index).padStart(3,'0')}.png`,v=r.metrics[m];return `<article><h2>${m==='original'?'Original photograph':names[m]}</h2><a href="${path}" target="_blank"><img src="${path}" alt="${m}, validation ${r.index}"></a><p>${v?`LPIPS ${v.lpips.toFixed(4)} · PSNR ${v.psnr_db.toFixed(2)} dB`:'Same input for all paths'}</p></article>`}).join('')}
document.querySelector('#prev').onclick=()=>{position=(position+rows.length-1)%rows.length;show()};document.querySelector('#next').onclick=()=>{position=(position+1)%rows.length;show()};
function selectIndex(v){const i=Math.min(rows.length-1,Math.max(0,Math.round(Number(v)||0)));position=sequence.indexOf(i);show()}
slider.oninput=e=>selectIndex(e.target.value);number.onchange=e=>selectIndex(e.target.value);document.querySelector('#control').onchange=show;
document.querySelector('#order').onchange=e=>{sequence=rows.map(r=>r.index);if(e.target.value==='degradation')sequence.sort((a,b)=>(rows[b].metrics.compact_laser.lpips-rows[b].metrics.native_laser.lpips)-(rows[a].metrics.compact_laser.lpips-rows[a].metrics.native_laser.lpips));position=0;show()};
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.key==='ArrowRight')document.querySelector('#next').click();if(e.key==='ArrowLeft')document.querySelector('#prev').click()});window.addEventListener('resize',show);show();</script></html>'''
    html = template.replace('__TABLE__',cells).replace('__ROWS__',json.dumps(rows)).replace('__NAMES__',json.dumps(names))
    (out/'comparison.html').write_text(html)

    # Scientific contact sheets: preserve measured RGB PNGs without enhancement.
    # Columns are original, native LASER, active compact LASER, released RQVAE.
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 16)
    except OSError:
        font = ImageFont.load_default()
    selected = [round(i*(len(rows)-1)/7) for i in range(8)]
    worst = sorted(range(len(rows)),key=lambda i:rows[i]['metrics']['compact_laser']['lpips']-
                   rows[i]['metrics']['native_laser']['lpips'],reverse=True)[:8]
    for filename,indices in [('representative-grid.png',selected),('largest-compact-degradation.png',worst)]:
        methods = ['original','native_laser','compact_laser','released_rqvae']
        sheet = Image.new('RGB', (1044, 44+len(indices)*284), 'white')
        draw = ImageDraw.Draw(sheet)
        for column,method in enumerate(methods):
            draw.text((4+column*260, 8), 'Original photograph' if method=='original' else names[method],
                      fill='#17212d', font=font)
        for row,index in enumerate(indices):
            y = 44+row*284
            draw.text((4, y), f'Validation image {index}', fill='#526174', font=font)
            for column,method in enumerate(methods):
                with Image.open(out/method/f'{index:03d}.png') as source:
                    assert source.size == (256,256) and source.mode == 'RGB'
                    sheet.paste(source, (4+column*260,y+24))
        sheet.save(out/filename)
    (out/'grid-selection.json').write_text(json.dumps(dict(columns=['original','native_laser','compact_laser','released_rqvae'],
        representative_indices=selected,largest_compact_degradation_indices=worst,
        representative_selection='Eight evenly spaced validation indices, independent of model error',
        worst_selection='Largest compact minus native per-image LPIPS'),indent=2)+'\n')
    print(out/'comparison.html')


if __name__ == '__main__':
    main()
