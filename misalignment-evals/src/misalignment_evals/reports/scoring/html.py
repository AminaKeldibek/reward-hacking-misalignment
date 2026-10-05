"""Render the analysis as one self-contained HTML page.

Marks are drawn as SVG here rather than in the browser, so a chart can be asserted on in a test.
"""
import html
import json

# Okabe-Ito, which stays distinguishable under the common colour vision deficiencies.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#8c6d1f"]
WIDTH = 880
MARGIN = {"top": 12, "right": 16, "bottom": 34, "left": 44}

_STYLE = """
:root{--ink:#14181d;--ink-2:#424b56;--ink-3:#6b7480;--bg:#fbfbfc;--card:#fff;
 --line:#e3e6ea;--line-2:#eef0f3;--warn-bg:#fdf4e7;--warn-line:#e0a95a;--warn-ink:#7a4e08;
 --mono:"SFMono-Regular",ui-monospace,Menlo,monospace;
 --sans:system-ui,-apple-system,"Segoe UI",sans-serif}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
 --ink:#e7eaee;--ink-2:#aab3be;--ink-3:#7f8894;--bg:#0f1216;--card:#161b21;
 --line:#272e37;--line-2:#1e242b;--warn-bg:#2b2113;--warn-line:#9a7430;--warn-ink:#f0cf95}}
:root[data-theme="dark"]{--ink:#e7eaee;--ink-2:#aab3be;--ink-3:#7f8894;--bg:#0f1216;
 --card:#161b21;--line:#272e37;--line-2:#1e242b;--warn-bg:#2b2113;--warn-line:#9a7430;
 --warn-ink:#f0cf95}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:15px;
 line-height:1.6;margin:0}
.wrap{max-width:1000px;margin:0 auto;padding-block:40px 64px;padding-left:20px;padding-right:20px}
h1{font-size:clamp(24px,4vw,34px);letter-spacing:-.02em;margin:0 0 6px;text-wrap:balance}
h2{font-size:clamp(18px,2.4vw,22px);letter-spacing:-.01em;margin:0 0 6px}
h3{font-size:14.5px;margin:0 0 2px}
p{margin:0 0 12px;max-width:68ch;color:var(--ink-2)}
.sub{color:var(--ink-3);font-size:14px;margin-bottom:24px}
section{margin-top:46px;padding-top:26px;border-top:1px solid var(--line)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.11em;text-transform:uppercase;
 color:var(--ink-3);margin:0 0 8px}
.tiles{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}
.tile{flex:1 1 140px;background:var(--card);border:1px solid var(--line);border-radius:7px;
 padding:12px 14px}
.tile .v{font-family:var(--mono);font-size:21px;font-weight:600;font-variant-numeric:tabular-nums}
.tile .l{font-size:11.5px;color:var(--ink-3);margin-top:3px}
.tblw{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:480px}
th,td{text-align:right;padding:8px 11px;border-bottom:1px solid var(--line-2);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);
 border-bottom:1px solid var(--line)}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:0}
.note{border-left:3px solid var(--warn-line);background:var(--warn-bg);color:var(--warn-ink);
 padding:12px 15px;border-radius:0 6px 6px 0;margin:16px 0;font-size:14px}
.note p{color:inherit;margin:0;max-width:none}
.chart{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:15px 14px 8px;margin:14px 0;position:relative}
.cs{font-size:12px;color:var(--ink-3);margin:0 0 8px}
svg{display:block;width:100%;height:auto;overflow:visible}
.gl{stroke:var(--line-2);stroke-width:1}
.ax{stroke:var(--line);stroke-width:1}
.at{fill:var(--ink-3);font-family:var(--mono);font-size:10px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin:2px 0 10px;font-size:12.5px;
 color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:13px;height:3px;border-radius:2px;flex:none}
.tip{position:absolute;pointer-events:none;opacity:0;transition:opacity .08s;background:var(--card);
 border:1px solid var(--line);border-radius:6px;padding:7px 9px;font-family:var(--mono);
 font-size:11.5px;color:var(--ink);box-shadow:0 5px 18px rgba(0,0,0,.14);white-space:nowrap;z-index:5}
code{font-family:var(--mono);font-size:.9em;background:var(--line-2);padding:1px 5px;border-radius:4px}
.foot{margin-top:40px;padding-top:18px;border-top:1px solid var(--line);font-size:12.5px;
 color:var(--ink-3)}
@media(max-width:560px){table{min-width:420px}}
"""

_SCRIPT = """
document.querySelectorAll('.chart[data-series]').forEach(box=>{
  const series=JSON.parse(box.dataset.series), steps=JSON.parse(box.dataset.steps);
  const svg=box.querySelector('svg'), tip=box.querySelector('.tip');
  if(!svg||!steps.length) return;
  const rule=svg.querySelector('.cursor');
  svg.addEventListener('pointerleave',()=>{tip.style.opacity=0;rule.setAttribute('opacity',0);});
  svg.addEventListener('pointermove',ev=>{
    const box_=svg.getBoundingClientRect();
    const frac=(ev.clientX-box_.left)/box_.width;
    const i=Math.max(0,Math.min(steps.length-1,Math.round(frac*(steps.length-1))));
    const step=steps[i], x=Number(svg.dataset.l)+ (steps.length<2?0:i/(steps.length-1))*Number(svg.dataset.iw);
    rule.setAttribute('x1',x); rule.setAttribute('x2',x); rule.setAttribute('opacity',.35);
    let out=`<b>step ${step}</b>`;
    series.forEach(s=>{
      const p=s.points.find(q=>q.step===step);
      if(!p||p.rate==null) return;
      out+=`<br><span style="color:${s.colour}">&#9632;</span> ${s.name}: `
        +(p.rate*100).toFixed(1)+'%'+(p.n!=null?` (${p.k!=null?p.k+'/':''}${p.n})`:'');
    });
    tip.innerHTML=out; tip.style.opacity=1;
    const px=x/Number(svg.dataset.w)*box_.width;
    tip.style.left=Math.min(Math.max(px-tip.offsetWidth/2,4),box_.width-tip.offsetWidth-4)+'px';
    tip.style.top='30px';
  });
});
"""


def _escape(value) -> str:
    return html.escape(str(value), quote=True)


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{_escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f'<td class="{"" if i == 0 else "num"}">{cell}</td>' for i, cell in enumerate(row)
        ) + "</tr>" for row in rows)
    return f'<div class="tblw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def chart(title: str, caption: str, series: list[dict], steps: list[int], height: int = 240) -> str:
    """One line chart. `series` entries carry name, colour, points, and whether to shade."""
    inner_w = WIDTH - MARGIN["left"] - MARGIN["right"]
    inner_h = height - MARGIN["top"] - MARGIN["bottom"]
    index = {step: position for position, step in enumerate(steps)}
    span = max(len(steps) - 1, 1)

    def x_of(step):
        return MARGIN["left"] + index[step] / span * inner_w

    def y_of(value):
        return MARGIN["top"] + (1 - value) * inner_h

    parts = []
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = y_of(fraction)
        parts.append(f'<line class="gl" x1="{MARGIN["left"]}" x2="{WIDTH - MARGIN["right"]}" '
                     f'y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="at" x="{MARGIN["left"] - 8}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end">{int(fraction * 100)}%</text>')
    parts.append(f'<line class="ax" x1="{MARGIN["left"]}" x2="{WIDTH - MARGIN["right"]}" '
                 f'y1="{y_of(0):.1f}" y2="{y_of(0):.1f}"/>')

    ticks = steps if len(steps) <= 8 else [steps[round(i * span / 6)] for i in range(7)]
    for step in dict.fromkeys(ticks):
        parts.append(f'<text class="at" x="{x_of(step):.1f}" y="{height - MARGIN["bottom"] + 16}" '
                     f'text-anchor="middle">{step}</text>')

    for entry in series:
        points = [p for p in entry["points"] if p.get("rate") is not None and p["step"] in index]
        if not points:
            continue
        colour = entry["colour"]
        if entry.get("band") and all(p.get("lo") is not None for p in points):
            up = " ".join(f"{x_of(p['step']):.1f},{y_of(p['hi']):.1f}" for p in points)
            down = " ".join(f"{x_of(p['step']):.1f},{y_of(p['lo']):.1f}" for p in reversed(points))
            parts.append(f'<path d="M{up} L{down} Z" fill="{colour}" opacity="0.14"/>')
        if entry.get("dots"):
            for p in points:
                if p.get("lo") is not None:
                    parts.append(f'<line x1="{x_of(p["step"]):.1f}" x2="{x_of(p["step"]):.1f}" '
                                 f'y1="{y_of(p["lo"]):.1f}" y2="{y_of(p["hi"]):.1f}" '
                                 f'stroke="{colour}" stroke-width="1.5" opacity="0.5"/>')
                parts.append(f'<circle cx="{x_of(p["step"]):.1f}" cy="{y_of(p["rate"]):.1f}" '
                             f'r="3.5" fill="{colour}"/>')
        else:
            path = " L".join(f"{x_of(p['step']):.1f},{y_of(p['rate']):.1f}" for p in points)
            dash = ' stroke-dasharray="3 3"' if entry.get("dashed") else ""
            parts.append(f'<path d="M{path}" fill="none" stroke="{colour}" '
                         f'stroke-width="2" stroke-linejoin="round"{dash}/>')

    parts.append(f'<line class="cursor ax" x1="0" x2="0" y1="{MARGIN["top"]}" '
                 f'y2="{y_of(0):.1f}" opacity="0"/>')
    legend = "".join(
        f'<span><i class="sw" style="background:{e["colour"]}"></i>{_escape(e["name"])}</span>'
        for e in series)
    payload = _escape(json.dumps([{"name": e["name"], "colour": e["colour"],
                                   "points": e["points"]} for e in series]))
    return (f'<div class="chart" data-series="{payload}" data-steps="{_escape(json.dumps(steps))}">'
            f"<h3>{_escape(title)}</h3><p class='cs'>{_escape(caption)}</p>"
            f'<div class="legend">{legend}</div>'
            f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{_escape(title)}" '
            f'data-l="{MARGIN["left"]}" data-iw="{inner_w}" data-w="{WIDTH}">'
            f'{"".join(parts)}</svg><div class="tip"></div></div>')


def _coverage_section(data: dict) -> str:
    split_column = data["split_column"]
    headers = ["Scorer", "Scored", "of total", "Coverage"]
    arms = sorted({arm for row in data["coverage"] for arm in row.get("invalid_by_arm", {})})
    headers += [f"Invalid at {split_column}={arm}" for arm in arms]
    rows = []
    for row in sorted(data["coverage"], key=lambda r: -r["pct"]):
        cells = [f'<code>{_escape(row["scorer"])}</code>', f'{row["scored"]:,}',
                 f'{row["total"]:,}', f'{row["pct"]:.2f}%']
        cells += [f'{row.get("invalid_by_arm", {}).get(arm, 0):.2f}%' for arm in arms]
        rows.append(cells)
    out = _table(headers, rows)

    reasons = [(row["scorer"], reason, count)
               for row in data["coverage"] for reason, count in row["reasons"].items()]
    if reasons:
        out += "<h3>Why rows were dropped</h3>" + _table(
            ["Scorer", "Reason", "Rows"],
            [[f'<code>{_escape(s)}</code>', _escape(r), c] for s, r, c in reasons])
    if arms and len(arms) == 2:
        out += (f'<div class="note"><p>Compare the two <code>{_escape(split_column)}</code> '
                f'columns. A scorer that drops many more rows on one side is measuring a biased '
                f'subset, and its split should not be read.</p></div>')
    return out


def _reliability_section(data: dict) -> str:
    n = data["totals"]["per_step"]
    rows = [[f'{row["p"]:.0%}', f'&plusmn;{row["half_width"] * 100:.1f} pp',
             f'&gt; {row["gap"] * 100:.1f} pp'] for row in data["reliability"]]
    return (f"<p>Each step holds about {n} rollouts, so a step's rate is that many coin flips. "
            f"These are the limits that follow.</p>"
            + _table(["If the true rate is near", f"95% interval on one step (n={n})",
                      "Gap two steps need"], rows)
            + '<div class="note"><p>Two steps closer together than the last column are not '
              'distinguishable. Read the trend across steps, not the step-to-step wiggles.</p></div>')


def render(data: dict) -> str:
    """The whole page, as one string."""
    steps = data["steps"]
    totals = data["totals"]
    tiles = "".join(
        f'<div class="tile"><div class="v">{value}</div><div class="l">{label}</div></div>'
        for value, label in [
            (f'{totals["rollouts"]:,}', "rollouts"),
            (f'{totals["steps"]:,}', "training steps"),
            (f'{totals["per_step"]:,}', "rollouts per step"),
            (f'{totals["scorers"]:,}', "scorers"),
            (f'{totals["calls"]:,}', "judge calls")])

    body = [f'<div class="wrap"><h1>{_escape(data["title"])}</h1>',
            '<p class="sub">Coverage, per-step rates, and how far they can be trusted.</p>',
            f'<div class="tiles">{tiles}</div>',
            '<section><p class="eyebrow">01 &middot; Coverage</p>'
            "<h2>How many rollouts got scored</h2>"
            "<p>A row counts as scored when the judge returned valid output and its evidence "
            "checked out. Anything else is dropped and counted here.</p>",
            _coverage_section(data), "</section>",
            '<section><p class="eyebrow">02 &middot; Reliability</p>'
            "<h2>What one step can resolve</h2>", _reliability_section(data), "</section>"]

    if data["reference"]:
        reference = [{"name": name, "colour": PALETTE[i % len(PALETTE)],
                      "points": points, "band": i == 0}
                     for i, (name, points) in enumerate(data["reference"].items())]
        body += ['<section><p class="eyebrow">03 &middot; Reference signals</p>'
                 "<h2>What the environment recorded</h2>"
                 "<p>Columns taken straight from the rollouts, not from a judge, so these have "
                 "full coverage.</p>",
                 chart("Reference signals per step", "Rate per training step.", reference, steps),
                 "</section>"]

    body += ['<section><p class="eyebrow">04 &middot; Scorers</p>'
             "<h2>Judge scores per step</h2>"
             "<p>Rate per training step with its 95% interval.</p>"]
    first_reference = next(iter(data["reference"].items()), None)
    for i, (name, points) in enumerate(data["series"].items()):
        series = [{"name": name, "colour": PALETTE[i % len(PALETTE)],
                   "points": points, "band": True}]
        if first_reference:
            series.append({"name": f"{first_reference[0]} (reference)", "colour": "#8a949f",
                           "points": first_reference[1], "dashed": True})
        scored = next((c["pct"] for c in data["coverage"] if c["scorer"] == name), 0.0)
        body.append(chart(name, f"{scored:.2f}% of rollouts scored. Shaded band is the "
                                f"95% interval.", series, steps))
    body.append("</section>")

    if data["split"]:
        column = data["split_column"]
        body += [f'<section><p class="eyebrow">05 &middot; Split</p>'
                 f"<h2>Split by <code>{_escape(column)}</code></h2>"
                 f"<p>Drawn only at steps where both sides have at least "
                 f"{data['min_per_arm']} rollouts; elsewhere a rate carries no information. "
                 f"Points are not joined, because the steps between them hold no comparison.</p>"]
        for i, (name, rows) in enumerate(data["split"].items()):
            usable = [row for row in rows if row["usable"]]
            if not usable:
                body.append(f'<div class="note"><p><code>{_escape(name)}</code>: no step has '
                            f'enough rollouts on both sides, so the split is not shown.</p></div>')
                continue
            series = [{"name": f"{column} = {level}", "colour": PALETTE[level % len(PALETTE)],
                       "dots": True,
                       "points": [{"step": row["step"], "rate": row[f"rate{level}"],
                                   "lo": row[f"lo{level}"], "hi": row[f"hi{level}"],
                                   "n": row[f"n{level}"]} for row in usable]}
                      for level in (0, 1)]
            body.append(chart(f"{name} by {column}",
                              f"{len(usable)} of {len(rows)} steps are usable.", series, steps))
        body.append("</section>")

    body.append('<div class="foot">Generated by '
                '<code>misalignment_evals.reports.scoring</code>. Intervals are Wilson score '
                'intervals on the rollouts within a step.</div></div>')
    body.append(f"<script>{_SCRIPT}</script>")
    return (f"<title>{_escape(data['title'])}</title>"
            f"<style>{_STYLE}</style>" + "".join(body))
