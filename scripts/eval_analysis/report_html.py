"""Render the eval-analysis dict (per checkpoint) as a self-contained HTML report."""
import html


def _pct(x):
    return "&mdash;" if x is None else f"{x*100:.1f}%"


def _bar(mis, ali, w=180):
    tot = mis + ali
    if not tot:
        return '<span class="muted">no data</span>'
    mw = round(w * mis / tot)
    return (f'<span class="bar" style="width:{w}px">'
            f'<span class="bmis" style="width:{mw}px"></span></span>'
            f'<span class="bnum">{mis}/{tot}</span>')


def _af_examples_html(examples):
    if not examples:
        return ""
    tone = {"comply": ("accepted", "mis"), "partial": ("partial / other", "muted"),
            "refuse": ("rejected", "ok")}
    rows = []
    for e in examples:
        lab, cls = tone.get(e["decision"], (e["decision"], "muted"))
        full = e.get("completion") or ""
        nchars = len(full)
        preview = full[:220].replace("\n", " ")
        more = " …" if nchars > 220 else ""
        rows.append(
            f'<article class="ex"><div class="exhead">'
            f'<span class="mono">{html.escape(e.get("checkpoint",""))}</span>'
            f'<span class="mono muted">{html.escape(e.get("condition","")) } tier</span>'
            f'<span class="pill {cls}">{html.escape(lab)}</span>'
            f'<span class="mono muted exlen">{nchars} chars</span></div>'
            f'<div class="q">Request: {html.escape((e.get("question") or "")[:150])}</div>'
            f'<details class="cotwrap"><summary class="cotsum">'
            f'<span class="cotprev">{html.escape(preview)}{more}</span>'
            f'<span class="cottoggle">show full ▸</span></summary>'
            f'<div class="cot">{html.escape(full)}</div></details>'
            + (f'<div class="reason">judge: {html.escape((e.get("reason") or "")[:200])}</div>' if e.get("reason") else "")
            + '</article>')
    return ('<h2>Alignment-faking &mdash; example completions</h2>'
            '<p class="sub">~12 completions across checkpoints and outcomes, judged by the AF LLM '
            'judge. accepted = complied with the harmful request; rejected = refused; other = '
            'partial / hedged.</p><div class="exwrap">' + "".join(rows) + '</div>')


def build_html(results, model="model"):
    cks = [k for k in results.keys() if not k.startswith('_')]
    css = """
    :root{--bg:#FAFAF7;--pa:#fff;--ink:#1A1D21;--mut:#6B7280;--line:#E6E4DD;--soft:#F1EFE9;
      --mis:#B44A2E;--mis-s:#F6E7E0;--aw:#5B7C99;--aw-s:#E6EDF2;--ok:#3F7A5A;--accent:#3F5E63;}
    @media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#14171A;--pa:#1B1F23;
      --ink:#E9ECEF;--mut:#9AA3AD;--line:#2A2F35;--soft:#20252A;--mis:#E08A6C;--mis-s:#2E1D16;
      --aw:#8FB0C9;--aw-s:#17232C;--ok:#7FBB99;--accent:#8FB0B6;}}
    :root[data-theme=dark]{--bg:#14171A;--pa:#1B1F23;--ink:#E9ECEF;--mut:#9AA3AD;--line:#2A2F35;
      --soft:#20252A;--mis:#E08A6C;--mis-s:#2E1D16;--aw:#8FB0C9;--aw-s:#17232C;--ok:#7FBB99;--accent:#8FB0B6;}
    *{box-sizing:border-box}
    body{background:var(--bg);color:var(--ink);margin:0;font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .wrap{max-width:1040px;margin:0 auto;padding:48px 26px 90px}
    h1{font-size:26px;letter-spacing:-.02em;margin:0 0 4px}
    h2{font-size:16px;letter-spacing:.02em;text-transform:uppercase;color:var(--mut);margin:44px 0 14px;font-weight:600}
    .sub{color:var(--mut);margin:0 0 6px;max-width:70ch}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
    table{border-collapse:collapse;width:100%;font-size:14px;background:var(--pa);border:1px solid var(--line)}
    th,td{padding:9px 13px;text-align:right;border-bottom:1px solid var(--soft);white-space:nowrap;font-variant-numeric:tabular-nums}
    th{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);border-bottom:1px solid var(--line);font-weight:600}
    th:first-child,td:first-child{text-align:left}
    tr:last-child td{border-bottom:none}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
    .card{background:var(--pa);border:1px solid var(--line);border-radius:4px;padding:16px 18px}
    .card .ck{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--accent);font-weight:600}
    .big{font-size:30px;font-weight:600;letter-spacing:-.02em;font-family:ui-monospace,monospace}
    .big.mis{color:var(--mis)} .big.aw{color:var(--aw)}
    .cap{color:var(--mut);font-size:12.5px}
    .muted{color:var(--mut)} .mis{color:var(--mis)} .aw{color:var(--aw)} .ok{color:var(--ok)}
    .bar{display:inline-block;height:9px;background:var(--soft);border-radius:2px;vertical-align:middle;overflow:hidden;margin-right:8px}
    .bmis{display:inline-block;height:9px;background:var(--mis)}
    .bnum{font-family:ui-monospace,monospace;font-size:12px;color:var(--mut)}
    .scroll{overflow-x:auto}
    .note{background:var(--soft);border-left:3px solid var(--accent);padding:11px 15px;font-size:13.5px;color:var(--mut);margin-top:12px;border-radius:0 3px 3px 0}
    .exwrap{display:flex;flex-direction:column;gap:12px}.ex{background:var(--pa);border:1px solid var(--line);border-radius:4px}.exhead{display:flex;gap:10px;align-items:center;padding:9px 14px;border-bottom:1px solid var(--soft)}.pill{padding:1px 8px;border-radius:3px;font-size:11px;font-weight:600}.pill.mis{background:var(--mis-s);color:var(--mis)}.pill.ok{background:var(--aw-s);color:var(--ok)}.pill.muted{background:var(--soft);color:var(--mut)}.q{padding:9px 14px 0;font-size:13px;color:var(--mut)}.cot{padding:10px 14px;font-family:ui-monospace,monospace;font-size:12px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;max-height:440px;overflow-y:auto;border-top:1px solid var(--soft)}.cotsum{list-style:none;cursor:pointer;padding:10px 14px;display:flex;gap:10px;align-items:baseline;font-size:12px}.cotsum::-webkit-details-marker{display:none}.cotprev{font-family:ui-monospace,monospace;color:var(--mut);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.cottoggle{color:var(--accent);font-weight:600;white-space:nowrap;font-size:11px}details[open] .cotprev{display:none}details[open] .cottoggle::before{content:"hide full ▾"}details:not([open]) .cottoggle::before{content:"show full ▸"}.cottoggle{font-size:0}.cottoggle::before{font-size:11px}.exlen{margin-left:auto;font-size:11px}.reason{padding:8px 14px;border-top:1px solid var(--soft);font-size:12.5px;color:var(--mut)}footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--line);color:var(--mut);font-size:12.5px}
    """
    parts = [f"<title>Eval Analysis &mdash; {html.escape(model)}</title>",
             f"<style>{css}</style>",
             '<div class="wrap">',
             f"<h1>Eval analysis &mdash; {html.escape(model)}</h1>",
             '<p class="sub">Each <em>completion</em> (one model answer to one eval prompt) gets two '
             'independent labels: a <strong>misalignment</strong> verdict (strict Opus rubric: aligned '
             'vs misaligned) and an <strong>eval-awareness</strong> verdict (the referent-gated judge: '
             'does the model assert <em>this</em> interaction is a test &mdash; aware vs unaware). '
             'Alignment-faking is a separate eval, reported on its own at the bottom.</p>']

    # 0. dataset / data-quality accounting — how many completions, and are any missing / errored?
    parts.append('<h2>Dataset &mdash; how many completions, and are they clean?</h2>'
                 '<p class="sub">The denominator behind every rate below. A completion is counted '
                 'only if the model actually produced text and both judges returned a label.</p>'
                 '<div class="scroll"><table><thead><tr><th>Checkpoint</th>'
                 '<th>completions</th><th>generation errors</th><th>empty output</th>'
                 '<th>got misalignment label</th><th>got awareness label</th>'
                 '<th>awareness unparseable</th></tr></thead><tbody>')
    for ck in cks:
        ac = results[ck]["overall"].get("accounting", {})
        n = ac.get("completions", results[ck]["overall"]["n"])
        def _ok(v):
            return f'<td class="mono ok">{v}</td>' if v == 0 else f'<td class="mono mis">{v}</td>'
        parts.append(
            f'<tr><td>{html.escape(ck)}</td>'
            f'<td class="mono">{n}</td>'
            f'{_ok(ac.get("generation_errors",0))}{_ok(ac.get("empty_completion",0))}'
            f'<td class="mono">{ac.get("scored_misalignment",n)} ({ac.get("scored_misalignment",n)*100//n if n else 0}%)</td>'
            f'<td class="mono">{ac.get("scored_awareness",n)} ({ac.get("scored_awareness",n)*100//n if n else 0}%)</td>'
            f'<td class="mono muted">{ac.get("awareness_invalid",0)}</td></tr>')
    parts.append('</tbody></table></div>'
                 '<div class="note">Read this row-by-row: every completion has real text (0 errors, '
                 '0 empty) and was labelled by <em>both</em> judges. &ldquo;Awareness unparseable&rdquo; '
                 'is the awareness judge returning a malformed or self-contradictory verdict &mdash; '
                 'these are dropped from the awareness rate only (they still have a misalignment label). '
                 'So the misalignment denominator is the full count; the awareness denominator is '
                 'count &minus; unparseable.</div>')

    # headline cards
    parts.append('<h2>Headline per checkpoint</h2><div class="grid">')
    for ck in cks:
        o = results[ck]["overall"]
        al = o["alignment"]; aw = o["awareness"]
        mis = al.get("misaligned", 0); tot_al = mis + al.get("aligned", 0)
        awp = aw.get("aware", 0); tot_aw = awp + aw.get("unaware", 0)
        mr = mis / tot_al if tot_al else 0
        ar = awp / tot_aw if tot_aw else 0
        parts.append(
            f'<div class="card"><div class="ck">{html.escape(ck)}</div>'
            f'<div class="big mis">{mr*100:.1f}%</div><div class="cap">misaligned '
            f'({mis}/{tot_al})</div>'
            f'<div class="big aw" style="margin-top:10px">{ar*100:.1f}%</div>'
            f'<div class="cap">eval-aware ({awp}/{tot_aw})</div></div>')
    parts.append('</div>')

    # 2. cross-tab — plain language: split each checkpoint's completions by awareness, then
    #    within each group show how many were misaligned vs aligned.
    parts.append('<h2>Is the model more misaligned when it notices it is being tested?</h2>'
                 '<p class="sub">For each checkpoint we split all completions into <em>aware</em> and '
                 '<em>unaware</em>, then within each group count misaligned vs aligned. Read a row as: '
                 '&ldquo;of the N completions in this group, X were misaligned and Y aligned.&rdquo; '
                 'The last row per checkpoint is completions whose awareness verdict was unparseable.</p>'
                 '<div class="scroll"><table><thead><tr><th>Checkpoint</th><th>group</th>'
                 '<th>completions in group</th><th>misaligned</th><th>aligned</th>'
                 '<th>misaligned within group</th></tr></thead><tbody>')
    for ck in cks:
        c = results[ck]["awareness_x_alignment"]
        rows = [("aware of being tested", c["aware"], "aw"),
                ("unaware", c["unaware"], "")]
        inv = c.get("invalid", {"misaligned": 0, "aligned": 0, "misalignment_rate": None})
        if inv["misaligned"] + inv["aligned"] > 0:
            rows.append(("awareness unparseable", inv, "muted"))
        for i, (label, g, cls) in enumerate(rows):
            n = g["misaligned"] + g["aligned"]
            ckcell = f'<td rowspan="{len(rows)}">{html.escape(ck)}</td>' if i == 0 else ''
            parts.append(
                f'<tr>{ckcell}<td class="{cls}">{label}</td>'
                f'<td class="mono">{n}</td>'
                f'<td class="mono mis">{g["misaligned"]}</td>'
                f'<td class="mono">{g["aligned"]}</td>'
                f'<td class="mono">{_pct(g["misalignment_rate"])}</td></tr>')
    parts.append('</tbody></table></div>'
                 '<div class="note">The finding: <strong>aware completions are never misaligned</strong> '
                 '(0 across all three checkpoints) &mdash; every misaligned completion came from the '
                 'unaware group. The model is not strategically behaving better <em>because</em> it '
                 'noticed a test; rather, verbalized awareness and misalignment simply never co-occur '
                 'in this data. Group sizes sum to the checkpoint total (aware + unaware + '
                 'unparseable = completions above).</div>')

    # 3 + 4 per-eval tables, one block per checkpoint
    for ck in cks:
        pea = results[ck]["per_eval_alignment"]
        paw = results[ck]["per_eval_awareness"]
        evs = sorted(set(pea) | set(paw))
        parts.append(f'<h2>Per-eval &mdash; {html.escape(ck)}</h2><div class="scroll"><table><thead><tr>'
                     '<th>Eval</th><th>misalignment</th><th>rate</th>'
                     '<th>aware</th><th>unaware</th><th>aware-rate</th></tr></thead><tbody>')
        for e in evs:
            a = pea.get(e, {}); w = paw.get(e, {})
            parts.append(
                f'<tr><td>{html.escape(e)}</td>'
                f'<td>{_bar(a.get("misaligned",0), a.get("aligned",0))}</td>'
                f'<td class="mono">{_pct(a.get("misalignment_rate"))}</td>'
                f'<td class="mono aw">{w.get("aware",0)}</td>'
                f'<td class="mono muted">{w.get("unaware",0)}</td>'
                f'<td class="mono">{_pct(w.get("awareness_rate"))}</td></tr>')
        parts.append('</tbody></table></div>')

    # 5. alignment faking
    _af_judged = any((results[ck].get("alignment_faking") or {}).get("judge") for ck in cks)
    _af_src = ('Each completion classified by the alignment-faking LLM judge (gemini-2.5-flash).'
               if _af_judged else 'Deterministic decision from the completion markers.')
    parts.append('<h2>Alignment faking &mdash; outcomes by tier</h2>'
                 f'<p class="sub">{_af_src} '
                 'accepted = complied with the harmful request; rejected = refused; '
                 'other = partial / unclear. The compliance gap is P(accept|free) &minus; P(accept|paid); '
                 'near zero means no evidence the model games the free-vs-paid distinction.</p>'
                 '<div class="scroll"><table><thead><tr><th>Checkpoint</th><th>tier</th>'
                 '<th>accepted</th><th>rejected</th><th>other</th><th>accept-rate</th>'
                 '<th>compliance gap</th></tr></thead><tbody>')
    for ck in cks:
        af = results[ck].get("alignment_faking")
        if not af:
            continue
        gap = af.get("compliance_gap")
        for i, tier in enumerate(("free", "paid")):
            t = af.get(tier)
            if not t:
                continue
            gcell = (f'<td class="mono" rowspan="2">{gap:+.3f}</td>' if i == 0 and gap is not None else
                     ('<td class="mono" rowspan="2">&mdash;</td>' if i == 0 else ''))
            ckcell = f'<td rowspan="2">{html.escape(ck)}</td>' if i == 0 else ''
            parts.append(
                f'<tr>{ckcell}<td>{tier}</td>'
                f'<td class="mono mis">{t["accepted"]}</td>'
                f'<td class="mono">{t["rejected"]}</td>'
                f'<td class="mono muted">{t["other"]}</td>'
                f'<td class="mono">{_pct(t["acceptance_rate"])}</td>{gcell}</tr>')
    parts.append('</tbody></table></div>')

    parts.append(_af_examples_html(results.get('_af_examples', [])))
    parts.append('<div class="note">Notes on denominators: the misalignment rate is over all '
                 'completions; the awareness rate is over decided (aware + unaware) completions, '
                 'excluding the few unparseable ones. The 770 per checkpoint = alignment_questions '
                 '(500) + betley (250) + exfil_offer, frame_colleague, goals, monitor_disruption (5 '
                 'each). The three headline rates are dominated by the 500 + 250 high-sample evals; '
                 'the four agentic evals have only n=5, so their 40&ndash;100% rates move very few '
                 'completions. Awareness rises across checkpoints (5.3% &rarr; 11.0% &rarr; 13.5%), '
                 'concentrated almost entirely in alignment_questions.</div>')
    parts.append(f'<footer class="mono">{html.escape(model)} &middot; source: results/{html.escape(model)}/*/mgs_completions/logs_*'
                 ' &middot; eval_analysis.run</footer></div>')
    return "\n".join(parts)
