"""
Human-readable reports for an `Analysis`.

write_markdown -- a plain report.md that renders on GitHub.
write_html     -- one self-contained, interactive report.html: searchable and
                  sortable tables, embedded figures, plain-English verdicts.
                  No network access needed to open it.
"""

from __future__ import annotations

import base64
import html
import json
import os

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    if df is None or df.empty:
        return "_none_\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [str(v).replace("|", "\\|") for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(result, path: str) -> str:
    o = result.options
    out = [
        f"# Ghost-pattern report: {result.name}\n",
        f"{result.n_rows} rows x {result.n_columns} columns. "
        f"Null model: {o.null_model.__name__}, {o.n_null} replicates, "
        f"alpha = {o.alpha}. Runtime {result.seconds:.0f}s.\n",
        f"**Summary:** {result.headline()}\n",
    ]
    if result.dropped_constant:
        out.append(f"Dropped constant columns: {result.dropped_constant}\n")

    rep = result.leakage
    if rep is not None:
        out.append("## Leakage check\n")
        if rep.clean:
            out.append("No hard leakage detected.\n")
        else:
            out.append(f"**{len(rep.hard)} hard leakage issue(s).** Rules "
                       "that use these columns may be true by construction.\n")
            frame = rep.to_frame()
            out.append(md_table(frame[frame.kind != "strong_association"]))
        if rep.advisory:
            out.append(f"\n{len(rep.advisory)} strongly associated pair(s) "
                       "(advisory, see `leakage.csv`).\n")
        if result.dropped_leaking:
            out.append(f"\nDropped: {result.dropped_leaking}\n")

    r = result.rules
    if r is not None:
        out.append("## Association rules\n")
        if o.target:
            out.append(f"Question: which conditions predict **{o.target}**?\n")
        out.append(
            f"- Naive mining returned **{r.n_naive}** rules"
            + (f" ({r.n_candidates} predict the target)" if o.target else "")
            + f"; **{r.n_distinct}** remain after redundancy filtering.\n"
            f"- A structureless copy of the data yields **{r.ghost_rate:.1f}**"
            " rules on average (the ghost rate).\n"
            f"- **{len(r.survivors)}** rules beat the null.\n")
        if r.null_max is not None:
            out.append(f"- Strongest ghost lift: {r.null_max:.3f}; null 95th "
                       f"percentile: {r.null_p95:.3f}.\n")
        top = result.top_findings()
        if not top.empty:
            out.append("\n### Top findings\n")
            out.append(md_table(top[["rule", "support", "confidence", "lift",
                                     "fwer_p", "verdict"]], max_rows=len(top)))
            out.append("\n![top rules](figures/top_rules.png)\n"
                       "![lift vs null](figures/lift_null.png)\n"
                       "![ghost rate](figures/ghost_rate.png)\n")

    if result.attribution is not None:
        out.append("\n## Column attribution\n")
        out.append(md_table(result.attribution[
            ["column", "drop", "z_score", "p_value", "load_bearing"]]))
        out.append("\n![attribution](figures/column_attribution.png)\n")

    c = result.clusters
    if c is not None:
        out.append("\n## Clustering\n")
        out.append(f"- k values beating noise: {c.good_k or 'none'}\n"
                   f"- Gap statistic chooses k = {c.gap.attrs.get('chosen_k')}\n")
        out.extend(f"- {s}\n" for s in c.stability)
        out.append("\n![silhouette](figures/silhouette.png)\n")

    if result.anomalies:
        out.append("\n## Anomalies\n")
        out.append(f"Relative contrast: {result.relative_contrast:.3f}.\n\n")
        for a in result.anomalies:
            out.append(f"- {a}\n")
            out.append(f"\n![{a.method}](figures/anomaly_{a.method}.png)\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


# ------------------------------------------------------------------ HTML


def explain_rule(rule: str, confidence: float, lift: float) -> str:
    """'{a=x, b=y} -> {t=1}' as a sentence a non-statistician can read."""
    ante, cons = rule.split(" -> ")
    ante = ante.strip("{}").replace(", ", " and ")
    cons = cons.strip("{}")
    return (f"When {ante}, {cons} holds {confidence:.0%} of the time "
            f"({lift:.2f}x its usual rate).")


def _img(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f'<img alt="" src="data:image/png;base64,{data}">'


def _records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or df.empty:
        return []
    clean = df.replace([np.inf, -np.inf], np.nan).astype(object)
    return clean.where(pd.notna(clean), None).to_dict(orient="records")


def _json(obj) -> str:
    # "</" inside a <script> block would end it early.
    return json.dumps(obj, default=str).replace("</", "<\\/")


def _card(label: str, value, note: str = "", tone: str = "") -> str:
    return (f'<div class="card {tone}"><div class="label">{html.escape(label)}'
            f'</div><div class="value">{html.escape(str(value))}</div>'
            f'<div class="note">{html.escape(note)}</div></div>')


CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d2327;--muted:#66707a;--line:#e3e3df;
--real:#1a5f7a;--likely:#6c8ebf;--ghost:#c44536;--warn:#b7791f;--chip:#eef3f5}
@media (prefers-color-scheme:dark){:root{--bg:#121416;--panel:#1b1e21;
--ink:#e7e9ea;--muted:#9aa3ab;--line:#2c3136;--real:#5fb3d3;--likely:#8fb0e0;
--ghost:#e2735f;--warn:#e0b050;--chip:#23292e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}
h2{font-size:20px;margin:40px 0 12px;padding-bottom:6px;
border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 20px}
.summary{background:var(--panel);border:1px solid var(--line);
border-left:4px solid var(--real);padding:14px 18px;border-radius:8px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
gap:12px;margin:18px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:12px 14px}
.card .label{font-size:12px;color:var(--muted);text-transform:uppercase;
letter-spacing:.04em}
.card .value{font-size:26px;font-weight:700}
.card .note{font-size:12px;color:var(--muted)}
.card.real .value{color:var(--real)}.card.ghost .value{color:var(--ghost)}
.card.warn .value{color:var(--warn)}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 10px}
input,select,button{font:inherit;padding:6px 10px;border:1px solid var(--line);
border-radius:6px;background:var(--panel);color:var(--ink)}
button{cursor:pointer}input{flex:1;min-width:200px}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}
th{cursor:pointer;user-select:none;background:var(--panel);white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:var(--chip)}
.why{color:var(--muted);font-size:12.5px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;
font-weight:600;white-space:nowrap;color:#fff}
.pill.REAL{background:var(--real)}.pill.likely{background:var(--likely)}
.pill.GHOST,.pill.hard{background:var(--ghost)}
.pill.advisory{background:var(--chip);color:var(--muted)}
.count{color:var(--muted);font-size:13px;margin-top:6px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
gap:14px;margin-top:16px}
.figs img{width:100%;background:#fff;border:1px solid var(--line);
border-radius:8px}
.bar{height:10px;background:var(--real);border-radius:3px}
.muted{color:var(--muted)}
footer{margin-top:48px;color:var(--muted);font-size:13px}
code{background:var(--chip);padding:1px 5px;border-radius:4px}
"""

JS = r"""
function esc(s){return s==null?'':String(s).replace(/[&<>"]/g,
  c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function n(d){return v=>v==null?'':Number(v).toFixed(d)}
function pill(v){const c=v.startsWith('REAL')?'REAL':v.startsWith('likely')?
  'likely':'GHOST';return '<span class="pill '+c+'">'+esc(v)+'</span>'}
function table(id,rows,cols,opts){
  const host=document.getElementById(id);if(!host)return;
  let sortKey=opts.sort||null,asc=false,shown=opts.page||50;
  const search=host.querySelector('input'),sel=host.querySelector('select');
  const body=host.querySelector('tbody'),head=host.querySelector('thead tr');
  const count=host.querySelector('.count'),more=host.querySelector('button');
  cols.forEach(c=>{const th=document.createElement('th');th.textContent=c.label;
    th.onclick=()=>{asc=sortKey===c.key?!asc:false;sortKey=c.key;draw()};
    head.appendChild(th)});
  function draw(){
    const q=(search&&search.value||'').toLowerCase(),v=sel&&sel.value;
    let r=rows.filter(x=>(!q||JSON.stringify(x).toLowerCase().includes(q))
      &&(!v||x[opts.filterKey]===v));
    if(sortKey)r=r.slice().sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
      const d=(typeof x==='number'&&typeof y==='number')?x-y:
        String(x).localeCompare(String(y));return asc?d:-d});
    body.innerHTML=r.slice(0,shown).map(x=>'<tr>'+cols.map(c=>
      '<td class="'+(c.num?'num':'')+'">'+(c.fmt?c.fmt(x[c.key],x):
      esc(x[c.key]))+'</td>').join('')+'</tr>').join('');
    count.textContent='Showing '+Math.min(shown,r.length)+' of '+r.length;
    more.style.display=r.length>shown?'':'none';
  }
  if(search)search.oninput=draw;if(sel)sel.onchange=draw;
  more.onclick=()=>{shown+=100;draw()};
  draw();
}
"""


def _table_block(id_: str, placeholder: str, filter_values=None,
                 filter_label="All") -> str:
    sel = ""
    if filter_values:
        opts = "".join(f'<option value="{html.escape(v)}">{html.escape(v)}'
                       "</option>" for v in filter_values)
        sel = f'<select><option value="">{filter_label}</option>{opts}</select>'
    return (f'<div id="{id_}"><div class="controls"><input type="search" '
            f'placeholder="{html.escape(placeholder)}">{sel}</div>'
            '<div class="tablewrap"><table><thead><tr></tr></thead><tbody>'
            '</tbody></table></div><div class="count"></div>'
            '<div class="controls"><button type="button">Show more</button>'
            "</div></div>")


RULE_COLUMNS = """[
{key:'rule',label:'Rule',fmt:(v,x)=>esc(v)+'<div class="why">'+esc(x.why)+'</div>'},
{key:'verdict',label:'Verdict',fmt:pill},
{key:'lift',label:'Lift',num:1,fmt:n(3)},
{key:'confidence',label:'Confidence',num:1,fmt:v=>(v*100).toFixed(1)+'%'},
{key:'support',label:'Support',num:1,fmt:n(4)},
{key:'count',label:'Rows',num:1},
{key:'fwer_p',label:'FWER p',num:1,fmt:n(4)}]"""

LEAK_COLUMNS = """[
{key:'tier',label:'Tier',fmt:v=>'<span class="pill '+v+'">'+v+'</span>'},
{key:'kind',label:'Kind'},{key:'columns',label:'Columns'},
{key:'severity',label:'Severity',num:1,fmt:n(3)},
{key:'detail',label:'Detail'}]"""


def write_html(result, path: str, figures: dict[str, str] | None = None) -> str:
    """Write a single self-contained interactive HTML report."""
    figures = figures or {}
    o = result.options
    e = html.escape
    body, scripts = [], []

    body.append(f"<h1>Ghost-pattern report: {e(result.name)}</h1>")
    body.append(
        f'<p class="sub">{result.n_rows:,} rows &middot; {result.n_columns} '
        f"columns &middot; null model <code>{e(o.null_model.__name__)}</code>, "
        f"{o.n_null} replicates, &alpha; = {o.alpha} &middot; "
        f"{result.seconds:.0f}s</p>")
    body.append(f'<div class="summary"><strong>Verdict.</strong> '
                f"{e(result.headline())}</div>")

    cards = []
    r = result.rules
    if r is not None:
        cards += [
            _card("Naive rules", f"{r.n_naive:,}", "what plain Apriori reports"),
            _card("Ghost rate", f"{r.ghost_rate:.1f}",
                  "rules found in pure noise", "ghost"),
            _card("Distinct hypotheses", f"{r.n_distinct:,}",
                  "after redundancy filtering"),
            _card("Validated", f"{len(r.survivors):,}", "beat the null", "real"),
        ]
    rep = result.leakage
    if rep is not None:
        cards.append(_card("Hard leakage", len(rep.hard),
                           "true-by-construction risks",
                           "warn" if rep.hard else "real"))
    if result.clusters is not None:
        k = result.clusters.good_k
        cards.append(_card("Real clusters", k[0] if k else "none",
                           "best k beating noise" if k else "no k beats noise",
                           "real" if k else "ghost"))
    if result.anomalies:
        real = sum(a.real for a in result.anomalies)
        cards.append(_card("Anomaly detectors",
                           f"{real}/{len(result.anomalies)}",
                           "find excess over noise", "real" if real else "ghost"))
    body.append(f'<div class="cards">{"".join(cards)}</div>')

    # ---- rules
    if r is not None and not r.table.empty:
        body.append("<h2>Findings</h2>")
        if o.target:
            body.append(f"<p>Question: which conditions predict "
                        f"<strong>{e(o.target)}</strong>?</p>")
        body.append(
            f'<p class="muted">Every rule was tested against {o.n_null} '
            "shuffled copies of the data. <em>REAL (family-wise)</em> beats "
            "the best rule noise ever produced; <em>likely real (FDR)</em> "
            "passes a false-discovery-rate test; <em>GHOST</em> is what noise "
            "produces anyway. Click a column header to sort.</p>")
        table = r.table.copy()
        table["why"] = [explain_rule(a, c, lf) for a, c, lf in
                        zip(table.rule, table.confidence, table.lift)]
        verdicts = [v for v in ("REAL (family-wise)", "likely real (FDR)",
                                "GHOST") if (table.verdict == v).any()]
        body.append(_table_block("rules", "Search rules, e.g. odor=f",
                                 verdicts, "All verdicts"))
        scripts.append(f"table('rules',{_json(_records(table))},"
                       f"{RULE_COLUMNS},{{filterKey:'verdict',page:50}});")
        figs = [figures.get(k) for k in ("top_rules.png", "lift_null.png",
                                         "ghost_rate.png")]
        body.append('<div class="figs">' + "".join(_img(f) for f in figs)
                    + "</div>")

    # ---- attribution
    if result.attribution is not None:
        a = result.attribution
        top = max(float(a["drop"].max()), 1e-9)
        rows = "".join(
            f"<tr><td>{e(str(x.column))}</td><td style='width:45%'>"
            f"<div class='bar' style='width:{max(x.drop, 0) / top * 100:.0f}%'>"
            f"</div></td><td class='num'>{x.drop:g}</td><td class='num'>"
            f"{x.p_value:.4f}</td><td>{'yes' if x.load_bearing else 'no'}"
            "</td></tr>" for x in a.itertuples())
        body.append(
            "<h2>Which columns carry the structure</h2><p class='muted'>Each "
            "column is shuffled on its own; the drop is how many strong rules "
            "disappear. Columns with no drop are decoration.</p>"
            "<div class='tablewrap'><table><thead><tr><th>Column</th>"
            "<th>Effect</th><th>Drop</th><th>p</th><th>Load-bearing</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>")

    # ---- leakage
    if rep is not None:
        body.append("<h2>Leakage check</h2>")
        body.append("<p class='muted'>Leakage passes every statistical test, "
                    "so it is checked separately. <em>Hard</em> issues make "
                    "rules true by construction; <em>advisory</em> pairs are "
                    "strong dependencies worth a look.</p>")
        if rep.findings:
            lf = rep.to_frame()
            lf["tier"] = np.where(lf.kind == "strong_association",
                                  "advisory", "hard")
            body.append(_table_block("leak", "Search columns",
                                     sorted(lf.tier.unique()), "All tiers"))
            scripts.append(f"table('leak',{_json(_records(lf))},{LEAK_COLUMNS},"
                           "{filterKey:'tier',sort:'severity',page:25});")
        else:
            body.append("<p>No leakage detected.</p>")

    # ---- clusters
    c = result.clusters
    if c is not None:
        body.append("<h2>Clustering vs noise</h2>")
        body.append(f"<p>k values beating noise: <strong>"
                    f"{e(str(c.good_k or 'none'))}</strong>. Gap statistic "
                    f"chooses k = {e(str(c.gap.attrs.get('chosen_k')))}.</p>")
        body.extend(f"<p class='muted'>{e(s)}</p>" for s in c.stability)
        body.append('<div class="figs">' + _img(figures.get("silhouette.png"))
                    + "</div>")

    # ---- anomalies
    if result.anomalies:
        from .analyse import contrast_reading
        body.append("<h2>Anomalies vs noise</h2>")
        body.append(f"<p>Relative contrast {result.relative_contrast:.3f} "
                    f"({e(contrast_reading(result.relative_contrast))}).</p>")
        body.extend(f"<p>{e(str(a))}</p>" for a in result.anomalies)
        body.append('<div class="figs">' + "".join(
            _img(figures.get(f"anomaly_{a.method}.png"))
            for a in result.anomalies) + "</div>")

    body.append(
        "<footer>Generated by ghostpatterns. Report the ghost rate alongside "
        "every finding: a pattern is only a result once it beats what noise "
        "produces.</footer>")

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Ghost patterns: {e(result.name)}</title>"
        f"<style>{CSS}</style></head><body><main>{''.join(body)}</main>"
        f"<script>{JS}{''.join(scripts)}</script></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
