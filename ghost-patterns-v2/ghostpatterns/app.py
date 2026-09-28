"""
Interactive web app.

    ghostpatterns app            (or: streamlit run ghostpatterns/app.py)

Upload a CSV, pick the column you want to explain, and see which patterns
survive calibration against noise. Everything runs locally; the data never
leaves the machine unless you deploy the app yourself.
"""

from __future__ import annotations

import html
import io
import os
import tempfile
import zipfile

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from ghostpatterns.analyse import Options, analyse, contrast_reading, save
from ghostpatterns.datasets import planted, pure_noise
from ghostpatterns.report import explain_rule

st.set_page_config(page_title="Ghost Patterns", page_icon="👻",
                   layout="wide", initial_sidebar_state="expanded")

REAL = "#38bdf8"
LIKELY = "#a78bfa"
GHOST = "#f87171"
NULL = "#64748b"
VERDICT_COLORS = alt.Scale(
    domain=["REAL (family-wise)", "likely real (FDR)", "GHOST"],
    range=[REAL, LIKELY, GHOST])

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1250px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1120 0%,#111827 100%);
  border-right: 1px solid rgba(148,163,184,.12); }
[data-testid="stSidebar"] * { color: #e2e8f0; }

.hero { position: relative; overflow: hidden; border-radius: 22px;
  padding: 44px 44px 40px; margin-bottom: 26px;
  background: radial-gradient(circle at 15% 20%, rgba(56,189,248,.28), transparent 45%),
              radial-gradient(circle at 85% 30%, rgba(167,139,250,.30), transparent 45%),
              linear-gradient(135deg,#0f172a 0%,#1e1b4b 100%);
  border: 1px solid rgba(148,163,184,.18); color: #f8fafc; }
.hero h1 { font-size: 46px; font-weight: 800; line-height: 1.1; margin: 0 0 12px;
  letter-spacing: -.02em; color: #f8fafc; }
.hero h1 span { background: linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { font-size: 17px; color: #cbd5e1; max-width: 760px; margin: 0; }
.hero .ghost { position: absolute; right: 36px; top: 18px; font-size: 110px;
  opacity: .16; animation: float 4s ease-in-out infinite; }
@keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-14px) } }
.eyebrow { text-transform: uppercase; letter-spacing: .14em; font-size: 12px;
  font-weight: 700; color: #7dd3fc; margin-bottom: 10px; }

.grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 16px;
  margin: 8px 0 22px; }
.feature { border-radius: 16px; padding: 20px 22px; background: rgba(30,41,59,.55);
  border: 1px solid rgba(148,163,184,.16); color: #e2e8f0;
  transition: transform .15s ease, border-color .15s ease; }
.feature:hover { transform: translateY(-3px); border-color: rgba(56,189,248,.5); }
.feature .icon { font-size: 26px; }
.feature h4 { margin: 8px 0 6px; font-size: 16px; color: #f8fafc; }
.feature p { margin: 0; font-size: 14px; color: #94a3b8; }

.stat { border-radius: 16px; padding: 18px 20px; background: rgba(15,23,42,.72);
  border: 1px solid rgba(148,163,184,.16); position: relative; overflow: hidden; }
.stat:before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px;
  background: var(--accent); }
.stat .label { font-size: 12px; text-transform: uppercase; letter-spacing: .08em;
  color: #94a3b8; font-weight: 600; }
.stat .value { font-size: 34px; font-weight: 800; color: var(--accent); line-height: 1.2; }
.stat .note { font-size: 12.5px; color: #64748b; }

.banner { border-radius: 16px; padding: 18px 22px; margin: 4px 0 20px;
  background: linear-gradient(90deg,rgba(56,189,248,.16),rgba(167,139,250,.12));
  border: 1px solid rgba(56,189,248,.35); color: #e2e8f0; font-size: 16px; }
.banner b { color: #f8fafc; }

.funnel { display: flex; align-items: stretch; gap: 10px; flex-wrap: wrap; margin: 6px 0 24px; }
.funnel .step { flex: 1; min-width: 150px; border-radius: 14px; padding: 14px 16px;
  background: rgba(30,41,59,.55); border: 1px solid rgba(148,163,184,.14); }
.funnel .step .n { font-size: 26px; font-weight: 800; color: #f8fafc; }
.funnel .step .t { font-size: 12.5px; color: #94a3b8; }
.funnel .arrow { align-self: center; color: #475569; font-size: 22px; }

.rulecard { border-radius: 14px; padding: 14px 18px; margin-bottom: 10px;
  background: rgba(15,23,42,.6); border: 1px solid rgba(148,163,184,.14); }
.rulecard .r { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 13.5px;
  color: #e2e8f0; }
.rulecard .m { font-size: 13px; color: #94a3b8; margin-top: 4px; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11.5px;
  font-weight: 700; margin-right: 8px; color: #0f172a; }
.lift { float: right; font-weight: 800; color: #38bdf8; font-size: 18px; }

.steps { counter-reset: s; display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
  gap: 14px; }
.steps div { counter-increment: s; border-radius: 14px; padding: 16px 18px 16px 58px;
  position: relative; background: rgba(30,41,59,.4); border: 1px solid rgba(148,163,184,.12);
  color: #cbd5e1; font-size: 14px; }
.steps div:before { content: counter(s); position: absolute; left: 16px; top: 14px;
  width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center;
  background: linear-gradient(135deg,#38bdf8,#a78bfa); color: #0f172a; font-weight: 800; }
.steps b { color: #f8fafc; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 10px 10px 0 0; padding: 8px 16px; }
div.stButton > button[kind="primary"], .stDownloadButton > button {
  border-radius: 12px; font-weight: 700; border: 0;
  background: linear-gradient(90deg,#0ea5e9,#8b5cf6); color: white; }
div.stButton > button[kind="primary"]:hover { filter: brightness(1.1); }
footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers


def e(s) -> str:
    return html.escape(str(s))


def stat(label: str, value, note: str, accent: str) -> str:
    return (f'<div class="stat" style="--accent:{accent}"><div class="label">'
            f'{e(label)}</div><div class="value">{e(value)}</div>'
            f'<div class="note">{e(note)}</div></div>')


def pill(verdict: str) -> str:
    color = {"REAL (family-wise)": REAL, "likely real (FDR)": LIKELY}.get(
        verdict, GHOST)
    return f'<span class="pill" style="background:{color}">{e(verdict)}</span>'


def zip_folder(folder: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(root, f)
                z.write(full, os.path.relpath(full, folder))
    return buf.getvalue()


def chart_theme(c: alt.Chart) -> alt.Chart:
    return (c.configure_view(strokeWidth=0)
            .configure_axis(gridColor="#1e293b", domainColor="#334155",
                            labelColor="#94a3b8", titleColor="#94a3b8",
                            labelFontSize=11, titleFontSize=12)
            .configure_legend(labelColor="#cbd5e1", titleColor="#94a3b8",
                              orient="bottom")
            .configure(background="transparent"))


def lift_chart(r) -> alt.Chart:
    """Null lift distribution with every validated rule drawn on top."""
    surv = r.survivors.copy()
    surv["meaning"] = [explain_rule(a, c, lf) for a, c, lf in
                       zip(surv.rule, surv.confidence, surv.lift)]
    layers = []
    if len(r.null_lifts):
        null = pd.DataFrame({"lift": r.null_lifts})
        layers.append(alt.Chart(null).mark_bar(color=NULL, opacity=.7).encode(
            x=alt.X("lift:Q", bin=alt.Bin(maxbins=50), title="lift"),
            y=alt.Y("count():Q", title="null rules"),
            tooltip=[alt.Tooltip("count():Q", title="null rules")]))
    layers.append(alt.Chart(surv).mark_rule(strokeWidth=2.5, opacity=.85).encode(
        x="lift:Q",
        color=alt.Color("verdict:N", scale=VERDICT_COLORS, title=None),
        tooltip=["rule", "verdict", alt.Tooltip("lift:Q", format=".3f"),
                 alt.Tooltip("confidence:Q", format=".0%"), "meaning"]))
    if r.null_max is not None:
        layers.append(alt.Chart(pd.DataFrame({"x": [r.null_max]}))
                      .mark_rule(color=GHOST, strokeDash=[6, 4], strokeWidth=2)
                      .encode(x="x:Q", tooltip=[alt.Tooltip(
                          "x:Q", title="strongest ghost", format=".3f")]))
    return chart_theme(alt.layer(*layers).properties(height=320, width="container"))


def top_rules_chart(top: pd.DataFrame) -> alt.Chart:
    t = top.copy()
    t["meaning"] = [explain_rule(a, c, lf) for a, c, lf in
                    zip(t.rule, t.confidence, t.lift)]
    return chart_theme(alt.Chart(t).mark_bar(cornerRadiusEnd=6).encode(
        y=alt.Y("rule:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=420, labelFontSize=11)),
        x=alt.X("lift:Q", title="lift"),
        color=alt.Color("verdict:N", scale=VERDICT_COLORS, title=None),
        tooltip=["rule", "verdict", alt.Tooltip("lift:Q", format=".3f"),
                 alt.Tooltip("confidence:Q", format=".0%"),
                 alt.Tooltip("support:Q", format=".3f"), "count", "meaning"],
    ).properties(height=max(220, 26 * len(t)), width="container"))


def attribution_chart(a: pd.DataFrame) -> alt.Chart:
    a = a.assign(status=np.where(a.load_bearing, "load-bearing",
                                 "decoration"))
    return chart_theme(alt.Chart(a).mark_bar(cornerRadiusEnd=6).encode(
        y=alt.Y("column:N", sort="-x", title=None),
        x=alt.X("drop:Q", title="strong rules lost when column is shuffled"),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["load-bearing", "decoration"], range=[REAL, NULL]),
            title=None),
        tooltip=["column", "drop", alt.Tooltip("p_value:Q", format=".4f"),
                 "status"],
    ).properties(height=max(200, 30 * len(a)), width="container"))


def silhouette_chart(sil: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(sil).encode(x=alt.X("k:O", title="k"))
    band = base.mark_area(color=NULL, opacity=.25).encode(
        y=alt.Y("lo:Q", title="silhouette"), y2="hi:Q").transform_calculate(
        lo="datum.null_mean - 2*datum.null_std",
        hi="datum.null_mean + 2*datum.null_std")
    null = base.mark_line(color=NULL, strokeDash=[5, 4], point=True).encode(
        y="null_mean:Q")
    real = base.mark_line(color=REAL, strokeWidth=3,
                          point=alt.OverlayMarkDef(size=80)).encode(
        y="silhouette:Q",
        tooltip=["k", alt.Tooltip("silhouette:Q", format=".3f"),
                 alt.Tooltip("null_mean:Q", format=".3f"), "beats_noise"])
    return chart_theme((band + null + real).properties(height=300, width="container"))


# ----------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown("## 👻 Ghost Patterns")
    st.caption("Mining always finds something. This finds what's real.")
    source = st.radio("Data", ["Upload a CSV", "Demo: planted rules",
                               "Demo: pure noise"])

df, name = None, None
if source == "Upload a CSV":
    upload = st.sidebar.file_uploader("CSV file", type=["csv"])
    if upload is not None:
        df = pd.read_csv(upload)
        name = os.path.splitext(upload.name)[0]
elif source == "Demo: planted rules":
    df, name = planted(), "planted-demo"
else:
    df, name = pure_noise(), "pure-noise-demo"

# ----------------------------------------------------------------- landing

if df is None:
    st.markdown("""
<div class="hero"><div class="ghost">👻</div>
<div class="eyebrow">Null-calibrated data mining</div>
<h1>Which of your patterns<br>are <span>actually real?</span></h1>
<p>Apriori finds rules in shuffled data. k-means finds clusters in static.
Anomaly detectors flag their quota no matter what. Ghost Patterns reruns your
mining on structureless copies of your data and keeps only what beats them.</p>
</div>
<div class="grid">
<div class="feature"><div class="icon">🎯</div><h4>Calibrated rules</h4>
<p>Every rule is tested against hundreds of noise replicates, with FDR and
family-wise control.</p></div>
<div class="feature"><div class="icon">🚨</div><h4>Leakage detection</h4>
<p>Catches patterns that are true by construction before they fool you.</p></div>
<div class="feature"><div class="icon">🧬</div><h4>Column attribution</h4>
<p>Shows which columns carry the structure and which are decoration.</p></div>
<div class="feature"><div class="icon">📦</div><h4>Shareable report</h4>
<p>One-click interactive HTML report and a ZIP of every table and figure.</p></div>
</div>
<div class="steps">
<div><b>Upload a CSV</b> in the sidebar, or pick a demo dataset.</div>
<div><b>Pick a column to explain</b>, such as a label or outcome.</div>
<div><b>Run the analysis</b> and see which patterns survive the noise.</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------- settings

cols = list(df.columns)
default_target = {"planted-demo": "delay"}.get(name)
with st.sidebar:
    target = st.selectbox(
        "Column to explain", ["(none: test all rules)"] + cols,
        index=cols.index(default_target) + 1 if default_target in cols else 0,
        help="Only test rules that predict this column. Far sharper results.")
    target = None if target.startswith("(none") else target

    with st.expander("⚙️ Settings"):
        n_null = st.slider("Null replicates", 20, 300, 60, 10,
                           help="More replicates = finer p-values, slower. "
                                "Use 100+ for final results.")
        min_support = st.slider("Min support", 0.01, 0.30, 0.05, 0.01)
        min_conf = st.slider("Min confidence", 0.30, 0.95, 0.60, 0.05)
        min_lift = st.slider("Min lift", 1.0, 3.0, 1.2, 0.1)
        max_len = st.select_slider("Max items per rule", [2, 3, 4], 3)
        alpha = st.select_slider("Significance level", [0.01, 0.05, 0.10],
                                 0.05)
        attribution = st.checkbox("Column attribution (slower)", True)
        drop_leaking = st.checkbox("Drop leaking columns automatically", False)

    go = st.button("🚀 Run analysis", type="primary", width="stretch")

st.markdown(f"""
<div class="hero" style="padding:28px 34px">
<div class="eyebrow">Dataset</div>
<h1 style="font-size:34px">{e(name)}</h1>
<p>{len(df):,} rows &middot; {df.shape[1]} columns
{f"&middot; explaining <b>{e(target)}</b>" if target else ""}</p></div>
""", unsafe_allow_html=True)
with st.expander("🔍 Preview data"):
    st.dataframe(df.head(100), width="stretch")

key = (name, df.shape, target, n_null, min_support, min_conf, min_lift,
       max_len, alpha, attribution, drop_leaking)
if go:
    opts = Options(target=target, n_null=n_null, min_support=min_support,
                   min_confidence=min_conf, min_lift=min_lift,
                   max_len=max_len, alpha=alpha, attribution=attribution,
                   drop_leaking=drop_leaking)
    lines: list[str] = []
    with st.status("Hunting ghosts...", expanded=True) as status:
        bar = st.progress(0.0)
        try:
            result = analyse(df, opts, name=name, log=lines.append,
                             progress=lambda f, s: (bar.progress(min(f, 1.0)),
                                                    status.update(label=s)))
        except ValueError as err:
            status.update(label="Failed", state="error")
            st.error(str(err))
            st.stop()
        status.update(label="Writing reports")
        outdir = tempfile.mkdtemp(prefix="ghostpatterns_")
        paths = save(result, outdir)
        status.update(label=f"Done in {result.seconds:.0f}s", state="complete",
                      expanded=False)
    st.session_state.update(result=result, paths=paths, outdir=outdir,
                            log="\n".join(lines), key=key)

if "result" not in st.session_state:
    st.info("Choose settings in the sidebar, then press **🚀 Run analysis**.")
    st.stop()
if st.session_state.key != key:
    st.warning("Settings changed since the last run. Press **Run analysis** "
               "to update.")

result = st.session_state.result
paths = st.session_state.paths
r = result.rules

# ----------------------------------------------------------------- results

st.markdown(f'<div class="banner">✨ <b>Verdict.</b> {e(result.headline())}'
            "</div>", unsafe_allow_html=True)

cards = []
if r is not None:
    cards += [stat("Naive rules", f"{r.n_naive:,}", "what plain Apriori says",
                   "#cbd5e1"),
              stat("Ghost rate", f"{r.ghost_rate:.1f}",
                   "rules found in pure noise", GHOST),
              stat("Validated", f"{len(r.survivors):,}", "beat the null", REAL)]
if result.leakage is not None:
    n_hard = len(result.leakage.hard)
    cards.append(stat("Hard leakage", n_hard, "true-by-construction risks",
                      "#fbbf24" if n_hard else REAL))
if result.clusters is not None:
    k = result.clusters.good_k
    cards.append(stat("Real clusters", k[0] if k else "none",
                      "best k beating noise", REAL if k else GHOST))
st.markdown(f'<div class="grid">{"".join(cards)}</div>',
            unsafe_allow_html=True)

tabs = st.tabs(["🎯 Findings", "🧬 Columns", "🚨 Leakage",
                "🔵 Clusters & anomalies", "📜 Log", "📦 Download"])

with tabs[0]:
    if r is None or r.table.empty:
        st.write("No rules to show. Try lowering min support or min lift.")
    else:
        st.markdown(f"""
<div class="funnel">
<div class="step"><div class="n">{r.n_naive:,}</div><div class="t">naive rules</div></div>
<div class="arrow">›</div>
<div class="step"><div class="n">{r.n_candidates:,}</div><div class="t">
{"predict the target" if target else "candidates"}</div></div>
<div class="arrow">›</div>
<div class="step"><div class="n">{r.n_distinct:,}</div><div class="t">distinct after redundancy filter</div></div>
<div class="arrow">›</div>
<div class="step" style="border-color:{REAL}"><div class="n" style="color:{REAL}">
{len(r.survivors):,}</div><div class="t">beat {result.options.n_null} noise replicates</div></div>
</div>""", unsafe_allow_html=True)

        top = result.top_findings(n=8)
        if not top.empty:
            st.markdown("#### Headline findings")
            html_cards = "".join(
                f'<div class="rulecard"><span class="lift">{x.lift:.2f}×</span>'
                f'{pill(x.verdict)}<span class="r">{e(x.rule)}</span>'
                f'<div class="m">{e(explain_rule(x.rule, x.confidence, x.lift))}'
                "</div></div>" for x in top.itertuples())
            st.markdown(html_cards, unsafe_allow_html=True)

        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown("#### Strongest validated rules")
            if not r.survivors.empty:
                st.altair_chart(top_rules_chart(result.top_findings(n=16)),
                                width="stretch")
        with c2:
            st.markdown("#### Real rules vs the ghost distribution")
            if len(r.null_lifts):
                st.caption("Grey: lifts that noise produces. Lines: "
                           "validated rules. Red dashes: the strongest ghost.")
                st.altair_chart(lift_chart(r), width="stretch")
            else:
                st.markdown(
                    f'<div class="banner">👻 <b>Noise found nothing.</b> '
                    f"{result.options.n_null} shuffled copies of this data "
                    "produced zero rules at these thresholds, so every "
                    "validated rule clears a bar that noise never reaches."
                    "</div>", unsafe_allow_html=True)

        st.markdown("#### All rules")
        f1, f2 = st.columns([1, 2])
        verdicts = f1.multiselect(
            "Verdicts", sorted(r.table.verdict.unique()),
            default=[v for v in r.table.verdict.unique() if v != "GHOST"])
        q = f2.text_input("Search", placeholder="e.g. odor=f")
        t = r.table[r.table.verdict.isin(verdicts)]
        if q:
            t = t[t.rule.str.contains(q, case=False, regex=False)]
        t = t.assign(meaning=[explain_rule(a, c, lf) for a, c, lf in
                              zip(t.rule, t.confidence, t.lift)])
        st.dataframe(
            t[["rule", "verdict", "lift", "confidence", "support", "count",
               "fwer_p", "meaning"]],
            width="stretch", hide_index=True,
            height=min(420, 38 + 35 * max(len(t), 1)),
            column_config={
                "confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.2f"),
                "lift": st.column_config.NumberColumn(format="%.3f"),
                "support": st.column_config.NumberColumn(format="%.4f"),
                "fwer_p": st.column_config.NumberColumn("FWER p",
                                                        format="%.4f"),
            })

with tabs[1]:
    if result.attribution is None:
        st.write("Column attribution did not run.")
    else:
        st.caption("Each column is shuffled on its own. The drop is how many "
                   "strong rules disappear. No drop means the column is "
                   "decoration.")
        st.altair_chart(attribution_chart(result.attribution), width="stretch")
        st.dataframe(result.attribution, width="stretch", hide_index=True)

with tabs[2]:
    rep = result.leakage
    if rep is None or not rep.findings:
        st.success("No leakage detected.")
    else:
        if rep.hard:
            st.warning(f"{len(rep.hard)} hard leakage issue(s): rules that "
                       "use these columns may be true by construction.")
        else:
            st.success("No hard leakage. Advisory pairs below are strong "
                       "dependencies worth a look.")
        st.dataframe(rep.to_frame(), width="stretch", hide_index=True,
                     column_config={"severity": st.column_config.ProgressColumn(
                         "severity", min_value=0.0, max_value=1.0,
                         format="%.3f")})

with tabs[3]:
    if result.clusters is None and not result.anomalies:
        st.info("Clusters and anomalies need at least two numeric columns.")
    if result.clusters is not None:
        c = result.clusters
        st.markdown("#### Clustering: observed vs noise")
        st.caption(f"k values beating noise: {c.good_k or 'none'}. Gap "
                   f"statistic chooses k = {c.gap.attrs.get('chosen_k')}.")
        st.altair_chart(silhouette_chart(c.silhouette), width="stretch")
        for s in c.stability:
            st.caption(s)
    if result.anomalies:
        st.markdown("#### Anomalies")
        st.caption(f"Relative contrast {result.relative_contrast:.3f} "
                   f"({contrast_reading(result.relative_contrast)}).")
        acols = st.columns(len(result.anomalies))
        for col, a in zip(acols, result.anomalies):
            col.markdown(stat(a.method.replace("_", " "), f"{a.excess_rate:.1%}",
                              f"flagged vs {a.null_rate:.1%} in noise, "
                              f"p={a.p_value:.3f}",
                              REAL if a.real else GHOST),
                         unsafe_allow_html=True)
            img = paths.get(f"anomaly_{a.method}.png")
            if img:
                col.image(img)

with tabs[4]:
    st.code(st.session_state.log, language=None)

with tabs[5]:
    d1, d2 = st.columns(2)
    with open(paths["report.html"], "rb") as f:
        d1.download_button("📄 Interactive HTML report", f.read(),
                           file_name=f"{name}_report.html", mime="text/html",
                           width="stretch")
    d2.download_button("🗂️ Everything (CSVs, figures, reports) as ZIP",
                       zip_folder(st.session_state.outdir),
                       file_name=f"{name}_ghostpatterns.zip",
                       mime="application/zip", width="stretch")
