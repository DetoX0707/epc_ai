from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from epc_ai import (  # noqa: E402
    HybridKnowledgeBase,
    audit_vendor_submittal,
    extract_text_by_page,
    load_sample_corpus,
    polish_report_with_openai,
    serialize_kb,
)


SAMPLE_DIR = ROOT / "data" / "sample"
SCENARIOS: Dict[str, str] = {
    "Cooling Tower CT-04 - thermal/acoustic non-compliance": "vendor_cooling_tower_submittal.txt",
    "UPS-01 - autonomy and efficiency deviation": "vendor_ups_submittal.txt",
    "GEN-02 - fuel autonomy and event log risk": "vendor_generator_submittal.txt",
}


def init_state() -> None:
    if "kb" not in st.session_state:
        st.session_state.kb = HybridKnowledgeBase()
        load_sample_corpus(st.session_state.kb, SAMPLE_DIR)
    if "last_audit" not in st.session_state:
        st.session_state.last_audit = None
    if "last_results" not in st.session_state:
        st.session_state.last_results = []


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DIR / name)


def save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --navy: #101820;
          --ink: #17202a;
          --muted: #667282;
          --line: #d9e1ea;
          --blue: #2057a8;
          --yellow: #f1c232;
          --green: #1f8a5b;
          --red: #b3261e;
          --panel: #ffffff;
          --bg: #f5f7fa;
        }
        .stApp { background: var(--bg); color: var(--ink); }
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1440px; }
        [data-testid="stSidebar"] { background: #101820; }
        [data-testid="stSidebar"] * { color: #eef4fa; }
        [data-testid="stSidebar"] [data-testid="stMetric"] {
          background: #ffffff;
          border: 1px solid #d9e1ea;
        }
        [data-testid="stSidebar"] [data-testid="stMetric"] * {
          color: #17202a !important;
        }
        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button * {
          color: #17202a !important;
        }
        [data-testid="stMetric"] {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 8px;
          padding: 16px;
          box-shadow: 0 8px 22px rgba(16, 24, 32, 0.06);
        }
        .hero {
          border-radius: 10px;
          padding: 24px 26px;
          background: linear-gradient(135deg, #101820 0%, #173d75 68%, #2057a8 100%);
          color: #fff;
          border: 1px solid rgba(255,255,255,0.08);
          margin-bottom: 16px;
        }
        .hero h1 { margin: 0 0 8px; font-size: 2.1rem; line-height: 1.1; }
        .hero p { color: #d9e6f3; margin: 0; max-width: 920px; }
        .chip {
          display: inline-block;
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(241, 194, 50, 0.16);
          color: #ffdd62;
          border: 1px solid rgba(241, 194, 50, 0.36);
          font-size: 0.82rem;
          font-weight: 700;
          margin-bottom: 12px;
        }
        .card {
          background: #fff;
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 18px;
          min-height: 132px;
          box-shadow: 0 8px 24px rgba(16, 24, 32, 0.06);
        }
        .card h3 { margin: 0 0 8px; }
        .card p { margin: 0; color: var(--muted); line-height: 1.45; }
        .risk-high { color: var(--red); font-weight: 800; }
        .risk-watch { color: #a96600; font-weight: 800; }
        .risk-ok { color: var(--green); font-weight: 800; }
        .source-note {
          background: #eef4fb;
          border-left: 4px solid var(--blue);
          padding: 12px 14px;
          border-radius: 6px;
          color: #25384d;
          margin-top: 12px;
        }
        div[data-testid="stTabs"] button p { font-weight: 800; }
        .verdict {
          border-radius: 8px;
          padding: 14px 16px;
          background: #fff5f4;
          border: 1px solid #f2c8c5;
          color: var(--red);
          font-weight: 900;
          font-size: 1.2rem;
        }
        .verdict.ok {
          background: #edf8f2;
          border-color: #bde3ce;
          color: var(--green);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>SABAAAAA EPC Intelligence</h1>
          <p>AI-assisted project control tower for submittal QA, RFI search, commissioning readiness,
          and procurement risk evidence across data-centre construction packages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## SABAA EPC")
        st.caption("Data Centre EPC AI")
        st.divider()
        status = serialize_kb(st.session_state.kb)
        st.metric("Indexed documents", len(status["documents"]))
        st.metric("Evidence chunks", status["chunks"])
        if st.button("Reload Demo", use_container_width=True):
            load_sample_corpus(st.session_state.kb, SAMPLE_DIR)
            st.session_state.last_results = []
            st.session_state.last_audit = None
            st.success("Demo reloaded.")
        st.divider()
        st.markdown("**Demo questions**")
        st.caption("UPS runtime requirement")
        st.caption("RFI-026 acoustic exception")
        st.caption("Cooling tower BACnet gateway")
        st.caption("Generator fuel autonomy")


def render_kpis() -> None:
    equipment = load_csv("equipment_register.csv")
    risks = load_csv("risk_register.csv")
    cxs = load_csv("commissioning_status.csv")
    docs = serialize_kb(st.session_state.kb)
    blocked = int((equipment["qa_gate"] == "Blocked").sum())
    avg_readiness = int(cxs["readiness"].mean())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Packages", len(equipment), "+6 demo")
    col2.metric("Blocked QA Gates", blocked, "needs action")
    col3.metric("Open Risk Score", int(risks["score"].sum()))
    col4.metric("Cx Readiness", f"{avg_readiness}%")
    col5.metric("Evidence Chunks", docs["chunks"])


def style_risk_score(value: int) -> str:
    if value >= 20:
        return "background-color: #fdebea; color: #8f1d17; font-weight: 800"
    if value >= 12:
        return "background-color: #fff3df; color: #7a4b00; font-weight: 800"
    return "background-color: #edf8f2; color: #176a45; font-weight: 800"


def render_dashboard() -> None:
    render_kpis()
    st.markdown("### Project Control Tower")
    left, middle, right = st.columns([1.1, 1, 1])

    equipment = load_csv("equipment_register.csv")
    risks = load_csv("risk_register.csv")
    cxs = load_csv("commissioning_status.csv")

    with left:
        st.markdown("#### Critical Equipment")
        st.dataframe(
            equipment[["tag", "package", "vendor", "status", "lead_time_weeks", "qa_gate"]],
            use_container_width=True,
            hide_index=True,
        )

    with middle:
        st.markdown("#### Top Risks")
        top_risks = risks.sort_values("score", ascending=False).head(5)
        st.dataframe(
            top_risks[["risk_id", "area", "score", "status", "owner"]],
            use_container_width=True,
            hide_index=True,
            column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=25)},
        )

    with right:
        st.markdown("#### Commissioning Readiness")
        st.dataframe(
            cxs[["test_id", "system", "readiness", "blocking_item"]],
            use_container_width=True,
            hide_index=True,
            column_config={"readiness": st.column_config.ProgressColumn("readiness", min_value=0, max_value=100)},
        )

    st.markdown(
        """
        <div class="source-note">
        Domain anchor: Uptime Institute describes Tier III facilities as concurrently maintainable,
        with redundant components and distribution paths for power and cooling. The demo data mirrors
        that reality by treating UPS, generator, cooling, EPMS, and commissioning evidence as linked QA gates.
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_search_box() -> None:
    presets = [
        "What backup runtime is required for Tier III UPS systems under full load?",
        "What did RFI-026 decide about cooling tower acoustic exception?",
        "Is BACnet optional for cooling tower controls?",
        "What is the generator fuel autonomy requirement?",
        "Which procurement item is blocked by CT-04 non-compliance?",
    ]
    selected = st.selectbox("Fast demo prompt", presets)
    query = st.text_area("Question", value=selected, height=120)
    if st.button("Run Hybrid Search", type="primary"):
        answer = st.session_state.kb.answer(query, top_k=6)
        st.session_state.last_results = answer["results"]
        st.subheader("Cited Answer")
        st.code(answer["answer"], language="text")


def render_audit() -> None:
    scenario = st.selectbox("Built-in vendor scenario", list(SCENARIOS.keys()))
    upload = st.file_uploader("Or upload vendor submittal PDF/text", type=["pdf", "txt", "md"])

    col1, col2, col3 = st.columns(3)
    if col1.button("Run Scenario Audit", type="primary", use_container_width=True):
        vendor_text = (SAMPLE_DIR / SCENARIOS[scenario]).read_text(encoding="utf-8")
        st.session_state.last_audit = audit_vendor_submittal(vendor_text, st.session_state.kb)

    if col2.button("Audit Uploaded File", use_container_width=True):
        if upload is None:
            st.warning("Choose a vendor submittal first.")
        else:
            path = save_upload(upload)
            vendor_text = "\n".join(text for _, text in extract_text_by_page(path))
            st.session_state.last_audit = audit_vendor_submittal(vendor_text, st.session_state.kb)

    if col3.button("Polish Executive Report", use_container_width=True):
        if st.session_state.last_audit is None:
            st.warning("Run an audit first.")
        else:
            polished = polish_report_with_openai(st.session_state.last_audit)
            st.session_state.last_audit["executive_report"] = f"{polished['report']}\n\n{polished['note']}"

    if not st.session_state.last_audit:
        st.info("Run a scenario audit to generate a cited QA report.")
        return

    audit = st.session_state.last_audit
    verdict_class = "ok" if audit["verdict"] == "Compliant" else ""
    st.markdown(f"<div class='verdict {verdict_class}'>Verdict: {audit['verdict']} | {audit['equipment']}</div>", unsafe_allow_html=True)
    st.progress(int(audit["confidence"] * 100), text=f"Evidence confidence: {int(audit['confidence'] * 100)}%")

    summary = audit["summary"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Critical", summary["critical"])
    m2.metric("Missing", summary["missing"])
    m3.metric("Compliant", summary["compliant"])
    m4.metric("Total Checks", summary["total"])

    st.markdown("### Executive QA Summary")
    st.code(audit.get("executive_report", ""), language="text")

    st.markdown("### Findings")
    for item in audit["findings"]:
        expanded = item["status"] != "Compliant"
        with st.expander(f"{item['parameter']} - {item['status']}", expanded=expanded):
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.write(f"**Vendor:** {item['vendor']}")
            c2.write(f"**Required:** {item['required']}")
            c3.write(f"**Severity:** {item['severity']}")
            st.write(f"**Evidence:** {item['evidence']}")
            st.write(f"**Action:** {item['recommendation']}")

    st.markdown("### Source Evidence")
    for citation in audit.get("citations", []):
        st.markdown(f"**{citation['doc']} | Page {citation['page']} | {citation['section']}**")
        st.caption(citation["text"])


def render_risk_board() -> None:
    risks = load_csv("risk_register.csv")
    rfi = load_csv("rfi_status.csv")
    st.markdown("### Risk Register")
    st.dataframe(
        risks.style.map(style_risk_score, subset=["score"]),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("### RFI Impact Ledger")
    st.dataframe(
        rfi,
        use_container_width=True,
        hide_index=True,
        column_config={"rework_hours_saved": st.column_config.ProgressColumn("rework_hours_saved", min_value=0, max_value=20)},
    )


def render_evidence() -> None:
    results = st.session_state.get("last_results", [])
    if not results:
        st.info("Run a search first to populate retrieved evidence.")
        return
    for result in results:
        st.markdown(f"**{result['doc']} | Page {result['page']} | {result['section']}**")
        st.write(result["text"])
        st.caption(f"Dense score {result['dense_score']} | Sparse score {result['sparse_score']} | Fused {result['score']}")
        st.divider()


def render_architecture() -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h3>Supervisor Agent</h3><p>Routes questions, uploads, scenario audits, and report-polish requests into the right workflow.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h3>Knowledge Agent</h3><p>Searches master specs, RFIs, change orders, commissioning checklists, and procurement trackers using hybrid retrieval.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h3>Compliance Auditor</h3><p>Extracts submittal parameters, retrieves project truth, compares values, and creates a cited QA report.</p></div>", unsafe_allow_html=True)

    
    
    


def render_demo_data() -> None:
    st.markdown("### Demo Data Files")
    files = sorted(SAMPLE_DIR.glob("*"))
    for path in files:
        if path.is_file():
            st.write(f"- `{path.name}`")

    st.markdown("### Equipment Register")
    st.dataframe(load_csv("equipment_register.csv"), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="SABAA EPC Intelligence", page_icon="DC", layout="wide")
    init_state()
    inject_css()
    render_sidebar()
    render_hero()

    tabs = st.tabs(["Command Center", "Ask", "Audit", "Risk Board", "Evidence", "Architecture", "Demo Data"])
    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        st.markdown("### Project Knowledge Agent")
        uploaded_master = st.file_uploader("Add master PDF/text document", type=["pdf", "txt", "md"], key="master_upload")
        if uploaded_master and st.button("Ingest Document"):
            path = save_upload(uploaded_master)
            chunks = st.session_state.kb.ingest_file(path)
            st.success(f"Indexed {uploaded_master.name}: {chunks} evidence chunks added.")
        run_search_box()
    with tabs[2]:
        render_audit()
    with tabs[3]:
        render_risk_board()
    with tabs[4]:
        render_evidence()
    with tabs[5]:
        render_architecture()
    with tabs[6]:
        render_demo_data()


if __name__ == "__main__":
    main()
