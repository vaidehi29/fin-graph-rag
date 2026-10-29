"""
COMPONENT : User interface / demo (Streamlit)
PURPOSE   : Ask questions about the indexed annual reports (Apple, Microsoft,
            IBM, NVIDIA) and get
            cited, verified answers; browse the knowledge graph.
INPUTS    : Neo4j and Gemini settings (NEO4J_*, GEMINI_API_KEY) from st.secrets on
            Streamlit Cloud or .env locally; local embedding model
OUTPUTS   : web UI at http://localhost:8501
DEPENDS ON: streamlit, pyvis, fin_graph_rag.answer, fin_graph_rag.graph_view
PANEL NOTE: The UI adds no logic of its own: answers come from
            fin_graph_rag.answer.ask (same code the evaluation measured), and it
            surfaces that code's checks - confidence, invalid citations,
            unverified numbers - instead of hiding them. Failures are shown,
            never replaced by made-up output.

Run:  streamlit run app.py
"""

from __future__ import annotations

import re
import urllib.error

import streamlit as st

from fin_graph_rag.embedders import load_env

# Streamlit Cloud: NEO4J_*, GEMINI_API_KEY etc. come from st.secrets; locally from
# .env. Loaded before fin_graph_rag.config, which reads settings at import time.
load_env()

from fin_graph_rag.config import ANSWER_MODELS, COMPANIES  # noqa: E402  (needs the settings above)

st.set_page_config(page_title="Financial Graph RAG", page_icon="📊", layout="wide")
HISTORY_SIZE = 5


def masked_connection() -> str:
    """Connection details for error messages - the password is never shown."""
    from fin_graph_rag.graph_builder import connection_settings
    try:
        s = connection_settings()
    except RuntimeError as e:
        return str(e)
    return f"{s['uri']} (user: {s['auth'][0]}, password: ****)"


@st.cache_resource(show_spinner="Connecting to Neo4j...")
def get_driver():
    from neo4j import GraphDatabase

    from fin_graph_rag.graph_builder import connection_settings
    s = connection_settings()
    driver = GraphDatabase.driver(s["uri"], auth=s["auth"])
    driver.verify_connectivity()
    return driver, s["database"]


@st.cache_resource(show_spinner=False)
def get_llm(model: str):
    from fin_graph_rag.llm import GeminiJSON
    return GeminiJSON(model=model)


def source_parts(label: str) -> tuple[str, str, str]:
    """"Apple Annual Report FY2024, p.32" -> ("Apple", "FY2024", "p.32")."""
    m = re.match(r"(.+?) Annual Report (FY\d{4}), (p\.\d+)", label)
    return (m.group(1), m.group(2), m.group(3)) if m else (label, "", "")


def md_text(text: str) -> str:
    """Escape "$" for st.markdown: text between two dollar signs is otherwise
    rendered as a LaTeX formula ("$383,285 million ... $394,328" turned into
    green monospace math in the browser)."""
    return text.replace("$", "\\$")


def confidence_badge(level: str) -> str:
    color, text = {"high": ("#1e8e3e", "High confidence - backed by graph facts"),
                   "medium": ("#b08800", "Medium confidence - text evidence only")}[level]
    return (f"<span style='background:{color};color:white;padding:3px 10px;border-radius:12px;"
            f"font-size:0.85em'>{text}</span>")


def show_result(item: dict, high_only: bool) -> None:
    r = item["result"]
    st.markdown(confidence_badge(r["confidence"]), unsafe_allow_html=True)
    if high_only and r["confidence"] != "high":
        st.info("Hidden by the confidence filter: this answer is not backed by graph facts. "
                "Turn off the filter in the sidebar to see it.")
        return
    st.markdown(md_text(r["answer"]))
    if r["invalid_citations"]:
        st.warning("Cited but not among the retrieved sources: " + "; ".join(r["invalid_citations"]))
    if r["unverified_numbers"]:
        st.warning("Numbers not found in the retrieved evidence: " + ", ".join(r["unverified_numbers"]))
    with st.expander(f"Sources ({len(r['sources'])})"):
        for s in r["sources"]:
            company, year, page = source_parts(s)
            st.markdown(f"- **{company}** - {year} - {page}")
        st.caption(f"Model: {r['model']} · graph facts used: {r['n_graph_facts']}")


# --- Sidebar -----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    chosen = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                            help="Used when a question does not name a company.")
    model = st.selectbox("Answer model (Gemini free tier)", ANSWER_MODELS)
    high_only = st.toggle("Only show high-confidence answers", value=False,
                          help="High = the answer is backed by exact facts from the knowledge graph.")
    st.caption(f"Answers use only the indexed 10-K reports of {', '.join(COMPANIES)} (fiscal years "
               "2023-2025; NVIDIA's fiscal year ends in late January, IBM's on December 31).")

st.title("Financial Graph RAG")
st.caption(f"{', '.join(COMPANIES)} annual reports · knowledge graph + text retrieval · cited answers")

try:
    driver, database = get_driver()
except Exception as e:  # FALLBACK: no offline answers - say what failed.
    st.error(f"Cannot connect to Neo4j at {masked_connection()}: {type(e).__name__}: {e}")
    st.stop()

qa_tab, graph_tab = st.tabs(["Q&A", "Knowledge Graph"])

# --- Q&A -----------------------------------------------------------------------
with qa_tab:
    # A form so pressing Enter in the question box submits, same as clicking Search.
    # The CSS hides Streamlit's "Press Enter to submit form" hint; Enter still works.
    st.markdown("<style>[data-testid='InputInstructions'] {display: none;}</style>", unsafe_allow_html=True)
    with st.form("ask"):
        question = st.text_input("Ask a financial question…",
                                 placeholder="e.g. Compare Apple and Microsoft net income in fiscal 2025")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted and question.strip():
        if not chosen:
            st.warning("Select at least one company in the sidebar.")
        else:
            from fin_graph_rag.answer import ask
            from fin_graph_rag.embedders import QuotaExhausted
            try:
                with st.spinner("Retrieving evidence and generating a cited answer..."):
                    result = ask(question, driver, database, llm=get_llm(model),
                                 default_companies=[COMPANIES[c] for c in chosen])
            except (QuotaExhausted, urllib.error.URLError, RuntimeError, KeyError) as e:
                st.warning(f"The answer model failed: {type(e).__name__}: {e}")
            except Exception as e:  # neo4j driver errors and anything unexpected
                st.error(f"Retrieval failed (Neo4j at {masked_connection()}): {type(e).__name__}: {e}")
            else:
                history = st.session_state.setdefault("history", [])
                history.insert(0, {"question": question, "result": result})
                del history[HISTORY_SIZE:]

    history = st.session_state.get("history", [])
    if history:
        st.subheader(md_text(history[0]["question"]))
        show_result(history[0], high_only)
    if len(history) > 1:
        with st.expander(f"Previous questions ({len(history) - 1})", expanded=False):
            for item in history[1:]:
                st.markdown(f"**{md_text(item['question'])}**")
                show_result(item, high_only)
                st.divider()

# --- Knowledge graph -------------------------------------------------------------
with graph_tab:
    from fin_graph_rag.graph_view import NODE_COLORS, REL_GROUPS, fetch_subgraph, to_html
    c1, c2, c3 = st.columns([1, 2, 1])
    company_name = c1.selectbox("Company", list(COMPANIES))
    groups = c2.multiselect("Show", list(REL_GROUPS), default=["Financial metrics", "Segments & products", "Risks",
                                                               "People", "Competitors"])
    year = c3.selectbox("Fiscal year", ["All", 2025, 2024, 2023])
    limit = st.slider("Max first-hop nodes", 20, 300, 120, step=20)
    st.markdown(" ".join(f"<span style='color:{c}'>●</span> {lbl}" for lbl, c in NODE_COLORS.items()),
                unsafe_allow_html=True)
    try:
        graph = fetch_subgraph(driver, COMPANIES[company_name], groups, None if year == "All" else int(year),
                               limit, database=database)
    except Exception as e:
        st.error(f"Graph query failed (Neo4j at {masked_connection()}): {type(e).__name__}: {e}")
    else:
        st.caption(f"{len(graph['nodes'])} nodes, {len(graph['edges'])} relationships "
                   f"(2 hops from {company_name})")
        if graph["nodes"]:
            st.iframe(to_html(graph), height=680)   # labels are HTML-escaped in to_html
        else:
            st.info("Nothing to show for this selection.")
