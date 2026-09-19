"""
app.py — Daraz Customer-Support Operations Assistant

Streamlit chat app that answers internal support questions from a PRE-BUILT
FAISS index of Daraz policy PDFs.

READ-ONLY knowledge base: loads faiss_index/index.faiss and
faiss_index/metadata.json. Never re-reads, re-chunks, or re-embeds the PDFs.
Run ingest.py separately whenever the source PDFs change.

Folder layout:
    project/
    |- app.py
    |- requirements.txt
    |- assets/                  <- optional images (see ASSET NOTES below)
    |   |- logo.png
    |   |- hero.png
    |   |- returns.png
    |   |- delivery.png
    |   |- refunds.png
    |   |- sellers.png
    |   |- payments.png
    |   |- customer_support.png
    |- faiss_index/
    |   |- index.faiss
    |   |- metadata.json
    |- .streamlit/
        |- secrets.toml         <- GROQ_API_KEY = "gsk_..."

ASSET NOTES:
    Every image is OPTIONAL. If a file is missing, the app falls back to a
    generated inline SVG, so nothing breaks on a fresh clone. Drop in your own
    PNG/JPG/SVG files to replace the fallbacks. Recommended sizes:
        logo.png       ~ 240x80   (transparent background)
        hero.png       ~ 1200x300 (wide banner)
        <section>.png  ~ 128x128  (square icons)
"""

import base64
import json
import os
from datetime import datetime

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INDEX_DIR = "faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")
ASSETS_DIR = "assets"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # MUST match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5
CANDIDATE_MULTIPLIER = 8

DARAZ_ORANGE = "#F57224"
DARAZ_ORANGE_DARK = "#D95F16"
DARAZ_ORANGE_LIGHT = "#FFF1E6"

SECTIONS = ["returns", "delivery", "refunds", "sellers", "payments", "customer_support"]

SECTION_META = {
    "returns":          {"label": "Returns",          "emoji": "\u21a9\ufe0f", "color": "#F57224"},
    "delivery":         {"label": "Delivery",         "emoji": "\U0001f69a", "color": "#2E7D32"},
    "refunds":          {"label": "Refunds",          "emoji": "\U0001f4b8", "color": "#1565C0"},
    "sellers":          {"label": "Sellers",          "emoji": "\U0001f3ea", "color": "#6A1B9A"},
    "payments":         {"label": "Payments",         "emoji": "\U0001f4b3", "color": "#C62828"},
    "customer_support": {"label": "Customer Support", "emoji": "\U0001f3a7", "color": "#00838F"},
}

STARTER_QUESTIONS = [
    "How long does a COD refund take?",
    "What items are non-returnable?",
    "How do I escalate a late delivery?",
    "What are the seller penalty rules?",
]

SYSTEM_PROMPT = """You are the Daraz Customer-Support Operations Assistant.

You help Daraz support agents answer customer questions accurately by relying \
strictly on the internal policy excerpts provided to you.

Rules:
- Answer ONLY from the provided policy excerpts. Do not use outside knowledge \
about Daraz or general e-commerce practice.
- If the excerpts do not contain the answer, say clearly that the policy \
documents provided do not cover it, and suggest which team or document the \
agent should check next.
- Cite the source file for any specific rule, timeframe, or figure you state, \
using the format [source_file].
- Be concise and practical. Support agents are usually mid-conversation with a \
customer, so lead with the direct answer, then add conditions or exceptions.
- Never invent policy numbers, timeframes, fees, or eligibility criteria.
"""


# ---------------------------------------------------------------------------
# Image helpers - every image degrades gracefully to a generated SVG
# ---------------------------------------------------------------------------
def asset_path(filename: str):
    """Return the path to an asset if it exists, else None."""
    path = os.path.join(ASSETS_DIR, filename)
    return path if os.path.exists(path) else None


def find_asset(basename: str):
    """Look for basename with any common image extension."""
    for ext in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
        path = asset_path(basename + ext)
        if path:
            return path
    return None


def svg_to_data_uri(svg: str) -> str:
    """Encode an SVG string as a data URI usable in an <img> tag."""
    encoded = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def file_to_data_uri(path: str) -> str:
    """Encode a local image file as a data URI usable in an <img> tag."""
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".webp": "image/webp",
    }.get(ext, "image/png")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def fallback_logo_svg() -> str:
    """A simple generated wordmark used when assets/logo.* is absent."""
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="240" height="64" viewBox="0 0 240 64">
      <rect x="0" y="8" width="48" height="48" rx="12" fill="#FFFFFF"/>
      <text x="24" y="42" font-family="Segoe UI, Arial, sans-serif" font-size="26"
            font-weight="700" fill="{DARAZ_ORANGE}" text-anchor="middle">D</text>
      <text x="60" y="34" font-family="Segoe UI, Arial, sans-serif" font-size="22"
            font-weight="700" fill="#FFFFFF">daraz</text>
      <text x="60" y="50" font-family="Segoe UI, Arial, sans-serif" font-size="11"
            font-weight="500" fill="rgba(255,255,255,0.85)">SUPPORT OPS</text>
    </svg>
    """


def fallback_section_svg(section: str) -> str:
    """Generated circular icon for a section when no image file is provided."""
    meta = SECTION_META[section]
    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 96 96">
      <circle cx="48" cy="48" r="44" fill="{meta['color']}22"
              stroke="{meta['color']}" stroke-width="2"/>
      <text x="48" y="62" font-size="40" text-anchor="middle">{meta['emoji']}</text>
    </svg>
    """


def logo_uri() -> str:
    path = find_asset("logo")
    return file_to_data_uri(path) if path else svg_to_data_uri(fallback_logo_svg())


def section_icon_uri(section: str) -> str:
    path = find_asset(section)
    return file_to_data_uri(path) if path else svg_to_data_uri(fallback_section_svg(section))


# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Ops Assistant",
    page_icon="\U0001f6cd\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .daraz-header {{
          background: linear-gradient(120deg, {DARAZ_ORANGE} 0%, #FF9A3C 60%, #FFB56B 100%);
          padding: 1.4rem 1.8rem;
          border-radius: 16px;
          margin-bottom: 1.1rem;
          display: flex;
          align-items: center;
          gap: 1.2rem;
          box-shadow: 0 6px 20px rgba(245, 114, 36, 0.28);
      }}
      .daraz-header img {{ height: 58px; }}
      .daraz-header .title {{ color: #fff; font-size: 1.55rem; font-weight: 700; margin: 0; }}
      .daraz-header .sub {{ color: rgba(255,255,255,.92); font-size: .92rem; margin: .25rem 0 0; }}

      .hero-banner img {{
          width: 100%; border-radius: 14px; margin-bottom: 1rem;
          box-shadow: 0 4px 14px rgba(0,0,0,.10);
      }}

      /* Sidebar section cards */
      .kb-card {{
          display: flex; align-items: center; gap: .7rem;
          padding: .55rem .7rem; border-radius: 10px;
          background: #fff; border: 1px solid #ECECEC;
          margin-bottom: .3rem; transition: all .18s ease;
      }}
      .kb-card:hover {{
          transform: translateX(3px);
          border-color: {DARAZ_ORANGE};
          box-shadow: 0 2px 10px rgba(245,114,36,.16);
      }}
      .kb-card.active {{
          background: {DARAZ_ORANGE_LIGHT};
          border-color: {DARAZ_ORANGE};
      }}
      .kb-card img {{ width: 30px; height: 30px; border-radius: 6px; }}
      .kb-card .name {{ font-weight: 600; font-size: .88rem; color: #222; }}
      .kb-card .count {{ font-size: .72rem; color: #888; }}

      .section-pill {{
          display: inline-block; background: {DARAZ_ORANGE_LIGHT};
          color: {DARAZ_ORANGE}; border: 1px solid #FFD9BF;
          border-radius: 999px; padding: .2rem .8rem;
          font-size: .76rem; font-weight: 600;
      }}

      .source-card {{
          border-left: 3px solid {DARAZ_ORANGE};
          background: #FCFCFC; padding: .5rem .75rem;
          border-radius: 0 8px 8px 0; margin-bottom: .35rem;
          transition: background .15s ease;
      }}
      .source-card:hover {{ background: {DARAZ_ORANGE_LIGHT}; }}
      .source-card .f {{ font-weight: 600; font-size: .84rem; color: #222; }}
      .source-card .m {{ font-size: .74rem; color: #888; }}

      .score-bar {{
          height: 5px; border-radius: 3px; background: #EEE;
          margin-top: .35rem; overflow: hidden;
      }}
      .score-bar > div {{
          height: 100%;
          background: linear-gradient(90deg, {DARAZ_ORANGE}, #FFB56B);
      }}

      div.stButton > button {{
          border-radius: 9px; font-weight: 600;
          transition: all .15s ease;
      }}
      div.stButton > button:hover {{
          border-color: {DARAZ_ORANGE}; color: {DARAZ_ORANGE};
          transform: translateY(-1px);
      }}
      div.stButton > button[kind="primary"] {{
          background: {DARAZ_ORANGE}; color: #fff; border: none;
      }}
      div.stButton > button[kind="primary"]:hover {{
          background: {DARAZ_ORANGE_DARK}; color: #fff;
      }}

      section[data-testid="stSidebar"] {{ background-color: #FAFAFA; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header with logo image ----
st.markdown(
    f"""
    <div class="daraz-header">
      <img src="{logo_uri()}" alt="Daraz logo"/>
      <div>
        <p class="title">Support Ops Assistant</p>
        <p class="sub">Policy answers for support agents, grounded in the internal knowledge base.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Optional wide hero banner ----
_hero = find_asset("hero")
if _hero:
    st.markdown(
        f'<div class="hero-banner"><img src="{file_to_data_uri(_hero)}" alt="banner"/></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Cached loaders - load once per session, never re-ingest
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
        return None, None
    idx = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return idx, meta


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    return Groq(api_key=api_key) if api_key else None


index, metadata = load_index_and_metadata()

if index is None:
    st.error(
        f"No pre-built index found. Expected `{INDEX_PATH}` and `{METADATA_PATH}`.\n\n"
        "Run `ingest.py` first, then place the `faiss_index/` folder next to `app.py`."
    )
    st.stop()

embedder = load_embedder()
groq_client = load_groq_client()

if groq_client is None:
    st.error(
        "`GROQ_API_KEY` was not found in Streamlit secrets.\n\n"
        "Add it to `.streamlit/secrets.toml` locally, or under "
        "**App settings -> Secrets** on Streamlit Community Cloud:\n\n"
        '```toml\nGROQ_API_KEY = "gsk_..."\n```'
    )
    st.stop()

# Chunk counts per section, derived from loaded metadata only
SECTION_COUNTS = {}
for _rec in metadata:
    _d = _rec.get("department", "unknown")
    SECTION_COUNTS[_d] = SECTION_COUNTS.get(_d, 0) + 1


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, section, top_k: int = TOP_K):
    """Embed the query and pull the most similar chunks from the pre-built index."""
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    k = min(top_k * CANDIDATE_MULTIPLIER, index.ntotal) if section else min(top_k, index.ntotal)
    scores, ids = index.search(q_vec, k)

    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        rec = metadata[idx]
        if section and rec.get("department") != section:
            continue
        hits.append({**rec, "score": float(score)})
        if len(hits) >= top_k:
            break
    return hits


def build_context(hits):
    return "\n\n".join(
        f"[Excerpt {i}]\nSection: {h.get('department')}\n"
        f"Source file: {h.get('source_file')}\nContent: {h.get('text')}"
        for i, h in enumerate(hits, start=1)
    )


def generate_answer(query: str, hits, history):
    """Stream an answer from Groq using only the retrieved excerpts."""
    user_message = (
        f"Policy excerpts from the Daraz knowledge base:\n\n{build_context(hits)}\n\n"
        f"---\nSupport agent's question: {query}\n\n"
        f"Answer using only the excerpts above."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    stream = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def render_sources(hits, key_prefix=""):
    """Render retrieved chunks as cards with a similarity bar and text preview."""
    for n, h in enumerate(hits):
        meta = SECTION_META.get(h["department"], {"label": h["department"], "emoji": "\U0001f4c4"})
        pct = max(0, min(100, int(h["score"] * 100)))
        st.markdown(
            f"""
            <div class="source-card">
              <div class="f">{meta['emoji']} {meta['label']} &mdash; {h['source_file']}</div>
              <div class="m">chunk {h['chunk_index']} &middot; similarity {h['score']:.3f}</div>
              <div class="score-bar"><div style="width:{pct}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View excerpt text"):
            st.write(h["text"])


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "section" not in st.session_state:
    st.session_state.section = None


# ---------------------------------------------------------------------------
# Sidebar - sections as image cards
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f'<img src="{logo_uri()}" style="width:100%;background:{DARAZ_ORANGE};'
        f'border-radius:10px;padding:.5rem;margin-bottom:1rem;">',
        unsafe_allow_html=True,
    )

    st.markdown("### \U0001f4da Knowledge base")
    st.caption("Pick a section to restrict search, or search everything.")

    if st.button(
        "\U0001f310  All sections",
        use_container_width=True,
        type="primary" if st.session_state.section is None else "secondary",
    ):
        st.session_state.section = None
        st.rerun()

    for s in SECTIONS:
        meta = SECTION_META[s]
        count = SECTION_COUNTS.get(s, 0)
        active = st.session_state.section == s

        st.markdown(
            f"""
            <div class="kb-card {'active' if active else ''}">
              <img src="{section_icon_uri(s)}" alt="{meta['label']}"/>
              <div>
                <div class="name">{meta['label']}</div>
                <div class="count">{count:,} chunks</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            f"Search {meta['label']}",
            key=f"sec_{s}",
            use_container_width=True,
            type="primary" if active else "secondary",
            disabled=count == 0,
        ):
            st.session_state.section = s
            st.rerun()

    st.divider()
    st.markdown("### \u2699\ufe0f Retrieval")
    top_k = st.slider("Excerpts to retrieve", 3, 10, TOP_K)
    show_sources = st.toggle("Show sources with answers", value=True)

    st.divider()
    st.caption(f"\U0001f4ca {index.ntotal:,} chunks indexed")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("\U0001f5d1\ufe0f Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        transcript = "\n\n".join(
            f"{'AGENT' if m['role'] == 'user' else 'ASSISTANT'}: {m['content']}"
            for m in st.session_state.messages
        )
        st.download_button(
            "\u2b07\ufe0f Export",
            data=transcript or "No conversation yet.",
            file_name=f"daraz_support_chat_{datetime.now():%Y%m%d_%H%M}.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not st.session_state.messages,
        )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
active_section = st.session_state.section
scope_text = "All sections" if active_section is None else SECTION_META[active_section]["label"]

top_l, top_r = st.columns([3, 1])
with top_l:
    st.markdown(
        f'<span class="section-pill">\U0001f50d Searching: {scope_text}</span>',
        unsafe_allow_html=True,
    )
with top_r:
    searchable = index.ntotal if active_section is None else SECTION_COUNTS.get(active_section, 0)
    st.markdown(
        f'<div style="text-align:right;color:#888;font-size:.78rem;padding-top:.2rem;">'
        f'{searchable:,} chunks in scope</div>',
        unsafe_allow_html=True,
    )

# ---- Empty state: section tiles + starter question buttons ----
if not st.session_state.messages:
    st.write("")
    cols = st.columns(6)
    for col, s in zip(cols, SECTIONS):
        meta = SECTION_META[s]
        with col:
            st.markdown(
                f'<div style="text-align:center;">'
                f'<img src="{section_icon_uri(s)}" style="width:56px;height:56px;"/>'
                f'<div style="font-size:.74rem;font-weight:600;color:#555;margin-top:.3rem;">'
                f'{meta["label"]}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption("\U0001f4a1 Try one of these to get started:")
    qcols = st.columns(len(STARTER_QUESTIONS))
    for col, q in zip(qcols, STARTER_QUESTIONS):
        with col:
            if st.button(q, key=f"starter_{q}", use_container_width=True):
                st.session_state.pending = q
                st.rerun()

st.write("")

# ---- Replay conversation ----
for i, msg in enumerate(st.session_state.messages):
    avatar = "\U0001f6cd\ufe0f" if msg["role"] == "assistant" else "\U0001f9d1\u200d\U0001f4bc"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") and show_sources:
            with st.expander(f"\U0001f4c4 {len(msg['sources'])} sources used"):
                render_sources(msg["sources"], key_prefix=f"m{i}")
        if msg["role"] == "assistant":
            st.feedback("thumbs", key=f"fb_{i}")

# ---- Input ----
typed = st.chat_input("Ask about a Daraz policy - e.g. 'How long does a COD refund take?'")
prompt = typed or st.session_state.pending
st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="\U0001f9d1\u200d\U0001f4bc"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="\U0001f6cd\ufe0f"):
        with st.status("Searching the knowledge base...", expanded=False) as status:
            hits = retrieve(prompt, active_section, top_k=top_k)
            if hits:
                status.update(
                    label=f"Found {len(hits)} relevant excerpts in {scope_text.lower()}",
                    state="complete",
                )
            else:
                status.update(label="No relevant excerpts found", state="error")

        if not hits:
            scope_note = ("the knowledge base" if active_section is None
                          else f"the **{scope_text}** section")
            answer = (
                f"I couldn't find anything relevant in {scope_note}. "
                "Try rephrasing, or widen the scope to **All sections** in the sidebar."
            )
            st.markdown(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": []}
            )
        else:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            try:
                answer = st.write_stream(generate_answer(prompt, hits, history))
            except Exception as e:
                answer = f"The model request failed: `{e}`"
                st.error(answer)

            if show_sources:
                with st.expander(f"\U0001f4c4 {len(hits)} sources used"):
                    render_sources(hits, key_prefix="live")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits}
            )
    st.rerun()
