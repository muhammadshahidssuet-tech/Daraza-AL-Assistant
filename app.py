"""
app.py — Daraz Customer-Support Operations Assistant

A Streamlit chat app that answers internal support questions from a
pre-built FAISS index of Daraz policy PDFs.

IMPORTANT: This app is READ-ONLY with respect to the knowledge base.
It loads faiss_index/index.faiss and faiss_index/metadata.json and never
re-reads, re-chunks, or re-embeds the source PDFs. Run ingest.py separately
whenever the PDFs change.

Setup:
    pip install -r requirements.txt

    .streamlit/secrets.toml:
        GROQ_API_KEY = "gsk_..."

    streamlit run app.py
"""

import json
import os

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

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # must match the model used in ingest.py
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5                 # chunks passed to the LLM
CANDIDATE_MULTIPLIER = 8  # over-fetch factor when a section filter is active

SECTIONS = [
    "returns",
    "delivery",
    "refunds",
    "sellers",
    "payments",
    "customer_support",
]

SECTION_LABELS = {
    "returns": "Returns",
    "delivery": "Delivery",
    "refunds": "Refunds",
    "sellers": "Sellers",
    "payments": "Payments",
    "customer_support": "Customer Support",
}

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
# Page config + branding
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Daraz Support Ops Assistant",
    page_icon="🛍️",
    layout="wide",
)

DARAZ_ORANGE = "#F57224"
DARAZ_DARK = "#1A1A1A"

st.markdown(
    f"""
    <style>
      .daraz-header {{
          background: linear-gradient(90deg, {DARAZ_ORANGE} 0%, #FF9A3C 100%);
          padding: 1.15rem 1.5rem;
          border-radius: 12px;
          margin-bottom: 1.25rem;
      }}
      .daraz-header h1 {{
          color: #FFFFFF;
          font-size: 1.6rem;
          margin: 0;
          font-weight: 700;
          letter-spacing: -0.01em;
      }}
      .daraz-header p {{
          color: rgba(255, 255, 255, 0.92);
          margin: 0.3rem 0 0 0;
          font-size: 0.92rem;
      }}
      .section-pill {{
          display: inline-block;
          background: #FFF1E6;
          color: {DARAZ_ORANGE};
          border: 1px solid #FFD9BF;
          border-radius: 999px;
          padding: 0.15rem 0.7rem;
          font-size: 0.75rem;
          font-weight: 600;
          margin-right: 0.35rem;
      }}
      .source-line {{
          font-size: 0.82rem;
          color: #555;
          padding: 0.15rem 0;
      }}
      div.stButton > button {{
          background-color: {DARAZ_ORANGE};
          color: #FFFFFF;
          border: none;
          border-radius: 8px;
          font-weight: 600;
      }}
      div.stButton > button:hover {{
          background-color: #D95F16;
          color: #FFFFFF;
      }}
      section[data-testid="stSidebar"] {{
          background-color: #FAFAFA;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="daraz-header">
      <h1>🛍️ Daraz Support Ops Assistant</h1>
      <p>Policy answers for support agents, grounded in the internal knowledge base.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders — index and model load once per session, never re-ingested
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        return None, None

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


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
        "**App settings → Secrets** on Streamlit Community Cloud:\n\n"
        "```toml\nGROQ_API_KEY = \"gsk_...\"\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, section: str | None, top_k: int = TOP_K):
    """
    Embed the query and pull the most similar chunks from the pre-built index.

    If `section` is given, we over-fetch candidates and then keep only the
    chunks whose department matches, so the result set stays inside one
    knowledge-base section.
    """
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    if section:
        k = min(top_k * CANDIDATE_MULTIPLIER, index.ntotal)
    else:
        k = min(top_k, index.ntotal)

    scores, ids = index.search(q_vec, k)

    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        record = metadata[idx]
        if section and record.get("department") != section:
            continue
        hits.append({**record, "score": float(score)})
        if len(hits) >= top_k:
            break

    return hits


def build_context(hits):
    """Format retrieved chunks into a numbered context block for the LLM."""
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(
            f"[Excerpt {i}]\n"
            f"Section: {h.get('department')}\n"
            f"Source file: {h.get('source_file')}\n"
            f"Content: {h.get('text')}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, hits, history):
    """Call Groq with the retrieved context and stream the answer back."""
    context = build_context(hits)

    user_message = (
        f"Policy excerpts from the Daraz knowledge base:\n\n"
        f"{context}\n\n"
        f"---\n"
        f"Support agent's question: {query}\n\n"
        f"Answer using only the excerpts above."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include a short window of prior turns so follow-ups keep their thread.
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


# ---------------------------------------------------------------------------
# Sidebar — knowledge base sections
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge base")
    st.caption("Restrict search to a single section, or search everything.")

    scope_options = ["All sections"] + [SECTION_LABELS[s] for s in SECTIONS]
    chosen_label = st.radio(
        "Search scope",
        scope_options,
        index=0,
        label_visibility="collapsed",
    )

    if chosen_label == "All sections":
        active_section = None
    else:
        active_section = next(
            s for s in SECTIONS if SECTION_LABELS[s] == chosen_label
        )

    st.divider()

    st.markdown("### ⚙️ Retrieval")
    top_k = st.slider("Excerpts to retrieve", min_value=3, max_value=10, value=TOP_K)

    st.divider()

    # Index stats, derived from the loaded metadata only.
    counts = {}
    for record in metadata:
        dept = record.get("department", "unknown")
        counts[dept] = counts.get(dept, 0) + 1

    st.markdown("### 📊 Index")
    st.caption(f"{index.ntotal:,} chunks loaded")
    for s in SECTIONS:
        st.caption(f"• {SECTION_LABELS[s]} — {counts.get(s, 0):,}")

    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

scope_text = (
    "all sections"
    if active_section is None
    else SECTION_LABELS[active_section]
)
st.markdown(
    f'<span class="section-pill">Searching: {scope_text}</span>',
    unsafe_allow_html=True,
)
st.write("")

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🛍️" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources used"):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-line">'
                        f'<b>{SECTION_LABELS.get(s["department"], s["department"])}</b> — '
                        f'{s["source_file"]} '
                        f'<span style="color:#999;">(chunk {s["chunk_index"]}, '
                        f'score {s["score"]:.3f})</span></div>',
                        unsafe_allow_html=True,
                    )

prompt = st.chat_input("Ask about a Daraz policy — e.g. 'How long does a COD refund take?'")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    hits = retrieve(prompt, active_section, top_k=top_k)

    with st.chat_message("assistant", avatar="🛍️"):
        if not hits:
            scope_note = (
                "the knowledge base"
                if active_section is None
                else f"the **{SECTION_LABELS[active_section]}** section"
            )
            answer = (
                f"I couldn't find anything relevant in {scope_note}. "
                "Try rephrasing the question, or widen the search scope in the sidebar."
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

            with st.expander("Sources used"):
                for h in hits:
                    st.markdown(
                        f'<div class="source-line">'
                        f'<b>{SECTION_LABELS.get(h["department"], h["department"])}</b> — '
                        f'{h["source_file"]} '
                        f'<span style="color:#999;">(chunk {h["chunk_index"]}, '
                        f'score {h["score"]:.3f})</span></div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": hits}
            )
