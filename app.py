import os
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from src.embedding import VDRMultiEmbedder
from src.pinecone_helper import get_pinecone_client, ensure_index, query_vectors, describe_stats


load_dotenv()

st.set_page_config(page_title="VDR Image Search", layout="wide")

# Sidebar configuration
st.sidebar.header("Settings")
index_name = st.sidebar.text_input("Pinecone Index", os.getenv("INDEX_NAME", "vdr-images"))
namespace = st.sidebar.text_input("Namespace", os.getenv("NAMESPACE", "default"))
dimension = int(st.sidebar.number_input("Embedding Dimension", min_value=128, max_value=1536, value=int(os.getenv("DIMENSION", 1536)), step=64))

# Lazy init embedder and pinecone
if "embedder" not in st.session_state:
    st.session_state.embedder = VDRMultiEmbedder()

if "pinecone_index" not in st.session_state or st.session_state.get("_index_name") != index_name:
    pc = get_pinecone_client()
    st.session_state.pinecone_index = ensure_index(pc, index_name=index_name, dimension=dimension)
    st.session_state._index_name = index_name

st.title("Visual Document Retrieval (vdr-2b-multi-v1)")
st.write("Search your screenshots by semantics across multiple languages.")

# Diagnostics panel
with st.expander("Index diagnostics", expanded=True):
    st.write(f"Index: `{index_name}` | Namespace: `{namespace}` | Dimension: `{dimension}`")
    if st.button("Refresh index stats"):
        stats = describe_stats(st.session_state.pinecone_index, namespace=namespace)
        # Ensure JSON-serializable payload for st.json
        def _jsonable(obj):
            import json
            try:
                json.dumps(obj)
                return obj
            except Exception:
                if isinstance(obj, dict):
                    return {k: _jsonable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_jsonable(x) for x in obj]
                to_dict = getattr(obj, "to_dict", None)
                if callable(to_dict):
                    return _jsonable(to_dict())
                model_dump = getattr(obj, "model_dump", None)
                if callable(model_dump):
                    return _jsonable(model_dump())
                try:
                    return dict(obj)
                except Exception:
                    return str(obj)
        st.json(_jsonable(stats))

query = st.text_input("Enter your query", placeholder="e.g., invoice with due date in October")
search = st.button("Search", type="primary")

if search and query.strip():
    with st.spinner("Embedding query and searching..."):
        q_emb = st.session_state.embedder.encode_queries([query], dimension=dimension)[0]
        try:
            import numpy as np
            qnorm = float((np.linalg.norm(q_emb) if hasattr(q_emb, "__array__") else (q_emb.norm().item() if hasattr(q_emb, "norm") else 0.0)))
        except Exception:
            qnorm = None
        res = query_vectors(st.session_state.pinecone_index, q_emb, top_k=5, namespace=namespace, include_metadata=True)
        matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    if not matches:
        st.info("No results found.")
        if qnorm is not None:
            st.caption(f"Query embedding L2 norm: {qnorm:.6f}")
    else:
        st.subheader("Top results")
        if qnorm is not None:
            st.caption(f"Query embedding L2 norm: {qnorm:.6f}")
        cols = st.columns(5)
        for i, m in enumerate(matches[:5]):
            md = m.get("metadata", {}) if isinstance(m, dict) else (m.metadata or {})
            image_path = md.get("image_path")
            score = m.get("score") if isinstance(m, dict) else getattr(m, "score", None)
            with cols[i % 5]:
                st.caption(f"Score: {score:.3f}" if score is not None else "")
                if image_path and Path(image_path).exists():
                    img = Image.open(image_path)
                    st.image(img, caption=Path(image_path).name, use_container_width=True)
                    if st.button("View full size", key=f"view_{i}"):
                        with st.modal("Preview"):
                            st.image(img, caption=image_path, use_container_width=True)
                else:
                    st.warning(f"Image not found: {image_path}")

st.markdown("---")
st.markdown("Small vectors via Matryoshka: set DIMENSION in .env (e.g., 512) to reduce vector size.")
