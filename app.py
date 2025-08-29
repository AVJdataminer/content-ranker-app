
import os
import json
import math
import time
import pathlib
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# OpenAI embeddings support
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # openai>=1.0.0
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


APP_TITLE = "Ask Dolly — Community Content Ranker (Demo)"
DESCRIPTION = """
This demo ranks community posts/questions by **relevance**, **helpfulness**, and **trustworthiness**
to produce a feed tailored to a topic or query.

- **Relevance**: semantic similarity to a user query.
- **Helpfulness**: presence of actionable tips, concrete numbers, step-by-step guidance.
- **Trustworthiness**: signs of source quality (links to .gov/.edu, key finance terms) and reduced hype.
"""

DEFAULT_TOPIC = "Beginner investing and budgeting for recent grads"


# ---------------------- Embeddings ----------------------
def embed_texts_openai(texts: List[str], api_key: str) -> np.ndarray:
    """Get embeddings from OpenAI using provided API key."""
    client = OpenAI(api_key=api_key)
    model = "text-embedding-3-small"  # Default model
    # OpenAI wants input as list of strings
    resp = client.embeddings.create(model=model, input=texts)
    arr = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # Normalize for cosine similarity stability
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr / norms


def embed_texts_tfidf(texts: List[str], fit_vectorizer: TfidfVectorizer = None):
    """TF-IDF embeddings for offline / no-API mode."""
    if fit_vectorizer is None:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        X = vec.fit_transform(texts)
        return X, vec
    else:
        X = fit_vectorizer.transform(texts)
        return X, fit_vectorizer


# ---------------------- Heuristics ----------------------
FINANCE_KEYWORDS = set([
    "apr","apy","roi","brokerage","index fund","expense ratio","etf","401k","ira","roth","capital gains",
    "dividend","rebalance","budget","cash flow","savings rate","debt","credit score","emergency fund",
    "compound","interest","tax","treasury","cd","certificate of deposit","dca","dollar-cost averaging"
])

HYPE_WORDS = set(["guaranteed","moon","to the moon","risk-free","secret","trick","hack","get rich quick","insider"])


def helpfulness_score(text: str) -> float:
    """Rule-based: more numbers, steps, and 'how to' hints -> higher score."""
    t = text.lower()
    num_tokens = sum(ch.isdigit() for ch in t)
    has_steps = any(k in t for k in ["step", "1)", "2)", "first,", "second,", "third,", "bullet", "- "])
    has_howto = any(k in t for k in ["how to", "you can", "do this", "here’s how", "here is how", "here is what to do"])
    score = 0.0
    score += min(num_tokens / 10.0, 1.0) * 0.5
    score += (1.0 if has_steps else 0.0) * 0.3
    score += (1.0 if has_howto else 0.0) * 0.2
    return float(score)


def trustworthiness_score(text: str) -> float:
    """Rule-based: presence of credible links and finance terms; penalize hype words."""
    t = text.lower()
    has_gov_edu = (".gov" in t) or (".edu" in t)
    has_keywords = sum(1 for kw in FINANCE_KEYWORDS if kw in t)
    hype_hits = sum(1 for hw in HYPE_WORDS if hw in t)
    score = 0.0
    score += 0.5 if has_gov_edu else 0.0
    score += min(has_keywords / 5.0, 1.0) * 0.5
    score -= min(hype_hits * 0.25, 0.75)  # penalize hype
    return float(max(0.0, min(1.0, score)))


# ---------------------- Ranking ----------------------
def rank_posts(posts: List[Dict[str, Any]], query: str, w_rel: float, w_help: float, w_trust: float, openai_api_key: str = None) -> pd.DataFrame:
    texts = [p.get("text", "") for p in posts]
    # embeddings for relevance
    if openai_api_key and OPENAI_AVAILABLE:
        try:
            embs = embed_texts_openai(texts + [query], openai_api_key)
            post_embs = embs[:-1]
            query_emb = embs[-1:]
            rel = (post_embs @ query_emb.T).squeeze()  # cosine similarity thanks to normalization
        except Exception as e:
            st.warning(f"OpenAI embeddings failed, falling back to TF‑IDF. Error: {e}")
            X, vec = embed_texts_tfidf(texts)
            qX, _ = embed_texts_tfidf([query], fit_vectorizer=vec)
            rel = cosine_similarity(X, qX).squeeze()
    else:
        X, vec = embed_texts_tfidf(texts)
        qX, _ = embed_texts_tfidf([query], fit_vectorizer=vec)
        rel = cosine_similarity(X, qX).squeeze()

    # normalize relevance to [0,1]
    if np.max(rel) > np.min(rel):
        rel_norm = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
    else:
        rel_norm = np.zeros_like(rel)

    help_scores = np.array([helpfulness_score(t) for t in texts])
    trust_scores = np.array([trustworthiness_score(t) for t in texts])

    final = w_rel * rel_norm + w_help * help_scores + w_trust * trust_scores

    df = pd.DataFrame({
        "title": [p.get("title", "") for p in posts],
        "text": texts,
        "relevance": rel_norm,
        "helpfulness": help_scores,
        "trustworthiness": trust_scores,
        "score": final
    })
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---------------------- Streamlit UI ----------------------
def load_posts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.write(DESCRIPTION)
    
    sample_path = pathlib.Path(__file__).parent / "data" / "sample_posts.json"
    if not sample_path.exists():
        st.error("Missing data/sample_posts.json")
        return

    posts = load_posts(str(sample_path))

    with st.sidebar:
        st.header("Controls")
        query = st.text_input("Community topic or query", value=DEFAULT_TOPIC)
        st.caption("Tip: Try 'Student loan repayment strategies' or 'First credit card and credit score'")

        st.subheader("Weights")
        w_rel = st.slider("Relevance weight", 0.0, 1.0, 0.5, 0.05)
        w_help = st.slider("Helpfulness weight", 0.0, 1.0, 0.25, 0.05)
        w_trust = st.slider("Trustworthiness weight", 0.0, 1.0, 0.25, 0.05)

        st.subheader("OpenAI Embeddings (Optional)")
        
        if OPENAI_AVAILABLE:
            # Check for environment variable first
            env_api_key = os.getenv("OPENAI_API_KEY")
            if env_api_key:
                st.success("✅ Using OpenAI API key from environment variable")
                openai_api_key = env_api_key
            else:
                # API key input
                openai_api_key = st.text_input(
                    "OpenAI API Key", 
                    type="password", 
                    placeholder="sk-...",
                    help="Enter your OpenAI API key for better relevance scoring"
                )
                
                if openai_api_key:
                    st.success("✅ API key provided - using OpenAI embeddings")
                else:
                    st.info("💡 Using TF-IDF embeddings (faster, but less accurate)")
                
                # Instructions for getting API key
                with st.expander("🔑 How to get an OpenAI API Key"):
                    st.markdown("""
                    **Step 1:** Visit [platform.openai.com](https://platform.openai.com)
                    
                    **Step 2:** Sign up or log in to your account
                    
                    **Step 3:** Go to the API section and click "Create new secret key"
                    
                    **Step 4:** Copy the key (starts with 'sk-') and paste it above
                    
                    **Cost:** Text embeddings are very affordable (~$0.10 per million tokens)
                    
                    **Why use it?** OpenAI embeddings provide much better semantic understanding compared to TF-IDF, leading to more accurate relevance scoring.
                    """)
        else:
            st.warning("⚠️ OpenAI package not available. Using TF-IDF only.")
            openai_api_key = None

        if st.button("Recompute"):
            pass

    # Add dynamic status message based on API key availability
    if openai_api_key and OPENAI_AVAILABLE:
        st.success("""
        🚀 **Enhanced Mode Active**: Using **OpenAI embeddings** for superior semantic understanding and relevance scoring.
        """)
    else:
        st.info("""
        🔍 **Standard Mode**: Currently using **TF-IDF embeddings** for relevance scoring (fast and free).
        
        💡 **Upgrade Tip**: For improved semantic understanding and more accurate relevance scoring, add your OpenAI API key using the sidebar controls and click "Recompute". OpenAI embeddings provide significantly better results for understanding context and meaning.
        """)
    st.markdown("---")

    df = rank_posts(posts, query, w_rel, w_help, w_trust, openai_api_key)

    st.subheader("Ranked Feed")
    st.dataframe(df[["title", "score", "relevance", "helpfulness", "trustworthiness"]], width="stretch")

    st.subheader("Top Items")
    for i, row in df.head(5).iterrows():
        with st.expander(f"#{i+1}: {row['title']} — score {row['score']:.3f}"):
            st.markdown(row["text"])

    st.markdown("---")
    st.caption("Demo only — simple heuristics for helpfulness/trust. Extend with human labels, click-through data, or pairwise preference learning.")
    
    # Attribution
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px; padding: 20px 0;'>
        <strong>Created by:</strong> Aiden Johnson for demonstration purposes<br>
        <strong>Date:</strong> August 29, 2025<br>
        <strong>Contact:</strong> <a href='mailto:aiden.dataminer@gmail.com'>aiden.dataminer@gmail.com</a>
    </div>
    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()
