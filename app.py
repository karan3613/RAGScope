import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from datasets import load_dataset
from dotenv import load_dotenv

from Flows.AdaptiveFlow import AdaptiveFlow
from Flows.BasicFlow import BasicFlow
from Flows.CragFlow import CragFlow
from Handlers.PineConeHandler import PineConeHandler
from Flows.SelfFlow import SelfFlow

# ============================================
# Page Config
# ============================================
st.set_page_config(
    page_title="RAG Model Comparison",
    page_icon="🤖",
    layout="wide"
)

load_dotenv()


# ============================================
# Dataset Loader (cached)
# ============================================
@st.cache_data
def load_data():
    dataset = load_dataset("prsdm/Machine-Learning-QA-dataset", split="train", encoding="utf-8")
    return pd.DataFrame(dataset)


# ============================================
# Base RAG Model Class
# ============================================
class BaseRAG:
    def __init__(self, name):
        self.name = name

    def run(self, query: str) -> dict:
        """Override in subclasses. Must return dict with 'generation' key"""
        raise NotImplementedError




# ============================================
# Load all models into cache
# ============================================
@st.cache_resource
def load_models():
    pinecone_handler = PineConeHandler(index_name= "clgevents")
    self_rag = SelfFlow(pinecone_handler=pinecone_handler)
    return {
        "BasicRAG": BasicFlow(pinecone_handler=pinecone_handler),
        "CRAG": CragFlow(pinecone_handler=pinecone_handler),
        "AdaptiveRAG": AdaptiveFlow(self_rag),
        "SelfRAG": self_rag
    }


# ============================================
# Helper to fetch random Q
# ============================================
def get_random_question(dataset):
    idx = np.random.randint(0, len(dataset))
    return dataset.iloc[idx]["Question"], idx


# ============================================
# Main App
# ============================================
def main():
    st.title("🔍 RAG Model Comparison Tool")
    st.markdown("Compare **Basic RAG, CRAG, Adaptive RAG, and Self RAG** side by side.")

    dataset = load_data()
    models = load_models()

    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.write(f"Total samples: {len(dataset)}")

        if st.button("🎲 Load Random Question"):
            question, idx = get_random_question(dataset)
            st.session_state.random_question = question
            st.session_state.random_idx = idx

    # Input Section
    st.header("📝 Input Section")

    # Show random question if exists
    if "random_question" in st.session_state:
        st.info(f"🎯 Random Question (Index {st.session_state.random_idx}): {st.session_state.random_question}")
        if st.button("Use Random Question"):
            st.session_state.user_input = st.session_state.random_question

    user_query = st.text_area(
        "Enter your query:",
        value=st.session_state.get("user_input", ""),
        height=100,
        placeholder="Type your question here..."
    )

    # Generate button
    if st.button("🚀 Run All Models", type="primary", use_container_width=True):
        if not user_query.strip():
            st.error("⚠️ Please enter a query first!")
        else:
            st.session_state.user_input = user_query
            results = {}

            with st.spinner("Generating responses from all models..."):
                for model_name, model in models.items():
                    print(model_name)
                    results[model_name] = model.run(user_query)["generation"]
                    time.sleep(0.3)  # simulate some delay

            st.session_state.results = results
            st.session_state.query = user_query

    # Display Results
    if "results" in st.session_state:
        st.header("📊 Model Outputs")
        st.markdown(f"**Query:** {st.session_state.query}")

        col1, col2, col3, col4 = st.columns(4)
        columns = [col1, col2, col3, col4]

        for col, (name, response) in zip(columns, st.session_state.results.items()):
            with col:
                st.subheader(f"🤖 {name}")
                st.write(response)
                st.metric("Length", f"{len(response)} chars")

        # Export section
        st.header("📥 Export Results")
        export_data = {
            "query": st.session_state.query,
            "results": st.session_state.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"rag_comparison_{int(time.time())}.json",
            mime="application/json"
        )

        with st.expander("View Raw Data"):
            st.json(export_data)


if __name__ == "__main__":
    main()
