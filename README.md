# RAGScope 🔍

**Compare, benchmark, and understand different Retrieval-Augmented Generation (RAG) workflows in one place.**

RAGScope is an experimental benchmarking tool built to help developers, researchers, and practitioners evaluate different RAG strategies side by side. Instead of reading vague blog posts or scattered benchmarks, you can **run the workflows, inspect the pipelines, and compare results** on a common dataset.

---

## 🚩 What’s Inside

* **Implemented Workflows**

  * 🟢 **CRAG** (Contextual RAG)
  * 🔵 **Self-RAG** (model-guided retrieval)
  * 🟣 **Adaptive-RAG** (dynamic retrieval strategy)

* **Key Features**

  * Common Q&A dataset for *apples-to-apples* comparison
  * **LangGraph** workflows → modular & extensible
  * **Pinecone integration** for scalable vector search
  * **Reusable Pinecone Handler class** for clean code structure
  * **Streamlit UI** with caching for fast iteration

---

## 📊 Why RAGScope?

Most RAG repos:

* ❌ Focus on just one method
* ❌ Lack reproducible benchmarks
* ❌ Don’t explain workflow trade-offs

RAGScope:

* ✅ Runs **multiple RAG workflows under the same conditions**
* ✅ Lets you inspect pipelines visually in LangGraph
* ✅ Provides a playground to extend and test new strategies

---

## 🛠️ Tech Stack

* **Python 3.10+**
* [LangGraph](https://python.langchain.com/docs/langgraph) – graph-based workflow orchestration
* [Pinecone](https://www.pinecone.io/) – vector DB
* [Streamlit](https://streamlit.io/) – frontend for experiments
* [dotenv](https://pypi.org/project/python-dotenv/) – API key management

---

## ⚡ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/RAGScope.git
cd RAGScope
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Create a `.env` file:

```
PINECONE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run src/app.py
```

---

## 📂 Repo Structure

```
RAGScope/
│── src/
│   ├── rag_workflows/
│   │   ├── crag.py
│   │   ├── selfrag.py
│   │   ├── adaptive_rag.py
│   │
│   ├── pinecone_handler.py   # Reusable Pinecone integration
│   ├── app.py                # Streamlit frontend
│
│── assets/                   # Demo screenshots
│── requirements.txt
│── README.md
```

---

## 📸 Demo

* ![UI Screenshot](./assets/demo1.png)
* ![Workflow Graph](./assets/demo2.png)
* ![Comparison Results](./assets/demo3.png)


## 🤝 Contributing

RAGScope is designed to be **extended**. You can:

* Add new RAG workflows in `src/rag_workflows/`
* Improve benchmark reporting
* Extend the UI

PRs and issues are welcome!

---

## 🔗 Links

* 📖 [LinkedIn Project Post](https://www.linkedin.com/posts/your-link-here)
* 📸 [Demo Screenshots](./assets/)
* 🧑‍💻 [Project Repo](https://lnkd.in/dMmgHjwA)

---

## ⭐ Acknowledgements

* [LangGraph](https://python.langchain.com/docs/langgraph) for workflow design
* [Pinecone](https://www.pinecone.io/) for powering fast retrieval
* [Streamlit](https://streamlit.io/) for rapid prototyping

---

### 📌 Next Steps

* Add more RAG workflows (Hybrid-RAG, Graph-RAG, etc.)
* Add support for multiple vector DBs (Weaviate, FAISS, Milvus)

---

**If this repo saves you time, don’t forget to ⭐ it!**
