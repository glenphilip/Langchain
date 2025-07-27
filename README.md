# LangChain Hands-On Projects

This repository contains practical implementations from a LangChain course, focusing on building end-to-end LLM applications using LangChain. Projects demonstrate integration with various tools and services for real-world use cases.

---

## What's Included

### Chatbot Systems
- LLMs via **Groq** (e.g., `llama3-8b-8192`, `llama3-70b-8192`) and **Ollama** (e.g., `gemma2:2b`, `llama2`)
- Streaming responses and robust API error handling

### RAG Pipelines
- Load data from **PDFs**, **websites**, and other unstructured sources
- Use of **vector databases**: ChromaDB, FAISS, LanceDB
- Embedding models: `all-MiniLM-L6-v2`, `bge-small-en-v1.5`
- Examples: `create_stuff_documents_chain`, `create_retrieval_chain`
- Projects include YouTube video assistants and PDF/document-based RAG

### Agent Implementations
- Tools: Wikipedia, Arxiv, and custom retrievers (e.g., LangSmith docs)
- Built using `initialize_agent` with `CHAT_CONVERSATIONAL_REACT_DESCRIPTION`

### FastAPI Integration
- Deploy LangChain chains as API endpoints using **FastAPI**, **LangServe**, and **Uvicorn**

---

## Tech Stack

- **LangChain**: Core framework
- **Groq API / Ollama**: LLM inference
- **Vector Stores**: ChromaDB, FAISS, LanceDB
- **Data Loaders**: PyPDFLoader, WebBaseLoader, TextLoader, etc.
- **Embeddings**: HuggingFace Sentence Transformers
- **Frontend**: Streamlit
- **Backend**: FastAPI + LangServe
- **Utils**: `python-dotenv`, `youtube-transcript-api`, `bs4`, `arxiv`, `wikipedia`

