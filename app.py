import os
import hashlib
from pathlib import Path
from typing import Annotated, TypedDict, List

import streamlit as st
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.retrievers import ArxivRetriever

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
SERPER_KEY = os.getenv("SERPER_API_KEY")

if not GROQ_KEY:
    st.error("❌ GROQ_API_KEY missing in environment.")
if not SERPER_KEY:
    st.error("❌ SERPER_API_KEY missing in environment.")


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Agentic RAG (PDF)", page_icon="📄", layout="wide")
st.title("📄 Agentic RAG Chat")
st.caption("Upload a PDF. The agent will decide between PDF RAG, web search, or arXiv search.")


# -----------------------------
# Helpers
# -----------------------------
def _file_sha256(b: bytes):
    return hashlib.sha256(b).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_vectordb(pdf_path, persist_dir):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
        raise ValueError("PDF contains no extractable text.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("Text splitter produced 0 chunks — reduce chunk size.")

    vectordb = Chroma.from_documents(splits, embedder(), persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


def format_hits(docs):
    out = ["PDF RAG Results:"]
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.replace("\n", " ")[:400]
        out.append(f"{i}. (page {d.metadata.get('page')}) {snippet}")
    return "\n".join(out)


# -----------------------------
# PDF Upload
# -----------------------------
uploaded = st.file_uploader("Upload PDF", type=["pdf"])

retriever = None
pdf_meta = None

if uploaded:
    file_bytes = uploaded.getvalue()
    file_id = _file_sha256(file_bytes)

    workspace = Path(".rag_workspace")
    workspace.mkdir(exist_ok=True)
    pdf_path = workspace / f"{file_id}.pdf"

    if not pdf_path.exists():
        pdf_path.write_bytes(file_bytes)

    with st.spinner("Indexing PDF…"):
        vectordb = load_vectordb(str(pdf_path), str(workspace / f"chroma_{file_id}"))
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    pdf_meta = uploaded.name
    st.success(f"Indexed PDF: {pdf_meta}")


# -----------------------------
# Tools
# -----------------------------
tools = []


if retriever:
    @tool
    def search_pdf(query: str) -> str:
        """Search the uploaded PDF and return relevant chunks."""
        hits = retriever.invoke(query)
        return format_hits(hits)

    tools.append(search_pdf)


# ---- Serper Web Search ----
os.environ["SERPER_API_KEY"] = SERPER_KEY
serper = GoogleSerperAPIWrapper()

@tool
def search_web(query: str):
    """Search the web using Serper."""
    results = serper.results(query)
    organic = results.get("organic", [])
    out = ["🌍 Web Search Results:"]
    for r in organic[:5]:
        out.append(f"- {r.get('title')}: {r.get('snippet')} ({r.get('link')})")
    return "\n".join(out)

tools.append(search_web)


# ---- ArXiv Search ----
arxiv = ArxivRetriever(max_results=3)

@tool
def search_arxiv(query: str):
    """Search arXiv for scientific papers."""
    papers = arxiv.invoke(query)
    out = ["📚 arXiv Results:"]
    for p in papers:
        out.append(
            f"- {p.metadata.get('title')}\n"
            f"  {p.metadata.get('summary','')[:250]}...\n"
            f"  {p.metadata.get('url')}"
        )
    return "\n".join(out)

tools.append(search_arxiv)


# -----------------------------
# Agent Definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def build_agent():
    llm = ChatGroq(
        model="moonshotai/kimi-k2-instruct-0905",
        temperature=0,
        groq_api_key=GROQ_KEY
    ).bind_tools(tools)

    def llm_node(state: AgentState):
        resp = llm.invoke(state["messages"])
        return {"messages": [resp]}

    graph = StateGraph(AgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(tools))

    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()


agent = build_agent() if retriever else None


# -----------------------------
# Chat UI
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

prompt = st.chat_input("Ask anything (PDF, web, science)…")

if prompt and agent:

    user_msg = HumanMessage(prompt)
    st.session_state.chat.append(user_msg)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = agent.invoke({"messages": st.session_state.chat})
            st.session_state.chat = result["messages"]

        # Show last AI message
        last_ai = next(m for m in reversed(st.session_state.chat) if isinstance(m, AIMessage))
        st.write(last_ai.content)

