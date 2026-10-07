"""Web demo: ask the LlamaIndex, LangChain and Haystack versions of the RAG pipeline the same question.

Run locally:
    uv run streamlit run streamlit_app.py
"""
import hmac
import importlib
import os
import time

import streamlit as st
from dotenv import load_dotenv

# Demo account, shown on the login screen on purpose: it only keeps crawlers away.
# What actually protects the Groq quota is the per-session limit on LLM calls.
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "finsolve"
MAX_LLM_CALLS_PER_SESSION = 30

# Framework name -> (module, function that builds its index)
FRAMEWORKS = {
    "LlamaIndex": ("llamaindex_rag", "get_query_engine"),
    "LangChain": ("langchain_rag", "get_vectorstore"),
    "Haystack": ("haystack_rag", "get_pipeline"),
}
COMPARE = "Compare all three"

# (category, question, why it is interesting)
SUGGESTIONS = [
    ("HR records", "What is Aadhya Patel's salary?",
     "One row among 100 similar employee rows."),
    ("Ambiguous name", "What is Ishaan Singh's leave balance?",
     "Three employees share this name: a good answer lists all three."),
    ("Finance", "What was FinSolve's revenue in Q4 2024?", ""),
    ("Lost context", "What was the cash flow from operations in Q1 2024?",
     "The quarter only appears in a parent header, which LangChain and Haystack do not pass to the LLM."),
    ("Marketing", "Which European countries did FinNova target in Q1 2024?", ""),
    ("Engineering", "What encryption is used for data at rest?", ""),
    ("Company policy", "When is a reimbursement paid after the claim is approved?", ""),
    ("Not in the documents", "Who is the CEO of FinSolve?",
     "The right answer is to say the documents don't contain it."),
]

HOW_IT_WORKS = """
- Same data, embedding model (`all-MiniLM-L6-v2`), FAISS exact index, top 4 chunks, prompt and LLM
  (Groq `gpt-oss-20b`) in all three versions: only the framework changes.
- The first question sent to a framework also builds its index, which takes about a minute.
- Under each answer, *Retrieved chunks* shows the 4 chunks the framework gave the LLM.
"""

FINDINGS = """
- With the same settings, LangChain and Haystack retrieve the same chunks.
- LlamaIndex embeds metadata with the text and keeps header-only chunks. It knows which quarter a
  figure belongs to, but has more trouble finding one employee among 100 similar rows.
- On 20 generated test questions graded by an LLM judge: LlamaIndex 17/20 correct,
  LangChain 20/20, Haystack 20/20. Twenty questions is a small sample.
"""


def load_groq_key():
    load_dotenv()  # local run: key in .env
    if "GROQ_API_KEY" not in os.environ and "GROQ_API_KEY" in st.secrets:  # Streamlit Community Cloud
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "GROQ_API_KEY" not in os.environ:
        st.error("GROQ_API_KEY is missing: add it to `.env` (local) or to the app's secrets (Streamlit Cloud).")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_framework(name):
    # Imported lazily so the login page shows up without waiting for three embedding models
    module_name, build_name = FRAMEWORKS[name]
    module = importlib.import_module(module_name)
    getattr(module, build_name)()  # builds the index once per server process
    return module.ask


def run(name, question):
    with st.spinner(f"{name} is answering (its first question also builds the index)..."):
        try:
            ask = load_framework(name)
            start = time.perf_counter()
            result = ask(question)
            result["latency_s"] = time.perf_counter() - start
        except Exception as error:  # show any framework or Groq error (e.g. quota) instead of crashing the page
            result = {"error": str(error)}
    return result


def login_page():
    st.title("RAG frameworks comparison")
    st.write("The same RAG pipeline over FinSolve's internal documents (a fictional company), "
             "built three times with LlamaIndex, LangChain and Haystack.")
    st.info(f"Demo account: username `{DEMO_USERNAME}`, password `{DEMO_PASSWORD}`")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Log in"):
            if (hmac.compare_digest(username.encode(), DEMO_USERNAME.encode())
                    and hmac.compare_digest(password.encode(), DEMO_PASSWORD.encode())):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Wrong username or password.")


def show_result(name, result):
    if "error" in result:
        st.markdown(f"**{name}**")
        st.error(result["error"])
        return
    st.markdown(f"**{name}** · {result['latency_s']:.1f} s")
    st.markdown(result["answer"].replace("$", "\\$"))  # "$" would otherwise start a LaTeX formula
    with st.expander("Retrieved chunks"):
        for i, context in enumerate(result["contexts"], 1):
            st.caption(f"Chunk {i}")
            st.text(context)


def chat_page():
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("llm_calls", 0)

    with st.sidebar:
        mode = st.radio("Framework", [*FRAMEWORKS, COMPARE], index=len(FRAMEWORKS))
        calls_left = st.empty()  # filled at the end, once this run's question has been counted
        with st.expander("How it works"):
            st.markdown(HOW_IT_WORKS)
        with st.expander("What the evaluation found"):
            st.markdown(FINDINGS)
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    st.title("Ask FinSolve's documents")
    st.caption("Employee records, finance and marketing reports, engineering documentation and the employee handbook.")

    question = None
    with st.expander("Suggested questions", expanded=not st.session_state["history"]):
        columns = st.columns(2)
        for i, (category, suggestion, why) in enumerate(SUGGESTIONS):
            with columns[i % 2]:
                st.caption(category)
                if st.button(suggestion, help=why or None, width="stretch"):
                    question = suggestion
    question = st.chat_input("Ask a question about FinSolve's documents") or question

    if question:
        names = list(FRAMEWORKS) if mode == COMPARE else [mode]
        if st.session_state["llm_calls"] + len(names) > MAX_LLM_CALLS_PER_SESSION:
            st.warning(f"This session has used its {MAX_LLM_CALLS_PER_SESSION} LLM calls.")
        else:
            st.session_state["llm_calls"] += len(names)
            results = {name: run(name, question) for name in names}
            st.session_state["history"].append({"question": question, "results": results})

    for entry in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            columns = st.columns(len(entry["results"]))
            for column, (name, result) in zip(columns, entry["results"].items()):
                with column:
                    show_result(name, result)

    remaining = MAX_LLM_CALLS_PER_SESSION - st.session_state["llm_calls"]
    calls_left.caption(f"LLM calls left in this session: {remaining} (comparing uses 3)")


st.set_page_config(page_title="RAG frameworks comparison", layout="wide")
load_groq_key()
if st.session_state.get("authenticated"):
    chat_page()
else:
    login_page()
