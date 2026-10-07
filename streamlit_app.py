"""Web demo: ask the LlamaIndex, LangChain and Haystack versions of the RAG pipeline the same question.

Run locally:
    uv run streamlit run streamlit_app.py
"""
import hmac
import importlib
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from common import DATA_DIR, ROLE_PERMISSIONS, load_csv_rows

# On Linux, NLTK refuses to read files that have several hard links (CWE-59 check), and uv installs
# packages as hard links, so LlamaIndex can't read the NLTK data bundled in site-packages. With
# NLTK_DATA pointing at a normal folder, LlamaIndex and Haystack download their NLTK data there instead.
NLTK_DATA_DIR = Path.home() / "nltk_data"
NLTK_DATA_DIR.mkdir(exist_ok=True)
os.environ.setdefault("NLTK_DATA", str(NLTK_DATA_DIR))

# Demo account, shown on the login screen on purpose: it only keeps crawlers away.
# What actually protects the Groq quota is the per-session limit on LLM calls.
DEMO_USERNAME = "demo"
DEMO_PASSWORD = "finsolve"
MAX_LLM_CALLS_PER_SESSION = 30

# Framework name -> (module, function that builds its index)
FRAMEWORKS = {
    "LlamaIndex": ("llamaindex_rag", "get_index"),
    "LangChain": ("langchain_rag", "get_vectorstore"),
    "Haystack": ("haystack_rag", "get_pipelines"),
}
COMPARE = "Compare all three"
DEFAULT_ROLE = "admin"

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
- Under each answer, *Retrieved chunks* shows the 4 chunks the framework gave the LLM and the file each
  one comes from. The *Documents* page shows the full files, to check an answer.
- *Role* sets which departments can be searched. Each version filters chunks by department at retrieval
  time (LangChain and Haystack with their built-in filters, LlamaIndex with a node postprocessor), so the
  LLM never sees a chunk the role isn't allowed to read. Try a finance question as `employee`.
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


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner=False)
def document_texts():
    # Whitespace-normalized text of each source file, to trace a retrieved chunk back to its file
    texts = {p.relative_to(DATA_DIR).as_posix(): normalize(p.read_text(encoding="utf-8"))
             for p in DATA_DIR.rglob("*.md")}
    for text, meta in load_csv_rows(DATA_DIR):  # CSV rows are indexed as "column : value" text, not raw lines
        name = f"{meta['department']}/{meta['file_name']}"
        texts[name] = texts.get(name, "") + " " + normalize(text)
    return texts


def source_of(chunk):
    start = normalize(chunk)[:100]
    matches = [name for name, text in document_texts().items() if start in text]
    return matches[0] if len(matches) == 1 else None


@st.cache_resource(show_spinner=False)
def load_framework(name):
    # Imported lazily so the login page shows up without waiting for three embedding models
    module_name, build_name = FRAMEWORKS[name]
    module = importlib.import_module(module_name)
    getattr(module, build_name)()  # builds the index once per server process
    return module.ask


def current_role():
    return st.session_state.get("role", DEFAULT_ROLE)


def role_selector():
    # Shown on every page: the Documents page is filtered by role too
    with st.sidebar:
        roles = list(ROLE_PERMISSIONS)
        role = st.selectbox("Role", roles, index=roles.index(DEFAULT_ROLE), key="role")
        st.caption("Can read: " + ", ".join(sorted(ROLE_PERMISSIONS[role])))
        st.caption("Demo only: in a real application the role comes from the login, never from a menu.")


def run(name, question, allowed_departments):
    with st.spinner(f"{name} is answering (its first question also builds the index)..."):
        try:
            ask = load_framework(name)
            start = time.perf_counter()
            result = ask(question, allowed_departments)
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
            source = source_of(context)
            st.caption(f"Chunk {i} · {source}" if source else f"Chunk {i}")
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
    st.caption("Employee records, finance and marketing reports, engineering documentation and the employee "
               "handbook. Check any answer against the full files on the Documents page.")

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
            role = current_role()
            results = {name: run(name, question, ROLE_PERMISSIONS[role]) for name in names}
            st.session_state["history"].append({"question": question, "role": role, "results": results})

    for entry in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(entry["question"])
            st.caption(f"Asked as {entry['role']}")
        with st.chat_message("assistant"):
            columns = st.columns(len(entry["results"]))
            for column, (name, result) in zip(columns, entry["results"].items()):
                with column:
                    show_result(name, result)

    remaining = MAX_LLM_CALLS_PER_SESSION - st.session_state["llm_calls"]
    calls_left.caption(f"LLM calls left in this session: {remaining} (comparing uses 3)")


def documents_page():
    role = current_role()
    allowed = ROLE_PERMISSIONS[role]
    st.title("Source documents")
    st.caption(f"The files the `{role}` role can read, i.e. what the pipelines search for this role. Under each "
               "answer, every retrieved chunk is labelled with the file it comes from.")
    paths = {p.relative_to(DATA_DIR).as_posix(): p for p in sorted(DATA_DIR.rglob("*.*"))
             if p.parent.name in allowed}
    path = paths[st.selectbox("Document", list(paths))]
    st.download_button("Download", path.read_bytes(), file_name=path.name)
    if path.suffix == ".csv":
        st.caption("Use the search icon at the top right of the table to find an employee.")
        st.dataframe(pd.read_csv(path), hide_index=True)
    else:
        st.markdown(path.read_text(encoding="utf-8").replace("$", "\\$"))


st.set_page_config(page_title="RAG frameworks comparison", layout="wide")
load_groq_key()
if st.session_state.get("authenticated"):
    role_selector()
    st.navigation([
        st.Page(chat_page, title="Ask", url_path="ask", default=True),
        st.Page(documents_page, title="Documents", url_path="documents"),
    ], position="top").run()
else:
    login_page()
