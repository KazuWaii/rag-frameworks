from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from common import CANDIDATES, DATA_DIR, EMBED_MODEL, LLM_MODEL, NO_ACCESS_ANSWER, TOP_K, SYSTEM_PROMPT, load_csv_rows

# Configuration
load_dotenv()
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
llm = ChatGroq(model=LLM_MODEL)

# Chargement du Markdown
def load_markdown(data_dir):
    loader = DirectoryLoader(str(data_dir), glob="**/*.md",
                             loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    documents = loader.load()
    for doc in documents:
        path = Path(doc.metadata["source"])
        doc.metadata = {"department": path.parent.name, "file_name": path.name}
    return documents

# Chargement du CSV
def load_csv(data_dir):
    return [Document(page_content=text, metadata=metadata) for text, metadata in load_csv_rows(data_dir)]

def load_files(data_dir=DATA_DIR):
    return load_markdown(data_dir) + load_csv(data_dir)

# Découpage
HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]

def split_document(documents, chunk_size=512, chunk_overlap=128):
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    sections = []
    for doc in documents:
        for section in header_splitter.split_text(doc.page_content):
            section.metadata = {**doc.metadata, **section.metadata}
            sections.append(section)
    return size_splitter.split_documents(sections)

# FAISS
@cache
def get_vectorstore():
    chunks = split_document(load_files())
    print(f"{len(chunks)} chunks")
    return FAISS.from_documents(chunks, embeddings)

# Prompt et ask
PROMPT = ChatPromptTemplate.from_template(
    SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)
chain = PROMPT | llm | StrOutputParser()

def ask(question, allowed_departments):
    docs = get_vectorstore().similarity_search(question, k=TOP_K, fetch_k=CANDIDATES, filter=lambda metadata: metadata["department"] in allowed_departments)
    if not docs:
        return {"answer": NO_ACCESS_ANSWER, "contexts": []}
    
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = chain.invoke({"context": context, "question": question})
    return {"answer": answer, "contexts": [doc.page_content for doc in docs]}

if __name__ == "__main__":
    result = ask("What is Aadhya Patel's salary?")
    print(result["answer"])
    for context in result["contexts"]:
        print("Context:", context)