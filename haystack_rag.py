from functools import cache

from dotenv import load_dotenv
from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import TextFileToDocument
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import MarkdownHeaderSplitter, RecursiveDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder)
from haystack_integrations.components.retrievers.faiss import FAISSEmbeddingRetriever
from haystack_integrations.document_stores.faiss import FAISSDocumentStore

from common import DATA_DIR, EMBED_MODEL, EMBED_DIM, LLM_MODEL, NO_ACCESS_ANSWER, TOP_K, SYSTEM_PROMPT, load_csv_rows

# Configuration
load_dotenv()

# Load
def load_markdown(data_dir):
    paths = sorted(data_dir.rglob("*.md"))
    metas = [{"department": p.parent.name, "file_name": p.name} for p in paths]
    return TextFileToDocument().run(sources=paths, meta=metas)["documents"]

def load_csv(data_dir):
    return [Document(content=text, meta=metadata) for text, metadata in load_csv_rows(data_dir)]

def load_files(data_dir=DATA_DIR):
    return load_markdown(data_dir) + load_csv(data_dir)

# Indexing
def build_document_store(documents, chunk_size=512, chunk_overlap=128):
    store = FAISSDocumentStore(embedding_dim=EMBED_DIM)

    indexing = Pipeline()
    indexing.add_component("header_splitter", MarkdownHeaderSplitter())
    indexing.add_component("size_splitter", RecursiveDocumentSplitter(
        split_unit="token", split_length=chunk_size, split_overlap=chunk_overlap))
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder(model=EMBED_MODEL))
    indexing.add_component("writer", DocumentWriter(document_store=store))

    indexing.connect("header_splitter.documents", "size_splitter.documents")
    indexing.connect("size_splitter.documents", "embedder.documents")
    indexing.connect("embedder.documents", "writer.documents")

    result = indexing.run({"header_splitter": {"documents": documents}})
    print(f"{result['writer']['documents_written']} chunks")
    return store

# Query and ask
PROMPT_TEMPLATE = [ChatMessage.from_user(
    SYSTEM_PROMPT
    + "\n\nContext:\n{% for doc in documents %}{{ doc.content }}\n\n{% endfor %}"
    + "Question: {{ question }}\n\nAnswer:"
)]

@cache
def get_pipelines():
    store = build_document_store(load_files())

    retrieval = Pipeline()
    retrieval.add_component("text_embedder", SentenceTransformersTextEmbedder(model=EMBED_MODEL))
    retrieval.add_component("retriever", FAISSEmbeddingRetriever(document_store=store, top_k=TOP_K))
    retrieval.connect("text_embedder.embedding", "retriever.query_embedding")

    generation = Pipeline()
    generation.add_component("prompt_builder", ChatPromptBuilder(template=PROMPT_TEMPLATE))
    generation.add_component("llm", OpenAIChatGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model=LLM_MODEL))
    generation.connect("prompt_builder.prompt", "llm.messages")
    return retrieval, generation

def ask(question, allowed_departments):
    retrieval, generation = get_pipelines()
    filters = {"field": "meta.department", "operator": "in", "value": sorted(allowed_departments)}
    documents = retrieval.run({"text_embedder": {"text": question},
                               "retriever": {"filters": filters}})["retriever"]["documents"]
    if not documents:
        return {"answer": NO_ACCESS_ANSWER, "contexts": []}
    result = generation.run({"prompt_builder": {"documents": documents, "question": question}})
    return {"answer": result["llm"]["replies"][0].text, "contexts": [d.content for d in documents]}

if __name__ == "__main__":
    result = ask("What is Aadhya Patel's salary?", {"engineering"})
    print(result["answer"])
    for context in result["contexts"]:
        print("Context:", context)