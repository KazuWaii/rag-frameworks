from dotenv import load_dotenv
from functools import cache
from pathlib import Path
from common import CANDIDATES, DATA_DIR, EMBED_MODEL, EMBED_DIM, LLM_MODEL, NO_ACCESS_ANSWER, TOP_K, SYSTEM_PROMPT, load_csv_rows

from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.postprocessor.types import BaseNodePostprocessor

import faiss


QA_TEMPLATE = PromptTemplate(
    SYSTEM_PROMPT + "\n\nContext:\n{context_str}\n\nQuestion: {query_str}\n\nAnswer:"
)

# Global configuration for LlamaIndex
load_dotenv()
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = Groq(model=LLM_MODEL)

# Loading the files
def load_markdown(data_dir):
    markdown_reader = SimpleDirectoryReader(input_dir=data_dir,
                                   recursive=True,
                                   required_exts=['.md'],
                                   file_metadata=lambda path: {
                                       "department": Path(path).parent.name,
                                       "file_name" : Path(path).name
                                        }
                                    )
    return markdown_reader.load_data()

def load_csv(data_dir):
    return [Document(text=text, metadata=metadata) for text, metadata in load_csv_rows(data_dir)]
    
def load_files(data_dir=DATA_DIR):
    return load_markdown(data_dir) + load_csv(data_dir)

# Splitting the documents into chunks
def split_document(documents, chunk_size=512, chunk_overlap=128):
    pipeline = IngestionPipeline(
        transformations=[
            MarkdownNodeParser(), # To split by header
            SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) # To split by sentence
        ]
    )
    return pipeline.run(documents=documents)


# FAISS
def create_faiss_index(nodes, embedding_size=EMBED_DIM):
    # Create a FAISS index
    vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(embedding_size))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index


class DepartmentFilter(BaseNodePostprocessor):
    allowed_departments: list[str]

    def _postprocess_nodes(self, nodes, query_bundle=None):
        return [n for n in nodes if n.metadata["department"] in self.allowed_departments][:TOP_K]
    
# Querying the index
@cache
def get_index():
    return create_faiss_index(split_document(load_files()))

def ask(question, allowed_departments):
    query_engine = get_index().as_query_engine(
        similarity_top_k=CANDIDATES, text_qa_template=QA_TEMPLATE,
        node_postprocessors=[DepartmentFilter(allowed_departments=sorted(allowed_departments))])
    response = query_engine.query(question)
    if not response.source_nodes:
        return {"answer": NO_ACCESS_ANSWER, "contexts": []}
    return {"answer": str(response), "contexts": [n.get_content() for n in response.source_nodes]}

if __name__ == "__main__":
    result = ask("What is Aadhya Patel's salary?", {"engineering"})
    print(result["answer"])
    for context in result["contexts"]:
        print("Context:", context)