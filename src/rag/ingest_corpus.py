"""Chunk + embed the RAG corpus (data/corpus/{regulations,steward_decisions,
race_summaries}/) into a local Chroma vector store. Embeddings run locally
(sentence-transformers) -- no external API calls anywhere in this pipeline."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"
PERSIST_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
COLLECTION_NAME = "f1_explainer_corpus"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DOC_TYPES = {
    "regulations": "regulation",
    "steward_decisions": "steward_decision",
    "race_summaries": "circuit_summary",
}


def load_corpus() -> list[Document]:
    """One Document per source file (PDF pages joined, txt read whole) --
    chunking happens separately so page boundaries don't create artificial
    chunk breaks mid-sentence."""
    docs = []
    for subdir, doc_type in DOC_TYPES.items():
        folder = CORPUS_DIR / subdir
        for path in sorted(folder.glob("*")):
            if path.suffix == ".pdf":
                text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            elif path.suffix == ".txt":
                text = path.read_text(encoding="utf-8")
            else:
                continue
            if not text.strip():
                continue
            docs.append(Document(page_content=text, metadata={"source": path.name, "doc_type": doc_type}))
    return docs


def build_vector_store() -> Chroma:
    docs = load_corpus()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = Chroma.from_documents(
        chunks, embeddings, collection_name=COLLECTION_NAME, persist_directory=str(PERSIST_DIR),
    )
    print(f"Indexed {len(docs)} source documents -> {len(chunks)} chunks -> {PERSIST_DIR}")
    return store


def load_vector_store() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(collection_name=COLLECTION_NAME, embedding_function=embeddings, persist_directory=str(PERSIST_DIR))


def format_retrieved_context(docs: list[Document]) -> str:
    """One shared formatting of retrieved chunks for the LLM prompt -- used by
    every caller of load_vector_store().similarity_search() (explain.py,
    build_finetune_dataset.py, chat.py) so the context shape they show the
    model can't quietly diverge between call sites."""
    return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)


if __name__ == "__main__":
    build_vector_store()
