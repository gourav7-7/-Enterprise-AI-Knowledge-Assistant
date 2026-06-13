from __future__ import annotations
 
from pathlib import Path
 
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
from app.config import Settings, get_settings
from app.core.exception import DocumentNotFoundError, IngestionError
from app.core.logger import get_logger
 
logger = get_logger(__name__)

class DocIngestor:
    def __init__(self, settings:Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._embeddings = OpenAIEmbeddings(
            model= self._settings.embedding_model,
            api_key = self._settings.openai_api_key
        )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size = self._settings.chunk_size,
            chunk_overlap = self._settings.chunk_overlap,
            add_start_index = True
        )

        self._store = self._build_store()

    def _build_store(self) -> Chroma:
        return Chroma(
            collection_name = self._settings.collection_name,
            embedding_function = self._embeddings,
            persist_directory= self._settings.chroma_dir
        )

    def ingest(self, file_path: str | Path) -> dict:
        path = Path(file_path)
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise IngestionError(f"Unsupported file type '{path.suffix}. Only .pdf is supported.")
        
        logger.info("Loading PDF : %s", path.name)
        pages = PyPDFLoader(str(path)).load()
        if not pages:
            raise IngestionError(f"No extractable text found in {path.name}.")


        chunks = self._splitter.split_documents(pages)
        if not chunks:
            raise IngestionError(f"splitting produced no chunks for {path.name}")

        for chunk in chunks:
            raw_src = chunk.metadata.get("source", path.name)
            chunk.metadata["source"] = Path(raw_src).name

        logger.info("Split into %d chunks; embedding and storing...", len(chunks))
        self._store.add_documents(chunks)
        logger.info("Stored %d chunks in collection '%s", len(chunks),self._settings.collection_name)

        return {"file": path.name, "pages": len(pages), "chunks": len(chunks)}


    def reset(self) -> None:

        logger.warning("Resetting collection '%s'",self._settings.collection_name)
        try:
            self._store.delete_collection()
        except Exception as e:
            logger.debug("delete_collection ignored: %s", e)
        self._store = self._build_store()


