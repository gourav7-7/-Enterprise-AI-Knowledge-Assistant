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

    def ingestDirectory(self, dir_path: str | Path) -> list[dict]:
        directory = Path(dir_path)
        if not directory.is_dir():
            raise DocumentNotFoundError(f"Not a directory; {directory}")

        pdfs = sorted(directory.glob("*.pdf"))
        if not pdfs:
            raise DocumentNotFoundError(f"No PDF files found in {directory}")

        logger.info("Ingesting %d PDF(s) from %s", len(pdfs), directory)
        return [self.ingest(pdf) for pdf in pdfs]


    def reset(self) -> None:

        logger.warning("Resetting collection '%s'",self._settings.collection_name)
        try:
            self._store.delete_collection()
        except Exception as e:
            logger.debug("delete_collection ignored: %s", e)
        self._store = self._build_store()


def _cli() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description = "INgest a PDF file or a directory of PDFs into the vectore store.")
    parser.add_argument("path", help= "Path to a .pdf file or a directory of PDFs")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the collection before ingesting (avoids duplicates)"
    )
    args = parser.parse_args()

    ingestor = DocIngestor()
    if args.reset:
        ingestor.reset()

    target = Path(args.path)
    if target.is_dir():
        res = ingestor.ingestDirectory(target)
        total = sum(r["chunks"] for r in res)
        print(f"Ingested {len(res)} file(s), {total} chunks total: ")
        for r in res:
            print(f" -{r['file']}: {r['pages']} pages -> {r['chunks']} chunks")
    else:
        r = ingestor.ingest(target)
        print(f"Ingested {r['file']}: {r['pages']} pages -> {r['chunks']} chunks")

if __name__ == "__main__":
    _cli()