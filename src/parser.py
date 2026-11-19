from transformers import AutoTokenizer

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


source = "docs/rgcg.pdf"
MAX_TOKENS = 384
EMBED_MODEL_ID = "intfloat/multilingual-e5-small"

tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
    # max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer` for HF case
)

def chunk():
    converter = DocumentConverter()
    doc = converter.convert(source).document

    chunker = HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
    )
    chunk_iter = chunker.chunk(dl_doc=doc)

    for i, chunk in enumerate(chunk_iter):
        print(f"=== {i} ===")
        print(f"chunk.text:\n{f'{chunk.text}'!r}")

        enriched_text = chunker.contextualize(chunk=chunk)
        print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")

        print()

if __name__ == "__main__":
    chunk()