from transformers import AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

source = "docs/rgcg.pdf"
MAX_TOKENS = 384
EMBED_MODEL_ID = "intfloat/multilingual-e5-small"
tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

opts = PdfPipelineOptions(do_formula_enrichment=True)

def convert(src: str) -> str:
    texto = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    ).convert(src).document.export_to_markdown()

    # O Docling identifica os titulos certo, porém não preserva a hierarquia
    # esse trecho retira os identificadores md de headings e coloca manualmente
    texto = (
        texto.replace("## ", "")
        .replace("\n- ", "\n")
        .replace("<!-- image -->", "")
        .replace("\nTÍTULO", "\n# TÍTULO")
        .replace("\nCapítulo", "\n## Capítulo")
        .replace("\nSeção", "\n### Seção")
    )

    # Junta o nome do header (Ex: Da Estrutura Curricular dos Cursos de Graduação) com
    # o header de fato (Seção II), usando o identificador md (#, ##, ###);
    linhas = []
    for linha in texto.splitlines():
        if not linha.strip():
            continue
        anterior = linhas[-1] if linhas else ""
        if anterior.startswith("#") and len(anterior.split()) <= 3 and not linha.startswith("#"):
            linhas[-1] += " - " + linha
        else:
            linhas.append(linha)

    return "\n\n".join(linhas)

    
def chunk(txt: str) -> list[dict]:
    chunks_md = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "titulo"), ("##", "capitulo"), ("###", "secao")],
    ).split_text(txt)

    recursive_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=330, chunk_overlap=0,
        separators=["\nArt. ", "\n\n", "\n", ". ", " "],
    )
    docs = recursive_splitter.split_documents(chunks_md)  # cada pedaço herda titulo/capitulo/secao

    # Adiciona os metadados (splitters do MdHeaderSplitter -> titulo, capitulo, secao)
    # Em cima dos chunks para contextualizar, e adiciona "continuacao do {artigo}"
    # em artigos quebrados no meio pelo chunking
    # A saída é no formato para ser indexado pelo Qdrant
    resultado, artigo = [], ""
    for d in docs:
        cabecalho = list(d.metadata.values())

        continuacao = None
        if not d.page_content.startswith("Art.") and artigo:
            continuacao = artigo
            cabecalho.append(f"(continuação do {artigo})")

        artigos = []  # todos os artigos que aparecem neste chunk
        for linha in d.page_content.splitlines():
            if linha.startswith("Art."):
                artigo = " ".join(linha.split()[:2]).rstrip(".")
                artigos.append(artigo)

        texto = d.page_content
        resultado.append({
            "texto_embedding": "\n".join(cabecalho) + "\n\n" + texto if cabecalho else texto,
            "texto": texto,
            "titulo": d.metadata.get("titulo"),
            "capitulo": d.metadata.get("capitulo"),
            "secao": d.metadata.get("secao"),
            "artigos": artigos + ([continuacao] if continuacao else []),
        })

    return resultado


if __name__ == "__main__":
    doc_text = convert(source)
    chunks = chunk(doc_text)
    for i, c in enumerate(chunks):
        print(f"===== CHUNK {i} =====")
        print(f"artigos: {', '.join(c['artigos'])}")
        print()
        print(c["texto_embedding"])
        print()