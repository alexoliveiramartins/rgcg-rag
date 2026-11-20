from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from src.parser import *

bm25 = SparseTextEmbedding("Qdrant/bm25", language="portuguese")  # stemming em Portugues
modelo = SentenceTransformer("intfloat/multilingual-e5-small")
client = QdrantClient("http://localhost:6333")

COLLECTION = "rgcg"

if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)

client.create_collection(
    COLLECTION,
    vectors_config={"dense": models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

client.create_payload_index(
    COLLECTION, 
    field_name="artigos", 
    field_schema=models.PayloadSchemaType.KEYWORD
)

chunks = chunk(convert(source))

textos = [c["texto_embedding"] for c in chunks]
dense = modelo.encode(["passage: " + t for t in textos], normalize_embeddings=True)
sparse = list(bm25.embed(textos))

client.upsert(COLLECTION, points=[
    models.PointStruct(
        id=i,
        vector={
            "dense": d.tolist(),
            "sparse": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist()),
        },
        payload=c,
    )
    for i, (c, d, s) in enumerate(zip(chunks, dense, sparse))
])