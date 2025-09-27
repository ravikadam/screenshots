import os
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec


def get_pinecone_client(api_key: Optional[str] = None) -> Pinecone:
    api_key = api_key or os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY. Set it in environment or .env file.")
    return Pinecone(api_key=api_key)


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str = "cosine",
    cloud: Optional[str] = None,
    region: Optional[str] = None,
):
    cloud = cloud or os.getenv("PINECONE_ENV_CLOUD", "aws")
    region = region or os.getenv("PINECONE_ENV_REGION", "us-east-1")

    # Pinecone client v5 returns a list-like of index models, not a dict
    indexes = pc.list_indexes()
    existing = set()
    for idx in indexes:
        # Handle both object and dict styles
        name = getattr(idx, "name", None)
        if name is None and isinstance(idx, dict):
            name = idx.get("name")
        if name is not None:
            existing.add(name)

    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
    return pc.Index(index_name)


def upsert_vectors(
    index,
    ids: List[str],
    vectors,  # List[List[float]] or numpy
    metadatas: Optional[List[Dict[str, Any]]] = None,
    namespace: Optional[str] = None,
):
    if metadatas is None:
        metadatas = [{} for _ in ids]
    # Convert vectors to python lists
    payload = []
    for _id, vec, md in zip(ids, vectors, metadatas):
        values = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        payload.append({"id": _id, "values": values, "metadata": md})
    return index.upsert(vectors=payload, namespace=namespace)


essential_metadata_keys = ["image_path"]


def query_vectors(
    index,
    vector,  # List[float]
    top_k: int = 5,
    namespace: Optional[str] = None,
    include_metadata: bool = True,
):
    return index.query(
        vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
        top_k=top_k,
        namespace=namespace,
        include_metadata=include_metadata,
    )


def describe_stats(index, namespace: Optional[str] = None):
    try:
        return index.describe_index_stats(filter=None, namespace=namespace)
    except Exception as e:
        return {"error": str(e)}
