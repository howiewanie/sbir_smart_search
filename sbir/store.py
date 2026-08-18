"""Qdrant access layer.

Defaults to Qdrant's embedded mode so a fresh clone needs no server and no
Docker. Set ``SBIR_QDRANT_URL`` to talk to a real deployment instead.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from qdrant_client import QdrantClient, models

from . import config

# Fields worth an index: everything the UI can filter on.
PAYLOAD_INDEXES = {
    "agency": models.PayloadSchemaType.KEYWORD,
    "branch": models.PayloadSchemaType.KEYWORD,
    "phase": models.PayloadSchemaType.KEYWORD,
    "program": models.PayloadSchemaType.KEYWORD,
    "state": models.PayloadSchemaType.KEYWORD,
    "company_key": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "amount": models.PayloadSchemaType.FLOAT,
}


def connect() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    config.QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(config.QDRANT_PATH))


def describe_backend() -> str:
    return config.QDRANT_URL or f"embedded ({config.QDRANT_PATH})"


def recreate_collection(client: QdrantClient, dimension: int) -> None:
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        ),
    )
    if not config.QDRANT_URL:
        # Embedded Qdrant filters in Python, so payload indexes are a no-op
        # there and only produce warnings.
        return

    for field, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(
            collection_name=config.COLLECTION_NAME,
            field_name=field,
            field_schema=schema,
        )


def upsert(client: QdrantClient, points: Iterable[models.PointStruct]) -> None:
    client.upsert(collection_name=config.COLLECTION_NAME, points=list(points))


def count(client: QdrantClient) -> int:
    if not client.collection_exists(config.COLLECTION_NAME):
        return 0
    return client.count(config.COLLECTION_NAME, exact=True).count


def query(client: QdrantClient, vector: list[float], limit: int,
          query_filter: models.Filter | None = None) -> list[models.ScoredPoint]:
    return client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points


def scroll(client: QdrantClient, limit: int,
           query_filter: models.Filter | None = None) -> list[models.Record]:
    records, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return records


def retrieve(client: QdrantClient, ids: list[int]) -> list[models.Record]:
    if not ids:
        return []
    return client.retrieve(
        collection_name=config.COLLECTION_NAME, ids=ids, with_payload=True
    )


def save_companies(mapping: dict[str, list[int]]) -> None:
    config.COMPANIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.COMPANIES_PATH.write_text(json.dumps(mapping))


def load_companies() -> dict[str, list[int]]:
    if not config.COMPANIES_PATH.exists():
        return {}
    try:
        return json.loads(config.COMPANIES_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_meta(meta: dict[str, Any]) -> None:
    config.INDEX_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INDEX_META_PATH.write_text(json.dumps(meta, indent=2))


def load_meta() -> dict[str, Any] | None:
    if not config.INDEX_META_PATH.exists():
        return None
    try:
        return json.loads(config.INDEX_META_PATH.read_text())
    except json.JSONDecodeError:
        return None
