from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from qdrant_client.http.exceptions import UnexpectedResponse

from src.app.settings import settings
from src.storage.qdrant_store import get_qdrant_client


@dataclass(frozen=True)
class QdrantPreflightResult:
    url: str
    collection: str
    status: str
    points_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "collection": self.collection,
            "status": self.status,
            "points_count": self.points_count,
        }


def _status_text(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "")


def check_qdrant_ready(*, require_points: bool = True) -> QdrantPreflightResult:
    """Fail early when evaluation points at the wrong or empty Qdrant collection."""
    client = get_qdrant_client()
    collection = str(getattr(settings, "qdrant_collection", "") or "").strip()
    url = str(getattr(settings, "qdrant_url", "") or "").strip()
    try:
        info = client.get_collection(collection)
    except UnexpectedResponse as exc:
        raise RuntimeError(
            f"Qdrant preflight failed: collection '{collection}' is not reachable at {url}. "
            "Check QDRANT_URL and QDRANT_COLLECTION before running evaluation."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Qdrant preflight failed for collection '{collection}' at {url}: {exc}"
        ) from exc

    points_count = int(getattr(info, "points_count", 0) or 0)
    result = QdrantPreflightResult(
        url=url,
        collection=collection,
        status=_status_text(getattr(info, "status", "")),
        points_count=points_count,
    )
    if require_points and points_count <= 0:
        raise RuntimeError(
            f"Qdrant preflight failed: collection '{collection}' at {url} has no points. "
            "Re-ingest local data or choose the correct collection."
        )
    return result
