from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Check service health")
def health_check() -> dict[str, str]:
    """Return the current health status of the API."""
    return {
        "status": "ok",
        "service": "docmind-ai",
    }
