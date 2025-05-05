from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def read_root():
    """Provides a simple health check / info endpoint."""
    return {"message": "OpenAI Proxy API with Tool & Think Tag Parsing is running"}