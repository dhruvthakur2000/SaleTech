from fastapi import APIRouter, Request, HTTPException
from saletech.main import session_manager


router = APIRouter()

@router.post("/session")
async def create_session():
    try:
        session = await session_manager.create_session()
        return {"session_id": session.session_id}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/session/{session_id}")
async def close_session(session_id: str):
    await session_manager.close_session(session_id)
    return {"status": "closed"}