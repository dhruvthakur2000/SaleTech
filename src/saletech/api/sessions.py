from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi import FastAPI


router = APIRouter()

def get_session_manager(request: Request):
    return request.app.state.session_manager
@router.post("/session")
async def create_session(request: Request):
    try:
        session_manager = get_session_manager(request)
        session = await session_manager.create_session()
        return {"session_id": session.session_id}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/session/{session_id}")
async def close_session(session_id: str, request: Request):
    session_manager = get_session_manager(request)
    await session_manager.close_session(session_id)
    return {"status": "closed"}