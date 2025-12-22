from fastapi import APIRouter, Request, HTTPException


router = APIRouter()

@router.post("/session")
async def create_session(request: Request):
    try:
        manager = request.app.state.session_manager
        session = await manager.create_session()
        return {"session_id": session.session_id}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.delete("/session/{session_id}")
async def close_session(session_id: str, request: Request):
    manager = request.app.state.session_manager
    await manager.close_session(session_id)
    return {"status": "closed"}