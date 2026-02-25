from fastapi import Request
from fastapi.responses import JSONResponse

from src.saletech.utils.errors import SaleTechException
from src.saletech.utils.logger import get_logger

logger = get_logger("saletech.api")


async def saletech_exception_handler(
    request: Request,
    exc: SaleTechException,
):
    logger.error(
        "exception_caught",
        path=str(request.url),
        method=request.method,
        error_code=exc.error_code,
        message=exc.message,
        context=exc.context,
        status_code=exc.status_code,
        exc_info=True,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )