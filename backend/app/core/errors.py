from fastapi import HTTPException, status

from app.core.error_mapping import ExternalServiceError


def to_http_exception(exc: ExternalServiceError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"message": exc.user_message, "debug": exc.detail},
    )
