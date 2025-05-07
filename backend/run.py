import uvicorn
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        workers=settings.WORKERS,
        backlog=settings.BACKLOG,
        timeout_keep_alive=settings.TIMEOUT,
        limit_concurrency=settings.LIMIT_CONCURRENCY,
        log_level=settings.LOG_LEVEL.lower()
    ) 