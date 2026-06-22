# -*- coding: utf-8 -*-
"""FastAPI application for the minimal InsureRAG service."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from api.routes import router
from config.settings import PROJECT_ROOT, settings

WEB_INDEX = PROJECT_ROOT / "web" / "index.html"


def create_app() -> FastAPI:
    app = FastAPI(
        title="InsureRAG API",
        description="HTTP service + upload/QA web UI for the InsureRAG pipeline.",
        version="0.1.0",
    )
    app.include_router(router)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        if WEB_INDEX.exists():
            return HTMLResponse(WEB_INDEX.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>InsureRAG API</h1><p>See <a href='/docs'>/docs</a>.</p>")

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=settings.api["host"],
        port=int(settings.api["port"]),
        reload=False,
    )
