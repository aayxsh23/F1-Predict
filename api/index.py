"""Vercel entry point: the FastAPI app as one Python serverless function, which
vercel.json routes /api/* to. Requests may arrive with or without the /api
prefix depending on the rewrite, so strip it before the app routes them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.main import app as api  # noqa: E402


async def app(scope, receive, send):
    if scope["type"] == "http" and scope["path"].startswith("/api/"):
        scope = {**scope, "path": scope["path"][4:]}
    await api(scope, receive, send)
