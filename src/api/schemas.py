"""Request bodies for the two POST endpoints -- the only user-submitted
structured input in this API, so the only place a pydantic model earns its
keep. Every GET's parameters are plain path/query args typed directly on the
route function; every response is a pass-through dict already shaped by
refresh_job.py/backtest_export.py/explain.py, so mirroring those shapes again
here would just be a second copy of the same schema to keep in sync."""
from pydantic import BaseModel


class ExplainRequest(BaseModel):
    season: int
    round: int
    driver: str
    target: str = "finish_position"


class AskAgentRequest(BaseModel):
    message: str
    conversation_id: str | None = None
