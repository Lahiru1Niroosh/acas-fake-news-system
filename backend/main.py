from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Create controller on startup to avoid instantiating LLMs at import time
@app.on_event("startup")
async def startup_event():
    from .pipeline.controller import PipelineController
    try:
        app.state.controller = PipelineController()
    except Exception as e:
        raise RuntimeError("Failed to initialize PipelineController. Ensure OPENAI_API_KEY is set in your environment.") from e

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/verify")
async def verify(payload: dict):
    # This calls the controller, which calls PII, then Routing, then Similarity
    controller = getattr(app.state, "controller", None)
    if not controller:
        raise RuntimeError("Controller not initialized; check startup logs.")
    return controller.run(payload)