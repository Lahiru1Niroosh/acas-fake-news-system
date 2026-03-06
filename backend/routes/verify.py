# backend/routes/verify.py
from fastapi import APIRouter
from ..pipeline.controller import PipelineController

router = APIRouter()
pipeline_controller = PipelineController()

@router.post("/verify")
def verify_tweet(payload: dict):
    print("\n--- VERIFY ENDPOINT HIT ---")
    print("RAW PAYLOAD:", payload)

    result = pipeline_controller.run(payload)

    print("FINAL RESULT:", result)
    print("--- END VERIFY ---\n")

    return result
