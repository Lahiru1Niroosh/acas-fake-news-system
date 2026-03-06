# backend/app.py
from fastapi import FastAPI
from backend.routes.verify import router as verify_router
from backend.routes.image_text_similarity import router as image_text_similarity_router

app = FastAPI(title="ACAS Privacy Agent Backend")

app.include_router(verify_router, prefix="/verify")
app.include_router(image_text_similarity_router, prefix="")

@app.get("/")
def root():
    return {"status": "Backend running"}
