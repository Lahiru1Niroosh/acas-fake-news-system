# backend/app.py
from fastapi import FastAPI
from .api.verify import router as verify_router

app = FastAPI(title="ACAS Privacy Agent Backend")

app.include_router(verify_router, prefix="/verify")

@app.get("/")
def root():
    return {"status": "Backend running"}
