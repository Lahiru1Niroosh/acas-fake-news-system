# backend/api/pipeline.py
from fastapi import APIRouter
from datetime import datetime
from backend.db.mongo_client import (
    tweets,
    raw_user_metadata,
    pipeline_logs
)

router = APIRouter()

@router.post("/pipeline/ingest")
def ingest_tweet(payload: dict):
    tweet = payload.get("tweet", {})
    user = payload.get("user", {})

    # 1️⃣ Store tweet
    tweets.insert_one({
        "tweet_id": tweet.get("id"),
        "text": tweet.get("text"),
        "image_url": tweet.get("image_url"),
        "created_at": datetime.utcnow()
    })

    # 2️⃣ Store raw user metadata (audit)
    raw_user_metadata.insert_one({
        "tweet_id": tweet.get("id"),
        "user": user,
        "created_at": datetime.utcnow()
    })

    # 3️⃣ Log pipeline step
    pipeline_logs.insert_one({
        "step": "INGEST",
        "tweet_id": tweet.get("id"),
        "timestamp": datetime.utcnow()
    })

    return {
        "status": "SUCCESS",
        "message": "Tweet ingested successfully"
    }
