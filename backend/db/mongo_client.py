# backend/db/mongo_client.py

import os
from pathlib import Path
from pymongo import MongoClient
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# Prefer env file located in backend/config/.env (when running scripts from repo root)
env_path = Path(__file__).resolve().parents[1] / "config" / ".env"
if load_dotenv and env_path.exists():
    # use str() for compatibility with older versions
    load_dotenv(str(env_path))
elif load_dotenv:
    load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "acas_db")

# if not MONGO_URI:
#     raise RuntimeError("MONGO_URI environment variable is not set. Ensure backend/config/.env exists or set the variable in your environment.")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

pii_audit = db.pii_audit      # Real User <-> Masked User link
analysis_logs = db.analysis_logs # Final results