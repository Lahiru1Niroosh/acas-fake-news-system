import uuid
from datetime import datetime
from ..db.mongo_client import pii_audit

class PIIAgent:
    def mask(self, user_data: dict):
        print(f"\n--- [PII AGENT] STARTING MASKING ---")
        print(f"RAW DATA RECEIVED: {user_data.get('name')} (@{user_data.get('username')})")
        
        masked_id = f"ANON_{uuid.uuid4().hex[:8]}"
        
        # Audit Log (Internal DB)
        pii_audit.insert_one({
            "timestamp": datetime.utcnow(),
            "masked_id": masked_id,
            "original_data": user_data
        })
        
        print(f"SUCCESS: Identity hidden. New ID: {masked_id}")
        print(f"--- [PII AGENT] COMPLETED ---\n")
        
        return {
            "masked_id": masked_id,
            "is_verified": user_data.get("verified", False)
        }