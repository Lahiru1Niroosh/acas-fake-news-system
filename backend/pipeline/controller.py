from ..agents.pii_agent import PIIAgent
from ..agents.text_agent import TextAgent
from ..agents.image_agent import ImageAgent
from ..agents.similarity_agent import SimilarityAgent
from ..db.mongo_client import analysis_logs
from .crew_manager import CrewManager

class PipelineController:
    def __init__(self):
        self.pii = PIIAgent()
        self.text_agent = TextAgent()
        self.image_agent = ImageAgent()
        self.similarity_agent = SimilarityAgent()

    def run(self, payload):
        user = payload.get("user")
        tweet = payload.get("tweet")

        print("==================================================")
        print("🚀 NEW VERIFICATION REQUEST RECEIVED")
        print("==================================================")

        # 1. MASKING STEP
        masked_user = self.pii.mask(user)

        # 2. CONTENT ANALYSIS & ROUTING
        text_out = None
        image_out = None

        # Check for Text
        if tweet.get("text"):
            print(f"📝 [ROUTING] Text detected. Sending to TextAgent...")
            text_out = self.text_agent.analyze(tweet["text"])
            print(f"✅ [TEXT AGENT] Result: {text_out['label']} (Score: {text_out['score']})")
        else:
            print(f"➖ [ROUTING] No text found in tweet.")

        # Check for Image
        if tweet.get("image_url"):
            print(f"🖼️ [ROUTING] Image detected. Sending to ImageAgent...")
            image_out = self.image_agent.analyze(tweet["image_url"])
            print(f"✅ [IMAGE AGENT] Result: {image_out['label']} (Score: {image_out['score']})")
        else:
            print(f"➖ [ROUTING] No image found in tweet.")

        # 3. FINAL GATEWAY (SIMILARITY AGENT)
        print(f"🧠 [FINAL GATE] Passing results to Similarity Agent for XAI explanation...")
        final_output = self.similarity_agent.finalize(text_out, image_out)
        
        print(f"🏁 [DECISION] Verdict: {final_output['verdict']} | Reason: {final_output['reason']}")

        # 4. DB STORAGE
        analysis_logs.insert_one({
            "masked_id": masked_user["masked_id"],
            "agent_outputs": {"text": text_out, "image": image_out},
            "final_decision": final_output
        })

        print("==================================================\n")
        
        return {
            "masked_user_id": masked_user["masked_id"],
            "credibility": final_output
        }