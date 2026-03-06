from ..agents.pii_agent import PIIAgent
from ..agents.text_agent import TextAgent
from ..agents.image_agent import ImageAgent
from ..agents.similarity_agent import SimilarityAgent
from ..db.mongo_client import analysis_logs
from .crew_manager import CrewManager
from .image_text_similarity import ImageTextSimilarityAnalyzer, MultimodalXAI

class PipelineController:
    def __init__(self):
        self.pii = PIIAgent()
        self.text_agent = TextAgent()
        self.image_agent = ImageAgent()
        self.similarity_agent = SimilarityAgent()
        
        # Initialize image-text similarity analyzer and XAI system
        self.image_text_analyzer = ImageTextSimilarityAnalyzer()
        self.xai_system = MultimodalXAI(
            self.image_text_analyzer.model,
            self.image_text_analyzer.preprocess,
            self.image_text_analyzer.device
        )

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
            # Pass text content if available for multimodal analysis
            text_for_image = tweet.get("text", "")
            image_out = self.image_agent.analyze(tweet["image_url"], text_content=text_for_image)
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
    
    def analyze_image_text_similarity(self, image_input, text, from_url=False):
        """Analyze similarity between image and text"""
        similarity, ocr_text = self.image_text_analyzer.multimodal_similarity(
            image_input, text, from_url=from_url
        )
        decision = self.image_text_analyzer.detect_context(similarity)
        
        return {
            "similarity_score": similarity,
            "decision": decision,
            "ocr_text": ocr_text,
            "input_text": text,
            "from_url": from_url
        }
    
    def get_xai_explanation(self, image_input, text, from_url=False):
        """Get XAI explanation for image-text similarity"""
        explanation = self.xai_system.explain_prediction(
            image_input, text, from_url=from_url
        )
        
        return explanation