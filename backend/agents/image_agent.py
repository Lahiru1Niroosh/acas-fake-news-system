# backend/agents/image_agent.py
class ImageAgent:
    def analyze(self, image_url: str):
        # TODO: image_model.predict(image_url)
        return {
            "score": 0.40,
            "label": "REAL",
            "reason": "Image metadata appears consistent with original source."
        }