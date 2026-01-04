class TextAgent:
    def analyze(self, text: str):
        # TODO: teammate_model.predict(text)
        return {
            "score": 0.85,
            "label": "FAKE",
            "reason": "Linguistic patterns suggest health misinformation."
        }