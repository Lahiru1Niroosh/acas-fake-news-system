class SimilarityAgent:
    def finalize(self, text_res=None, img_res=None):
        # Case 1: Multi-modal (Text + Image)
        if text_res and img_res:
            final_score = (text_res['score'] + img_res['score']) / 2
            xai_reason = (f"Cross-check complete. Text flagged as {text_res['label']} "
                          f"but Image flagged as {img_res['label']}. Consistency mismatch detected.")
        
        # Case 2: Text Only
        elif text_res:
            final_score = text_res['score']
            xai_reason = f"Final verdict based on Text Analysis only. No image provided for cross-modal verification."
            
        # Case 3: Image Only
        elif img_res:
            final_score = img_res['score']
            xai_reason = f"Final verdict based on Image Analysis only. No textual context provided for cross-modal verification."
            
        return {
            "verdict": "FAKE" if final_score > 0.5 else "REAL",
            "confidence": round(final_score * 100),
            "reason": xai_reason
        }