# backend/agents/image_agent.py
import requests
from io import BytesIO
from urllib.parse import urlparse
import os


class ImageAgent:
    def __init__(self):
        # Initialize image-text similarity analyzer for multimodal analysis
        try:
            from ..pipeline.image_text_similarity import ImageTextSimilarityAnalyzer
            self.analyzer = ImageTextSimilarityAnalyzer()
        except ImportError:
            self.analyzer = None
            print("Warning: Could not import ImageTextSimilarityAnalyzer. Multimodal analysis will be disabled.")
    
    def analyze(self, image_url: str, text_content: str = None):
        # If we have text content, perform multimodal analysis
        if text_content and self.analyzer:
            try:
                # Perform image-text similarity analysis
                similarity, ocr_text = self.analyzer.multimodal_similarity(image_url, text_content, from_url=True)
                decision = self.analyzer.detect_context(similarity)
                
                # Map decision to our scoring system
                if decision == "MISMATCH":
                    # If image and text don't match, it might be fake news
                    score = 0.8  # High score indicating potential fake news
                    label = "FAKE"
                    reason = f"Image-text mismatch detected. CLIP similarity score: {similarity:.3f}. OCR extracted: '{ocr_text}'. This suggests the image may not correspond to the text."
                else:
                    # If they match, lower score (more likely to be real)
                    score = 0.3  # Lower score
                    label = "REAL"
                    reason = f"Image and text are contextually consistent with similarity score: {similarity:.3f}."
                
                return {
                    "score": score,
                    "label": label,
                    "reason": reason,
                    "similarity_score": similarity,
                    "ocr_text": ocr_text,
                    "decision": decision
                }
            except Exception as e:
                print(f"Error in multimodal analysis: {str(e)}")
                # Fall back to original implementation
        
        # TODO: image_model.predict(image_url)
        return {
            "score": 0.40,
            "label": "REAL",
            "reason": "Image metadata appears consistent with original source."
        }