class SimilarityAgent:
    def finalize(self, text_res=None, img_res=None):
        # Case 1: Multi-modal (Text + Image)
        if text_res and img_res:
            # Check if the image agent performed multimodal analysis
            if img_res.get('similarity_score') is not None:
                # Use the similarity-based analysis from image agent
                # If image-text mismatch was detected, increase the final score
                if img_res.get('decision') == 'MISMATCH':
                    # Increase score since there's a mismatch
                    final_score = min(0.9, img_res['score'])  # Boost to high score but cap it
                    xai_reason = (f"Critical inconsistency detected: Image-text mismatch identified. "
                                  f"CLIP similarity score: {img_res['similarity_score']:.3f}, OCR: '{img_res['ocr_text']}'. "
                                  f"Text analysis: {text_res['label']}, Image analysis: {img_res['label']}. "
                                  f"Strong indicator of potential misinformation.")
                else:
                    # Image and text match, combine scores more conservatively
                    final_score = (text_res['score'] + img_res['score']) / 2
                    xai_reason = (f"Cross-verification completed. Text analysis: {text_res['label']} "
                                  f"and Image analysis: {img_res['label']} are consistent. "
                                  f"CLIP similarity score: {img_res['similarity_score']:.3f}. "
                                  f"Overall consistency suggests credibility.")
            else:
                # Traditional analysis without multimodal features
                final_score = (text_res['score'] + img_res['score']) / 2
                xai_reason = (f"Cross-check complete. Text flagged as {text_res['label']} "
                              f"but Image flagged as {img_res['label']}. Consistency mismatch detected.")
        
        # Case 2: Text Only
        elif text_res:
            final_score = text_res['score']
            xai_reason = f"Final verdict based on Text Analysis only. No image provided for cross-modal verification."
            
        # Case 3: Image Only
        elif img_res:
            if img_res.get('similarity_score') is not None:
                # Use the multimodal analysis result
                final_score = img_res['score']
                xai_reason = (f"Final verdict based on Image Analysis with multimodal verification. "
                              f"Image-text similarity score: {img_res['similarity_score']:.3f}, OCR: '{img_res['ocr_text']}', Decision: {img_res['decision']}. "
                              f"Cross-modal verification completed.")
            else:
                # Traditional image-only analysis
                final_score = img_res['score']
                xai_reason = f"Final verdict based on Image Analysis only. No textual context provided for cross-modal verification."
                
        return {
            "verdict": "FAKE" if final_score > 0.5 else "REAL",
            "confidence": round(final_score * 100),
            "reason": xai_reason
        }