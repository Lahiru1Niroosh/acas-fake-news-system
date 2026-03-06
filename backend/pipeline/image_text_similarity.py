import os
import torch
import clip
import numpy as np
from PIL import Image
import pytesseract
import easyocr
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision.transforms as transforms
import requests
from io import BytesIO
import tempfile
import json
from datetime import datetime
import traceback
import base64
from io import BytesIO as IOBytesIO


class ImageTextSimilarityAnalyzer:
    def __init__(self):
        # Initialize device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

        # Initialize OCR reader
        self.reader = easyocr.Reader(['en'])

    def load_image_from_url(self, image_url):
        """Load image from URL"""
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def extract_ocr_text(self, image_input, from_url=False):
        """Extract OCR text from image"""
        if from_url:
            image = self.load_image_from_url(image_input)
        else:
            image = Image.open(image_input).convert("RGB")

        result = self.reader.readtext(np.array(image), detail=0)
        return " ".join(result)

    def encode_image(self, image_input, from_url=False):
        """Encode image using CLIP"""
        if from_url:
            image = self.load_image_from_url(image_input)
        else:
            image = Image.open(image_input).convert("RGB")

        image = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

        return image_features / image_features.norm(dim=-1, keepdim=True)

    def encode_text(self, text):
        """Encode text using CLIP"""
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features / text_features.norm(dim=-1, keepdim=True)

    def cosine_similarity(self, image_feat, text_feat):
        """Calculate cosine similarity between image and text features"""
        return (image_feat @ text_feat.T).item()

    def multimodal_similarity(self, image_input, caption, from_url=False):
        """Calculate multimodal similarity considering both image-text and OCR-text"""
        # CLIP features
        img_feat = self.encode_image(image_input, from_url=from_url)
        txt_feat = self.encode_text(caption)

        # OCR text
        ocr_text = self.extract_ocr_text(image_input, from_url=from_url)
        if ocr_text.strip() != "":
            ocr_feat = self.encode_text(ocr_text)
            combined_text_feat = (txt_feat + ocr_feat) / 2
        else:
            combined_text_feat = txt_feat

        similarity_score = self.cosine_similarity(img_feat, combined_text_feat)

        return similarity_score, ocr_text

    def detect_context(self, similarity, threshold=0.24):
        """Detect if image and text match based on similarity score"""
        if similarity >= threshold:
            return "MATCH"
        else:
            return "MISMATCH"


# ----------------------------------------------------------------------------
# 1. GRAD-CAM FOR VISUAL EXPLANATION
# ----------------------------------------------------------------------------
class ClipGradCAM:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.visual = model.visual
        self.gradients = None
        self.activations = None

        # Hook into the transformer blocks
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Hook into the last transformer block
        last_block = self.visual.transformer.resblocks[-1]
        self.hooks.append(last_block.register_forward_hook(forward_hook))
        self.hooks.append(last_block.register_backward_hook(backward_hook))

    def generate_cam(self, image_tensor, text_tensor):
        """
        Generate Grad-CAM for image-text similarity
        """
        # Forward pass
        image_features = self.visual(image_tensor)
        text_features = self.model.encode_text(text_tensor)

        # Compute similarity score
        similarity = (image_features @ text_features.T).mean()

        # Backward pass
        self.model.zero_grad()
        similarity.backward()

        # Get gradients and activations
        gradients = self.gradients.mean(dim=[0, 1])  # [num_patches, 768]
        activations = self.activations[0]  # [num_patches + 1, 768]

        # Remove CLS token
        gradients = gradients[1:]  # [196, 768]
        activations = activations[1:]  # [196, 768]

        # Compute weights
        weights = gradients.mean(dim=1)  # [196]

        # Generate CAM
        cam = torch.zeros(activations.shape[0], dtype=torch.float32)
        for i in range(weights.shape[0]):
            cam[i] = torch.sum(weights[i] * activations[i])

        # Reshape to 14x14 (ViT-B/32 patch grid)
        cam = cam.reshape(14, 14)
        cam = torch.relu(cam)  # ReLU to keep positive contributions

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam.cpu().detach().numpy(), similarity.item()

    def visualize(self, original_image, cam, similarity_score):
        """
        Visualize Grad-CAM overlay on original image and return as base64
        """
        # Convert tensor image to numpy if needed
        if torch.is_tensor(original_image):
            original_image = original_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

        # Apply colormap
        heatmap = cm.jet(cam_resized)[:, :, :3]

        # Overlay heatmap on original image
        overlayed = heatmap * 0.5 + original_image * 0.5

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')

        axes[2].imshow(overlayed)
        axes[2].set_title(f'Overlay (Similarity: {similarity_score:.3f})')
        axes[2].axis('off')

        plt.tight_layout()
        
        # Save plot to base64 string
        img_buffer = IOBytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)  # Close figure to free memory
        
        return img_str

    def __del__(self):
        for hook in self.hooks:
            hook.remove()


# ----------------------------------------------------------------------------
# 2. TEXT ATTENTION VISUALIZATION
# ----------------------------------------------------------------------------
class TextAttentionExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_weights = None

        # Hook to capture attention weights
        self.hooks = []
        self.register_hooks()

    def register_hooks(self):
        def attention_hook(module, input, output):
            # output[1] contains attention weights in CLIP
            if len(output) > 1:
                self.attention_weights = output[1]

        # Hook into the last attention layer
        last_attn = self.model.transformer.resblocks[-1].attn
        self.hooks.append(last_attn.register_forward_hook(attention_hook))

    def explain_text_importance(self, text, image_features):
        """
        Show which words are important for the match
        """
        # Tokenize text
        tokens = clip.tokenize([text]).to(self.device)

        # Get text features
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)

        # Get attention weights (average across heads and layers)
        if self.attention_weights is not None:
            # Average over attention heads and select CLS token attention
            attn = self.attention_weights.mean(dim=1)[0, 0, :]  # CLS token attention to all tokens

            # Decode tokens (simplified - for visualization only)
            token_ids = tokens[0].cpu().numpy()

            # Return token importance scores
            return {
                'attention_scores': attn.cpu().detach().numpy(),
                'text_features': text_features.cpu().numpy()
            }

        return None

    def visualize_text_attention(self, text, attention_scores):
        """
        Visualize word importance and return as base64
        """
        # Simple token splitting (in practice, use proper tokenizer)
        words = text.split()

        # Pad or truncate attention scores
        if len(attention_scores) > len(words):
            attention_scores = attention_scores[:len(words)]
        elif len(attention_scores) < len(words):
            attention_scores = np.pad(attention_scores,
                                      (0, len(words) - len(attention_scores)))

        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(range(len(words)), attention_scores[:len(words)])

        # Color bars by importance
        for bar, score in zip(bars, attention_scores[:len(words)]):
            if max(attention_scores) > 0:
                bar.set_color(plt.cm.Reds(score / max(attention_scores)))

        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=45, ha='right')
        ax.set_ylabel('Attention Score')
        ax.set_title('Word Importance for Image-Text Matching')

        plt.tight_layout()
        
        # Save plot to base64 string
        img_buffer = IOBytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)  # Close figure to free memory
        
        return img_str


# ----------------------------------------------------------------------------
# 3. SIMILARITY DECOMPOSITION
# ----------------------------------------------------------------------------
class SimilarityDecomposer:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # Define concept categories
        self.concepts = {
            'objects': ['person', 'dog', 'cat', 'car', 'tree', 'house', 'chair', 'table'],
            'actions': ['running', 'standing', 'sitting', 'eating', 'drinking', 'talking'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
            'locations': ['indoors', 'outdoors', 'street', 'park', 'beach', 'room'],
            'attributes': ['young', 'old', 'large', 'small', 'bright', 'dark']
        }

    def decompose_similarity(self, image_path, caption):
        """
        Break down similarity score by concept categories
        """
        # Get original similarity using your existing function
        analyzer = ImageTextSimilarityAnalyzer()
        original_score, _ = analyzer.multimodal_similarity(image_path, caption)

        # Encode image
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Analyze by concepts
        decomposition = {}

        for category, concepts in self.concepts.items():
            category_scores = []

            for concept in concepts:
                # Create concept text
                concept_text = f"A photo of {concept}"
                text_tokens = clip.tokenize([concept_text]).to(self.device)

                with torch.no_grad():
                    concept_features = self.model.encode_text(text_tokens)
                    concept_features = concept_features / concept_features.norm(dim=-1, keepdim=True)

                # Calculate concept similarity
                concept_sim = (image_features @ concept_features.T).item()
                category_scores.append((concept, concept_sim))

            # Sort by similarity
            category_scores.sort(key=lambda x: x[1], reverse=True)
            decomposition[category] = category_scores[:5]  # Top 5

        # Overall score breakdown
        breakdown = {
            'overall_similarity': original_score,
            'concept_decomposition': decomposition,
            'top_concepts': []
        }

        # Get top overall concepts
        all_concepts = []
        for category, scores in decomposition.items():
            all_concepts.extend(scores)

        all_concepts.sort(key=lambda x: x[1], reverse=True)
        breakdown['top_concepts'] = all_concepts[:10]

        return breakdown

    def visualize_decomposition(self, breakdown):
        """
        Visualize similarity breakdown and return as base64
        """
        fig = plt.figure(figsize=(15, 10))

        # 1. Overall score
        ax1 = plt.subplot(2, 2, 1)
        ax1.bar(['Overall Similarity'], [breakdown['overall_similarity']])
        ax1.set_ylim(0, 1)
        ax1.set_title(f'Overall Score: {breakdown["overall_similarity"]:.3f}')
        ax1.set_ylabel('Similarity')

        # 2. Top concepts
        ax2 = plt.subplot(2, 2, 2)
        top_concepts = breakdown['top_concepts'][:8]
        concepts = [c[0] for c in top_concepts]
        scores = [c[1] for c in top_concepts]

        bars = ax2.barh(range(len(concepts)), scores)
        ax2.set_yticks(range(len(concepts)))
        ax2.set_yticklabels(concepts)
        ax2.set_xlim(0, 1)
        ax2.set_title('Top Image Concepts')
        ax2.set_xlabel('Similarity')

        # Color bars
        for bar, score in zip(bars, scores):
            bar.set_color(plt.cm.Blues(score))

        # 3. Concept category breakdown
        ax3 = plt.subplot(2, 2, 3)
        categories = list(breakdown['concept_decomposition'].keys())
        avg_scores = []

        for category in categories:
            scores = [s[1] for s in breakdown['concept_decomposition'][category]]
            avg_scores.append(np.mean(scores))

        bars = ax3.bar(categories, avg_scores)
        ax3.set_ylim(0, 1)
        ax3.set_title('Average Similarity by Concept Category')
        ax3.set_ylabel('Average Similarity')
        plt.xticks(rotation=45, ha='right')

        # Color bars
        for bar, score in zip(bars, avg_scores):
            bar.set_color(plt.cm.Greens(score))

        # 4. Decision explanation
        ax4 = plt.subplot(2, 2, 4)
        decision = "MATCH" if breakdown['overall_similarity'] >= 0.25 else "MISMATCH"

        explanation = f"""
        DECISION: {decision}

        Key Factors:
        1. Overall similarity: {breakdown['overall_similarity']:.3f}
        2. Strongest concept: {breakdown['top_concepts'][0][0]} ({breakdown['top_concepts'][0][1]:.3f})
        3. Dominant category: {categories[np.argmax(avg_scores)]}

        Reasoning:
        The image and text {'share strong visual concepts' if breakdown['overall_similarity'] >= 0.25 else 'lack common visual concepts'}.
        """

        ax4.text(0.1, 0.5, explanation, fontsize=10,
                verticalalignment='center', transform=ax4.transAxes)
        ax4.axis('off')
        ax4.set_title('Decision Explanation')

        plt.tight_layout()
        
        # Save plot to base64 string
        img_buffer = IOBytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)  # Close figure to free memory
        
        return img_str


# ----------------------------------------------------------------------------
# 4. INTEGRATED XAI PIPELINE
# ----------------------------------------------------------------------------
class MultimodalXAI:
    def __init__(self, model, preprocess, device):
        self.model = model
        self.preprocess = preprocess
        self.device = device

        # Initialize explainers
        self.gradcam = ClipGradCAM(model, device)
        self.text_explainer = TextAttentionExplainer(model, device)
        self.decomposer = SimilarityDecomposer(model, preprocess, device)

    def explain_prediction(self, image_path, caption, from_url=False):
        """
        Complete XAI pipeline for one image-text pair
        """
        # Initialize analyzer for similarity calculation
        analyzer = ImageTextSimilarityAnalyzer()
        
        # Get original prediction using your existing function
        similarity, ocr_text = analyzer.multimodal_similarity(image_path, caption, from_url=from_url)
        decision = analyzer.detect_context(similarity)

        # Prepare image for Grad-CAM
        if from_url:
            image = analyzer.load_image_from_url(image_path)
        else:
            image = Image.open(image_path)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize([caption]).to(self.device)

        # Generate Grad-CAM
        try:
            cam, cam_similarity = self.gradcam.generate_cam(image_tensor, text_tokens)

            # Convert original image for visualization
            original_np = image_tensor.squeeze(0).cpu()
            original_np = (original_np - original_np.min()) / (original_np.max() - original_np.min())

            # Get Grad-CAM visualization as base64
            gradcam_img = self.gradcam.visualize(original_np, cam, similarity)
        except Exception as e:
            print(f"⚠️  Grad-CAM failed: {e}")
            gradcam_img = None

        # Text attention analysis
        try:
            text_analysis = self.text_explainer.explain_text_importance(caption, None)

            if text_analysis and 'attention_scores' in text_analysis:
                text_attention_img = self.text_explainer.visualize_text_attention(
                    caption,
                    text_analysis['attention_scores']
                )
            else:
                text_attention_img = None
        except Exception as e:
            print(f"⚠️  Text analysis failed: {e}")
            text_attention_img = None

        # Similarity decomposition
        try:
            decomposition = self.decomposer.decompose_similarity(image_path, caption)
            decomp_img = self.decomposer.visualize_decomposition(decomposition)
        except Exception as e:
            print(f"⚠️  Decomposition failed: {e}")
            decomposition = None
            decomp_img = None

        # Generate comprehensive explanation
        explanation = {
            'similarity_score': similarity,
            'decision': decision,
            'ocr_text': ocr_text,
            'input_text': caption,
            'explanation': f"This {'matches' if decision == 'MATCH' else 'does not match'} because the visual content and text have a similarity score of {similarity:.4f} compared to the threshold of 0.24.",
            'confidence_level': 'high' if abs(similarity - 0.24) > 0.1 else 'medium' if abs(similarity - 0.24) > 0.05 else 'low',
            'visualizations': {
                'gradcam': gradcam_img,
                'text_attention': text_attention_img,
                'decomposition': decomp_img
            },
            'decomposition_details': decomposition
        }

        return explanation