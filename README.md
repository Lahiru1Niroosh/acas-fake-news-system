# 🩺 Sri Lankan Health News Fact Verification System

**This is an implementation of a Dynamic, Evidence-based Fact-checking "Multimodal-Crew AI"-powered system to combat health misinformation in Sri Lanka through automated fact-checking, real-time source verification, and an explainable credibility scoring claim verification system**

## Table of Contents
- [Installation](#installation)
- [Prepare Benchmarks](#prepare-benchmarks)
- [Usage](#usage)
- [Tools and Integrations](#tools-and-integrations)
- [API](#api)
- [Web Interface](#web-interface)
- [License](#license)
- [Cite this Work](#cite-this-work)



## Installation



### Integrated APIs
Here's an overview of all APIs that are integrated






## ✨ Key Features

### 🤖 **Intelligent Multi-Model Analysis**

### 🌐 **Comprehensive Source Integration**






# COMPONENT 1  - IT22576248




# COMPONENT 2  - IT22



# COMPONENT 3  - IT22

## Multimodal Contextual Similarity Verification Module

### Component Description
A multimodal module to verify contextual similarity between text and images using external tools and explainable AI principles.

---

### Project Overview
With the rapid spread of misinformation across digital platforms, verifying the contextual alignment between images and textual claims has become a critical challenge. This project proposes and implements a **Multimodal Contextual Similarity Verification Module** that evaluates whether an image and a given text are semantically and contextually aligned.

The system leverages a **pretrained CLIP (Contrastive Language–Image Pretraining)** model, integrates **OCR-based text extraction** for images containing embedded text, and applies **cosine similarity scoring** to detect contextual matches or mismatches. The solution is designed to be **scalable, explainable**, and suitable for **misinformation detection and fact-checking scenarios**.

---

### Objectives
- Verify contextual similarity between image and text inputs  
- Detect semantic mismatches indicative of misinformation  
- Integrate OCR for images containing embedded textual information  
- Support extension to external fact-checking tools (e.g., FactCrescendo)  
- Provide **explainable similarity decisions** rather than black-box predictions  

---

### System Architecture
The system follows a multimodal pipeline combining image processing, text understanding, and explainable decision logic.

**High-level flow:**


























---

### Explainable AI (XAI) Perspective
Unlike traditional classification-based models, this system is explicitly designed using **Explainable AI (XAI)** principles.

#### 1. Similarity Score Exposure
Each image–text pair produces a **numeric cosine similarity score** representing semantic alignment between visual and textual content.

#### 2. Threshold-Based Decision Logic
Decisions are made using a **transparent threshold mechanism**:
- Similarity ≥ Threshold → **MATCH**
- Similarity < Threshold → **MISMATCH**

The threshold value is **empirically adjusted** based on experimental observations.

#### 3. OCR Evidence Integration
For images containing embedded text, OCR-extracted content is:
- Explicitly displayed
- Included in similarity computation  
This allows users to understand **why** a decision was made.

#### 4. Interpretability Over Prediction
Instead of producing opaque predictions, the system outputs:
- Similarity score
- OCR-extracted text (if present)
- Final contextual decision  

This design makes the system suitable for **fact-checking, misinformation analysis, and research explainability requirements**.

---

### Technologies & Tools Used

###$ AI & ML Models
- **CLIP (ViT-B/32)** – Pretrained multimodal image–text model
- **EasyOCR** – Optical Character Recognition for embedded image text

#### Programming & Libraries
- Python 3  
- PyTorch  
- NumPy  
- Pandas  
- Pillow (PIL)  

#### Platforms
- Google Colab (GPU-enabled)  
- GitHub (Version Control)  

---

### Datasets
- **Custom Test Images**
  - Used to demonstrate real-world scenarios and explainability.

---

### Decision Logic
A cosine similarity threshold is applied to determine contextual alignment:
- Similarity ≥ Threshold → **MATCH**
- Similarity < Threshold → **MISMATCH**

The threshold is adjustable and tuned based on experimental results to balance sensitivity and robustness.

---

### External Verification (Conceptual Extension)
The architecture supports future integration with external verification tools such as:
- Web search APIs
- Fact-checking platforms (e.g., FactCrescendo)

These extensions can further validate detected mismatches against trusted external sources.

---

### Version Control & Collaboration
This project uses **GitHub** for:
- Version control
- Commit history tracking
- Branching and merging
- Collaborative development  

All project progress, updates, and experimental changes are maintained with full history for evaluation and transparency.

---

### Academic Context
This project is developed as a **Final Year Undergraduate IT Research Component**, emphasizing:
- Multimodal AI
- Explainable AI (XAI)
- Misinformation detection
- Practical applicability and research transparency

---

### License
This project is intended for academic and research purposes.


# COMPONENT 4  - IT22
