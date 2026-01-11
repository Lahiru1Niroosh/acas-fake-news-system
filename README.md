# ACAS – Fake News Detection & Privacy-Aware Verification System

## 📌 Project Overview
ACAS (Automated Credibility Assessment System) is a privacy-first, multi-agent fake news detection platform designed to verify health-related and news-related social media content.  
The system uses a **CrewAI-based agent orchestration framework** combined with **FastAPI** and **React** to analyze tweets containing text, images, or both, while ensuring **user metadata privacy through PII masking**.

The platform demonstrates a **plug-and-play architecture**, allowing independently trained AI models to be integrated seamlessly into the verification pipeline.

---

## 🎯 Problem Addressed
The rapid spread of fake news—especially in health, political, and public safety domains—causes misinformation, panic, and loss of trust.

Existing solutions often lack:
- Privacy protection
- Explainable decisions
- Modular AI integration

ACAS addresses these gaps through **privacy-aware processing, explainable AI (XAI), and modular agent orchestration**.

---

## 🧠 System Architecture (High-Level)

Frontend (React + Tailwind)
↓  
FastAPI `/verify` API  
↓  
CrewAI Pipeline Controller  
↓  
Privacy Agent (PII Masking) → MongoDB (Raw + Masked Metadata)  
↓  
Routing Logic  
- Text → Text Agent  
- Image → Image Agent  
- Text + Image → Both Agents  
↓  
Similarity Agent (Final Gate + XAI)  
↓  
MongoDB (Agent Scores + Final Verdict)  
↓  
Frontend Modal (Result + Explanation)

---

## 🧩 Individual Component Contribution
**Author:** Sathsara W.H.A.L.N - IT22568632

### 🔹 Component Title
**Plug-and-Play Agent-Orchestration Framework (ACIF) + Privacy Agent**

### 🔹 Description
This component ensures that user privacy is preserved before any AI-based analysis and orchestrates multiple AI agents using CrewAI in a strict linear pipeline.

### 🔹 Key Responsibilities
- Designed and implemented the **Privacy Agent** for PII masking
- Enforced **mandatory execution order** using CrewAI
- Built routing logic for text, image, and multimodal tweets
- Enabled **plug-and-play integration** for externally trained ML models

---

## 🧑‍🤝‍🧑 CrewAI Agents & Roles

| Agent Name | Role | Responsibility |
|-----------|------|----------------|
| Privacy_Agent | PII Masking Officer | Masks all user metadata before processing |
| Text_Agent | Linguistic Analyst | Analyzes textual credibility |
| Image_Agent | Visual Forensic Expert | Analyzes image credibility |
| Similarity_Agent | Final Decision Gatekeeper & XAI Specialist | Produces final verdict and explanation |

---

## 🔁 Pipeline Execution Flow (Mandatory)

1. Frontend sends raw tweet + user metadata
2. **Privacy Agent runs first (non-skippable)**
3. Routing logic selects relevant agents:
   - Text only → Text Agent
   - Image only → Image Agent
   - Text + Image → Both agents
4. Outputs passed to Similarity Agent
5. Final credibility score + XAI reason generated
6. All results logged in MongoDB
7. Response returned to frontend

---

## 🗄️ MongoDB Data Storage

Each verification stores:
- Original tweet data
- Raw user metadata (audit purpose)
- Masked user metadata
- Individual agent scores
- Final credibility score
- Explainable AI (XAI) reason

---

## 🔌 Plug-and-Play Model Integration

External teams can integrate their trained models by adding them to:

backend/models/
- text_model.pkl
- image_model.pkl

Integration points are clearly marked in code:



## 🖥️ Frontend Features

- Twitter-like live feed
- Replaces 5 fake tweets every 10 seconds
- Supports text-only, image-only, and multimodal tweets
- Verify button triggers backend pipeline
- Modal displays:
  - Credibility label
  - Confidence score
  - XAI explanation
  - Masked user ID

---

## ⚙️ Project Dependencies

### Backend
- Python 3.10+
- FastAPI
- CrewAI
- Pydantic
- PyMongo
- Uvicorn

### Frontend
- React (Vite)
- Tailwind CSS
- JavaScript (ES6+)

### Database
- MongoDB Atlas / Local MongoDB

---

## 📁 Repository Structure

acas-fake-news-system/
├── backend/
│   ├── agents/
│   ├── pipeline/
│   ├── routes/
│   ├── db/
│   └── main.py
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
└── README.md

---

## 🔄 Version Control & Collaboration

- Git-based development
- Feature-based commits
- Modular agent architecture
- Designed for collaborative model integration
- Supports branching and merging workflows

---

## 🚀 Future Enhancements
- Integration of real trained ML models
- Docker-based deployment
- Real-time social media ingestion
- Advanced analytics dashboards

---

## 👤 Author
**Lahiru Niroshan Sathsara**  
Final Year IT Undergraduate – SLIIT  
Component: Agent Orchestration & Privacy Infrastructure

---
### Dataset
> **Dataset Source:**  
> This project uses and modifies the [*Name*](link) by **author** (Kaggle).  
> Data was cleaned, header-fixed, and news articles for balanced training and clear visualization.  
> Used purely for **educational and research** purposes.
---
## Training the Model
Run the following command from the project root:
---
## Evaluation & Charts
After training, the model achieves **perfect classification accuracy** on this dataset.

---
### Integrated APIs
Here's an overview of all APIs that are integrated





---


<br><br><br><br>

# COMPONENT 1  - Text Credibility Assessment
IT22576248 -K.K.K.G. Kalehewatta

### Component Overview

This Component is an AI-based Text Credibility Assessment System (Text Credibility & Fact-Checking System) designed to detect fake or misleading news content.
The system combines Natural Language Processing (NLP), Machine Learning, Semantic Similarity, and News API–based verification to evaluate the trustworthiness of textual information.

👉This model reads news text
👉 Analyzes language, meaning, and writing style
👉 Checks credibility using ML + Deep Learning
👉 Verifies claims using external sources (WHO, News APIs)
👉 Produces a final credibility verdict with confidence

### Research Gap

### 🧠 System Features
Text preprocessing and normalization
Sentence embeddings using SBERT
Machine learning classification 
Trust score calculation using cosine similarity
External verification using News APIs
Visualization of label and trust score distributions
Prediction support for new/unseen text

### 🏗️ System Architecture (High Level)
Data Collection
Text Preprocessing
Embedding Generation
ML Model Training
News API Verification
Trust Score Calculation
Result Visualization & Output

### ⚙️ System & Software Requirements
Hardware Requirements
CPU (Intel i5 or higher recommended)
Minimum 8GB RAM
GPU (Optional, for faster embedding generation)
Python 3.11
Google Colab or Local Python Environment
Internet connection (for APIs)

### 🧪 Technologies, Techniques & Algorithms
Programming Language
Python
Libraries & Frameworks
Pandas, NumPy
Scikit-learn
SentenceTransformers (SBERT)
Hugging Face Transformers
Matplotlib, Seaborn

### 🌐 External APIs Used (Verification)
News Verification APIs
NewsAPI.org
GNews API
WorldNewsAPI

### 🔍 Methodology
Load and preprocess the dataset
Clean and normalize text
Generate sentence embeddings
Train ML classification model
Verify claims using News APIs
Compute trust score
Visualize results and save outputs

### 📊 Evaluation Metrics
Accuracy
Precision
Recall
F1-Score


<br><br><br><br>
---
# 🖼️ COMPONENT 2  - Image Authenticity Detection 

IT22594990 – Ranasinghe K.D

### 1. Component Overview

This component focuses on Image Authenticity Detection, aiming to determine whether an image shared on social media platforms is AI-generated or human-made (real). With the rapid growth of AI-generated visual content, visual misinformation has become increasingly difficult to detect through manual inspection.

The Image Authenticity Detection module leverages deep learning–based computer vision techniques to analyze visual artifacts, textures, and structural patterns that differentiate synthetic images from real photographs. This component operates as an ImageAgent within the CrewAI multi-agent framework and provides reliable image credibility outputs to the AggregatorAgent for multimodal credibility assessment.

Input: Image from social media
Output: AI-generated or Human-made label with a confidence score

### 2. System Features

Detection of AI-generated vs Human-made images
Automated image preprocessing (resizing, normalization)
Feature extraction using deep learning models
Supervised learning with labeled datasets
Confidence score generation (range: 0–1)
Integration with CrewAI multi-agent architecture
Storage of predictions and metadata in PostgreSQL
Scalable and real-time–ready system design

### 3. System Architecture (High-Level)

Image Ingestion (Dataset / Twitter Images)
Image Preprocessing
Feature Extraction using CNN
Image Classification (AI-generated vs Human-made)
Confidence Score Generation
Result Storage (PostgreSQL)
Output forwarded to the AggregatorAgent

### 4. System & Software Requirements

Software Requirements
- Python 3.10 / 3.11
- Google Colab or Local Python Environment
- Internet connection for dataset access

### 5. Technologies, Techniques & Algorithms
Programming Language
- Python
- Libraries & Frameworks
- PyTorch / TensorFlow
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Deep Learning Models
- Convolutional Neural Networks (CNN)
- Transfer Learning (optional – ResNet / EfficientNet)

### 6. External APIs Used

Kaggle API – for dataset downloading
(Twitter API planned for future fine-tuning and evaluation)

### 7. Methodology

Download labeled image datasets from Kaggle
Preprocess images (resizing, normalization, augmentation)
Split the dataset into training and testing sets
Train a CNN-based image classification model
Validate the model using unseen test images
Generate prediction labels and confidence scores
Store results for analysis and reporting

### 8. Evaluation Metrics

Model performance is evaluated using the following metrics:
Accuracy
Precision
Recall
F1-Score

These metrics ensure reliable classification between AI-generated and human-made images.

### 9. Dataset
Dataset Used

Real vs Fake AI Image Dataset
🔗 https://www.kaggle.com/datasets/virajinduruwa/real-vs-fakeai-image-dataset

#### Dataset Description

Two classes:
real → Human-made images
fake → AI-generated images

Fully labeled dataset suitable for supervised learning

### 10. Decision Logic

The decision-making process follows this logic:

CNN extracts visual features such as textures, patterns, and artifacts
The model compares extracted features with learned distributions from training data

#### The output layer predicts:

AI-generated or Human-made
A confidence score reflects prediction certainty
Final results are forwarded to the AggregatorAgent for multimodal credibility fusion


<br><br><br><br>
---
# COMPONENT 3  - IT22067074

## Multimodal Contextual Similarity Verification Module

### Component Description
A multimodal module to verify contextual similarity between text and images using external tools and explainable AI principles.


### Project Overview
With the rapid spread of misinformation across digital platforms, verifying the contextual alignment between images and textual claims has become a critical challenge. This project proposes and implements a **Multimodal Contextual Similarity Verification Module** that evaluates whether an image and a given text are semantically and contextually aligned.

The system leverages a **pretrained CLIP (Contrastive Language–Image Pretraining)** model, integrates **OCR-based text extraction** for images containing embedded text, and applies **cosine similarity scoring** to detect contextual matches or mismatches. The solution is designed to be **scalable, explainable**, and suitable for **misinformation detection and fact-checking scenarios**.

### Objectives
- Verify contextual similarity between image and text inputs  
- Detect semantic mismatches indicative of misinformation  
- Integrate OCR for images containing embedded textual information  
- Support extension to external fact-checking tools (e.g., FactCrescendo)  
- Provide **explainable similarity decisions** rather than black-box predictions  


### System Architecture
The system follows a multimodal pipeline combining image processing, text understanding, and explainable decision logic.

**High-level flow:**




























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


### Datasets
- **Custom Test Images**
  - Used to demonstrate real-world scenarios and explainability.


### Decision Logic
A cosine similarity threshold is applied to determine contextual alignment:
- Similarity ≥ Threshold → **MATCH**
- Similarity < Threshold → **MISMATCH**

The threshold is adjustable and tuned based on experimental results to balance sensitivity and robustness.




<br><br><br><br>
---

<br><br><br><br>
---
## Technologies Used

- **Python 3.11**
- **Development & Execution Environment** → Google Colab, Jupyter Notebook <br>
- **Machine Learning & NLP** → PyTorch, Sentence-Transformers <br>
- **Data Processing** →   Pandas, NumPy <br>
- **scikit-learn** → TF-IDF Vectorizer, Logistic Regression   <br>
- **matplotlib** → Model visualization <br>
- **Web & Data Collection** →   Requests, BeautifulSoup <br>
- **Databases** → MongoDB Atlas <br>
- **Deployment & DevOps** →  FastAPI, Docker <br>
- **Version Control** →  Git & GitHub <br>

---
<br><br>
## Future Improvements

-Extend the system to analyze videos along side images, and text together.<br>
-Extend the system across all social media platform posts not just twitter.<br>
-Multilingual Support -Sinhala 🇱🇰, Tamil 🇱🇰, English  🌍Use multilingual transformers (mBERT, XLM-R).<br>
-Browser extensions-Enable users to check credibility instantly.<br>
-Ethical & Bias Mitigation Enhancements- fairness across regions and languages.<br>
-Deploy live on **Hugging Face Spaces**.<br>
---
<br><br>
## License
This project is intended for academic and research purposes.
---
