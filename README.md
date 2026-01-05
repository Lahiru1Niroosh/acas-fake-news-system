# 🩺 Sri Lankan Health/ Political News Fact Verification System  <br><br><br><br>

## Table of Contents
- [Project Overview](#Projectoverview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Training the Model](#-training-the-model)
- [Tools and Integrations](#tools-and-integrations)
- [API](#api)
- [Web Interface](#web-interface)
- [Components](#COMPONENET)
- [Evaluation & Charts](#-evaluation--charts)
- [Technologies Used](#-technologies-used)
- [License](#-license)
- [Author](#-author)
- [Future Improvements](#-future-improvements)


### 📌 Project Overview

**This is an implementation of a Dynamic, Evidence-based Fact-checking "Multimodal-Crew AI"-powered system to combat health misinformation in Sri Lanka through automated fact-checking, real-time source verification, and an explainable credibility scoring claim verification system**<br><br>

🎯 Research Problem

The rapid spread of misinformation through online news platforms and social media poses serious risks to society.
Traditional manual fact-checking is slow and inefficient. Therefore, there is a need for an automated, scalable, and reliable system that can assess the credibility of news articles using AI techniques.<br><br>




<br><br>
## Installation
#### Clone the Repository
```bash
git clone https://github.com/Lahiru1Niroosh/acas-fake-news-system.git
cd fake-news-detector
```
#### Install Dependencies
```bash
pip install -r requirements.txt
```


### Dataset
> **Dataset Source:**  
> This project uses and modifies the [*Name*](link) by **author** (Kaggle).  
> Data was cleaned, header-fixed, and news articles for balanced training and clear visualization.  
> Used purely for **educational and research** purposes.

## Training the Model
Run the following command from the project root:

## Evaluation & Charts
After training, the model achieves **perfect classification accuracy** on this dataset.


### Integrated APIs
Here's an overview of all APIs that are integrated






## ✨ Key Features

### 🤖 **Intelligent Multi-Model Analysis**

### 🌐 **Comprehensive Source Integration**





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

#### Software Requirements
Python 3.9+
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
XGBoost
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
# COMPONENT 2  - IT22


<br><br><br><br>
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

<br><br><br><br>
# COMPONENT 4  - IT22
























## Technologies Used

- **Python 3.11**
- **Development & Execution Environment** → Google Colab, Jupyter Notebook
- **Machine Learning & NLP** → PyTorch, Sentence-Transformers
- **Data Processing** →   Pandas, NumPy
- **scikit-learn** → TF-IDF Vectorizer, Logistic Regression  
- **matplotlib** → Model visualization
- **Web & Data Collection** →   Requests, BeautifulSoup
- **Databases** → MongoDB Atlas
- **Deployment & DevOps** →  FastAPI, Docker
- **Version Control** →  Git & GitHub



## Future Improvements

-Extend the system to analyze videos along side images, and text together.
-Extend the system across all social media platform posts not just twitter.
-Multilingual Support -Sinhala 🇱🇰, Tamil 🇱🇰, English  🌍Use multilingual transformers (mBERT, XLM-R).
-Browser extensions-Enable users to check credibility instantly.
-Ethical & Bias Mitigation Enhancements- fairness across regions and languages.
-Deploy live on **Hugging Face Spaces**.
