<div align="center">

<br/>

```
 █████╗  ██████╗ █████╗ ███████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝
███████║██║     ███████║███████╗
██╔══██║██║     ██╔══██║╚════██║
██║  ██║╚██████╗██║  ██║███████║
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝
```

### **Automated Credibility Assessment System**
*Federated · Privacy-First · Explainable AI*

<br/>

[![Python](https://img.shields.io/badge/Python_3.10+-black?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-black?style=flat-square)](https://crewai.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-black?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![MongoDB](https://img.shields.io/badge/MongoDB_Atlas-black?style=flat-square&logo=mongodb)](https://mongodb.com)
[![SLIIT](https://img.shields.io/badge/SLIIT-Research_Project-black?style=flat-square)](https://sliit.lk)

<br/>

> *"In a world drowning in noise, ACAS doesn't just detect lies —*
> *it protects the people looking for truth."*

<br/>

</div>

---

## ◈ The Problem We're Solving

Every second, thousands of posts flood social media — some real, many fabricated, all mixed together. Existing tools either **sacrifice privacy** to analyze content or **sacrifice accuracy** for speed.

**ACAS does both.** It's a modular intelligence pipeline that strips away your identity before it ever touches your content. By the time your words reach our AI agents, you're anonymous. The truth gets verified. You stay protected.

---

## ◈ How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   RAW FEED                                                      │
│   ─────────                                                     │
│   "John Smith from Colombo says vaccines cause autism"          │
│                           │                                     │
│                           ▼                                     │
│              ┌────────────────────────┐                         │
│              │   🔒  PRIVACY SENTRY   │  ← spaCy + Regex        │
│              │   PII stripped cold    │                         │
│              └────────────┬───────────┘                         │
│                           │                                     │
│   SANITISED PAYLOAD                                             │
│   ─────────────────                                             │
│   "[NAME] from [LOCATION] says vaccines cause autism"           │
│                           │                                     │
│              ┌────────────▼───────────┐                         │
│              │  🧠  CrewAI ORCHESTRA  │                         │
│              └───┬────────┬───────┬───┘                         │
│                  │        │       │                             │
│           ┌──────▼──┐ ┌───▼───┐ ┌─▼──────┐                     │
│           │  TEXT   │ │ IMAGE │ │ MULTI  │                     │
│           │  AGENT  │ │ AGENT │ │ MODAL  │                     │
│           │  SBERT  │ │ResNet │ │  CLIP  │                     │
│           └──────┬──┘ └───┬───┘ └─┬──────┘                     │
│                  │        │       │                             │
│              ┌───▼────────▼───────▼───┐                         │
│              │   📊  AGGREGATOR + XAI │                         │
│              └────────────┬───────────┘                         │
│                           │                                     │
│              ┌────────────▼───────────┐                         │
│              │   🗄️  MongoDB Atlas    │                         │
│              └────────────┬───────────┘                         │
│                           │                                     │
│              ┌────────────▼───────────┐                         │
│              │   🖥️  React UI         │                         │
│              └────────────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**One rule: privacy is not a feature. It's a precondition.**
No data reaches any agent unless it has been fully sanitised first.

---

## ◈ The Four Agents

<br/>

### 〔 01 〕 Privacy Sentry & Orchestrator
> *"The gatekeeper that never sleeps."*

**Lead:** Lahiru Niroshan Sathsara · `IT22568632`

Before a single byte of user content reaches the AI layer, the Privacy Sentry intercepts it. Using a combination of **spaCy NLP** and battle-tested **Regex patterns**, every name, location, phone number, and email is redacted.

The CrewAI Orchestrator then takes the sanitised payload and routes it — intelligently — based on whether the content is text-only, image-only, or multimodal.

```
Capabilities
├── GDPR-compliant PII redaction (names, emails, locations, phone numbers)
├── Dynamic CrewAI routing by payload type
└── Plug-and-play model registry — drop .pkl → auto-registers
```

---

### 〔 02 〕 Neural Text Verifier
> *"Words lie. Semantics don't."*

**Lead:** K.K.K.G. Kalehewatta · `IT22576248`

Powered by **Sentence-BERT**, this agent doesn't just scan for keywords — it understands *meaning*. A claim about vaccine safety is cross-referenced live against **WHO**, **GNews**, and **NewsAPI** in real time. Logistic Regression and Random Forest classifiers then flag the linguistic fingerprints of misinformation.

```
Capabilities
├── Semantic similarity via SBERT (sentence-level embeddings)
├── Live fact-checking against WHO, GNews, NewsAPI
└── Misinformation classification — Logistic Regression + Random Forest
```

**Accuracy:** `92%` · **Latency:** `~400ms`

---

### 〔 03 〕 Visual Forensic Lab
> *"Every fake image leaves a trace. We find it."*

**Lead:** Ranasinghe K.D. · `IT22594990`

GAN-generated and diffusion-model images contain subtle frequency-domain artifacts invisible to the human eye. This agent runs every image through **ResNet-50** and **EfficientNet** to detect exactly that — returning a Softmax probability score of `Human-Made` vs `Synthetic`.

```
Capabilities
├── CNN-based artifact detection (ResNet-50 / EfficientNet)
├── Frequency-domain anomaly analysis
└── Confidence score: P(human) vs P(synthetic)
```

**Accuracy:** `89%` · **Latency:** `~800ms`

---

### 〔 04 〕 Multimodal Contextual Bridge
> *"Does the image match what the caption claims? Now we know."*

**Lead:** IT22067074

A meme can lie without a single false word — just by pairing a real photo with a fabricated caption. This agent uses **OpenAI CLIP** to project both the image and the text into the same vector space, then measures how far apart they are. **EasyOCR** also pulls any embedded text from the image itself to check for internal contradictions.

```
Capabilities
├── CLIP-based image-text semantic alignment
├── EasyOCR text extraction from memes and infographics
└── Cosine similarity thresholding (transparent, no black-box logic)
```

**Accuracy:** `94%` · **Latency:** `~600ms`

---

## ◈ Tech Stack

```
LAYER               TECHNOLOGIES
──────────────────────────────────────────────────────────────
Frontend          → React 18 (Vite) · Tailwind CSS · Framer Motion
Backend           → FastAPI · CrewAI · Pydantic
AI / ML           → PyTorch · SentenceTransformers · CLIP · EasyOCR · Scikit-Learn
Database          → MongoDB Atlas
Infrastructure    → Python 3.11 · GitHub · Google Colab (training)
──────────────────────────────────────────────────────────────
```

---

## ◈ Performance

| Agent | Model | Accuracy | Latency |
|:------|:------|:--------:|:-------:|
| Text Verifier | SBERT + Random Forest | **92%** | ~400ms |
| Visual Forensics | CNN (ResNet-50) | **89%** | ~800ms |
| Multimodal Bridge | CLIP (ViT-B/32) | **94%** | ~600ms |

---

## ◈ Getting Started

**Prerequisites:** Python 3.10+, Node.js 18+, MongoDB Atlas URI, NewsAPI key

```bash
# Clone
git clone https://github.com/your-repo/acas.git && cd acas

# Backend
pip install -r requirements.txt
cp .env.example .env          # add your keys
uvicorn main:app --reload

# Frontend
cd frontend && npm install && npm run dev
```

**Test it:**
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Claim to verify", "image_url": "optional"}'
```

---

## ◈ Adding a New Model (3 Steps)

```python
# 1 — Drop your model here
backend/models/your_model.pkl

# 2 — Register it
# config/models.yaml
models:
  - name: your_model
    path: models/your_model.pkl
    type: text   # text | image | multimodal

# 3 — You're done.
# Privacy Sentry wraps input automatically.
# Aggregator consumes your score automatically.
```

No rewiring. No boilerplate. The registry handles everything.

---

## ◈ Roadmap

```
v1.0  ████████████████████  COMPLETE — Core pipeline + all 4 agents
v1.1  ████████░░░░░░░░░░░░  IN PROGRESS — Real-time Twitter/X stream ingestion
v2.0  ░░░░░░░░░░░░░░░░░░░░  PLANNED — Sinhala 🇱🇰 + Tamil 🇱🇰 language support
v3.0  ░░░░░░░░░░░░░░░░░░░░  PLANNED — Deepfake video detection agent
v4.0  ░░░░░░░░░░░░░░░░░░░░  PLANNED — Browser extension with live Truth Overlay
```

---

## ◈ The Team

*Final Year Research Project · Department of Information Technology · SLIIT*

| | Name | ID | Role |
|:--:|:-----|:---|:-----|
| ◈ | Lahiru Niroshan Sathsara | IT22568632 | Architecture · Privacy · Orchestration |
| ◈ | K.K.K.G. Kalehewatta | IT22576248 | Text Intelligence |
| ◈ | Ranasinghe K.D. | IT22594990 | Visual Forensics |
| ◈ | — | IT22067074 | XAI · Multimodal Verification |

---

<div align="center">
<br/>

```
built at sliit · colombo, sri lanka · 2024
```

*"Truth is not a feature. It's a responsibility."*

<br/>
</div>
