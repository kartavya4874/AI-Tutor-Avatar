# Software Requirements Specification (SRS)
## AI-Powered Virtual Teacher with Personalized Avatar & Voice

**Version:** 1.1  
**Date:** September 2025  
**Team Size:** 2 Developers  
**Project Type:** Final Year Academic Project  
**Tech Stack:** Full Python Implementation

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [System Features](#3-system-features)
4. [External Interface Requirements](#4-external-interface-requirements)
5. [System Requirements](#5-system-requirements)
6. [Non-Functional Requirements](#6-non-functional-requirements)
7. [Technical Architecture](#7-technical-architecture)
8. [Implementation Timeline](#8-implementation-timeline)
9. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for an AI-Powered Virtual Teacher system that provides personalized education through a talking avatar with voice cloning capabilities and intelligent content refinement, built entirely in Python.

### 1.2 Document Scope

This SRS covers functional and non-functional requirements, system architecture, and technical specifications for the Virtual Teacher system using a full Python technology stack.

### 1.3 Intended Audience

- Final year computer science students (developers)
- Academic supervisors
- Project evaluators
- Future maintainers

### 1.4 Project Overview

The system combines Natural Language Processing, Voice Cloning, and Avatar Animation using Python technologies to create an interactive virtual teacher that explains concepts using a personalized voice and visual representation.

---

## 2. Overall Description

### 2.1 Product Perspective

The Virtual Teacher system is a standalone educational application built entirely in Python that integrates multiple AI technologies to provide an immersive learning experience.

### 2.2 Product Functions

- **Intelligent Query Processing:** Understands student questions via text/voice
- **Knowledge Refinement:** Converts raw AI responses into structured teaching explanations
- **Voice Cloning:** Synthesizes speech using user's cloned voice
- **Avatar Animation:** Provides lip-synced visual representation
- **Interactive Learning:** Enables Q&A sessions and quizzes

### 2.3 User Classes

- **Primary Users:** Students (age 16-25)
- **Secondary Users:** Educators (for voice setup and content customization)
- **System Administrators:** Technical maintenance personnel

### 2.4 Operating Environment

- **Platform:** Python web application (Streamlit) or desktop application (PyQt6)
- **Browsers:** Chrome, Firefox, Safari, Edge (for web version)
- **Hardware:** Standard desktop/laptop with microphone and speakers
- **Internet:** Required for cloud-based AI services

---

## 3. System Features

### 3.1 Query Input Module

**Priority:** High

#### 3.1.1 Description

Accepts student queries through multiple input modalities using Python libraries.

#### 3.1.2 Functional Requirements

- **FR-1.1:** System shall accept text input via Python web interface (Streamlit/Flask)
- **FR-1.2:** System shall accept voice input using sounddevice/pyaudio and convert to text
- **FR-1.3:** System shall validate and sanitize input queries using Python validation
- **FR-1.4:** System shall maintain conversation history in Python data structures
- **FR-1.5:** System shall support multiple languages (English primary, Hindi secondary)

#### 3.1.3 Input/Output

- **Input:** Text string or audio array from Python audio libraries
- **Output:** Processed query string

---

### 3.2 Knowledge Processing Layer

**Priority:** High

#### 3.2.1 Description

Core intelligence module that retrieves, processes, and refines educational content using Python ML libraries.

#### 3.2.2 Functional Requirements

- **FR-2.1:** System shall implement RAG pipeline using LangChain and Python
- **FR-2.2:** System shall maintain knowledge base using FAISS/Chroma vector stores
- **FR-2.3:** System shall refine raw AI responses using custom Python modules
- **FR-2.4:** System shall adapt explanation complexity based on user level
- **FR-2.5:** System shall generate relevant examples and analogies
- **FR-2.6:** System shall create step-by-step explanations
- **FR-2.7:** System shall fact-check responses for accuracy

#### 3.2.3 Input/Output

- **Input:** Student query string
- **Output:** Structured teaching response with metadata

---

### 3.3 Voice Cloning Module

**Priority:** High

#### 3.3.1 Description

Converts text responses to speech using personalized voice cloning with Python TTS libraries.

#### 3.3.2 Functional Requirements

- **FR-3.1:** System shall clone user voice using Coqui TTS (Python)
- **FR-3.2:** System shall generate natural-sounding speech from text
- **FR-3.3:** System shall support voice emotion modulation
- **FR-3.4:** System shall maintain voice consistency across sessions
- **FR-3.5:** System shall optimize audio quality using Python audio processing

#### 3.3.3 Input/Output

- **Input:** Refined text response + voice model
- **Output:** Audio array/file processed by Python libraries

---

### 3.4 Avatar Animation Module

**Priority:** Medium

#### 3.4.1 Description

Provides visual representation through animated avatar using Python computer vision libraries.

#### 3.4.2 Functional Requirements

- **FR-4.1:** System shall display 2D animated avatar using OpenCV
- **FR-4.2:** System shall synchronize lip movements using Wav2Lip (Python)
- **FR-4.3:** System shall support avatar customization
- **FR-4.4:** System shall animate gestures and expressions during speech
- **FR-4.5:** System shall maintain smooth animation using Python video processing

#### 3.4.3 Input/Output

- **Input:** Audio array + avatar model
- **Output:** Video stream via OpenCV/Python

---

### 3.5 Interactive Learning Module

**Priority:** Medium

#### 3.5.1 Description

Enables interactive educational features beyond basic Q&A using Python logic.

#### 3.5.2 Functional Requirements

- **FR-5.1:** System shall generate quiz questions on explained topics
- **FR-5.2:** System shall provide immediate feedback on quiz answers
- **FR-5.3:** System shall track learning progress using Python data structures
- **FR-5.4:** System shall suggest related topics for further learning
- **FR-5.5:** System shall support follow-up questions and clarifications

---

## 4. External Interface Requirements

### 4.1 User Interface Requirements

- **UI-1:** Clean web interface using Streamlit components or desktop UI with PyQt6
- **UI-2:** Avatar display area using OpenCV video streaming
- **UI-3:** Voice recording controls using Python audio libraries (sounddevice/pyaudio)
- **UI-4:** Progress indicators using Streamlit progress bars or PyQt widgets
- **UI-5:** Responsive design using Streamlit responsive components or PyQt layouts

### 4.2 Hardware Interfaces

- **HI-1:** Microphone access via Python audio libraries (sounddevice/pyaudio)
- **HI-2:** Speaker/headphone output via Python audio libraries
- **HI-3:** Camera (optional) using OpenCV for future gesture recognition

### 4.3 Software Interfaces

- **SI-1:** LLM API integration using Python HTTP libraries (requests/httpx)
- **SI-2:** Voice cloning via Coqui TTS (Python library)
- **SI-3:** Avatar animation using Wav2Lip (Python implementation)
- **SI-4:** Vector database using FAISS (Python) or Chroma
- **SI-5:** Speech-to-Text using Whisper (Python library)

### 4.4 Communication Interfaces

- **CI-1:** HTTPS protocol using Python requests library
- **CI-2:** WebSocket connections via Streamlit or Flask-SocketIO
- **CI-3:** REST API endpoints using FastAPI (Python)

---

## 5. System Requirements

### 5.1 Performance Requirements

- **PR-1:** Query processing time < 3 seconds using optimized Python code
- **PR-2:** Voice generation time < 5 seconds for 100-word responses
- **PR-3:** Avatar animation rendering using OpenCV optimization
- **PR-4:** System shall support 10 concurrent users via Python async
- **PR-5:** Knowledge base search latency < 1 second using FAISS

### 5.2 Safety Requirements

- **SR-1:** Input validation using Python validation libraries
- **SR-2:** Content filtering using Python NLP libraries
- **SR-3:** Voice sample privacy protection using Python encryption
- **SR-4:** Safe handling of user content using Python security practices

### 5.3 Security Requirements

- **SEC-1:** User authentication using Python frameworks (Flask-Login/Streamlit-Auth)
- **SEC-2:** Encrypted storage using Python cryptography libraries
- **SEC-3:** API key protection using Python environment variables
- **SEC-4:** HTTPS enforcement in Python web frameworks
- **SEC-5:** Regular security updates for Python dependencies

---

## 6. Non-Functional Requirements

### 6.1 Reliability

- **REL-1:** System uptime of 95% during development/demo phase
- **REL-2:** Graceful error handling with Python exception handling
- **REL-3:** Data backup using Python file operations

### 6.2 Availability

- **AVL-1:** 24/7 availability during demo periods
- **AVL-2:** Scheduled maintenance windows with advance notice

### 6.3 Maintainability

- **MAI-1:** Modular Python architecture for easy component updates
- **MAI-2:** Comprehensive logging using Python logging module
- **MAI-3:** Clear documentation and Python docstrings

### 6.4 Portability

- **POR-1:** Containerized deployment using Docker with Python base images
- **POR-2:** Cross-platform Python compatibility (Windows/Linux/macOS)
- **POR-3:** Virtual environment support using venv/conda

---

## 7. Technical Architecture

### 7.1 System Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Services   │
│  (Streamlit/    │◄──►│   (FastAPI)     │◄──►│   (Python)      │
│   PyQt6)        │    │                 │    │                 │
│                 │    │ • Query Handler │    │ • LLM APIs      │
│ • Chat Interface│    │ • Refinement    │    │ • Coqui TTS     │
│ • Avatar Display│    │ • Voice Synth   │    │ • Wav2Lip       │
│ • Voice Controls│    │ • Session Mgmt  │    │ • Whisper STT   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                    ┌─────────────────┐
                    │   Data Layer    │
                    │   (Python)      │
                    │                 │
                    │ • FAISS/Chroma  │
                    │ • SQLite/Postgres│
                    │ • File Storage  │
                    └─────────────────┘
```

### 7.2 Technology Stack

#### 7.2.1 Frontend Technologies (Python-Based)

- **Web Framework:** Streamlit (recommended) or Flask with Jinja2 templates
- **Desktop Alternative:** PyQt6/PySide6 for native desktop application
- **Styling:** Streamlit built-in themes or Bootstrap CSS (for Flask)
- **Audio Handling:** sounddevice, pyaudio for voice recording and playback
- **Video Display:** OpenCV (cv2) for avatar video streaming
- **Real-time Updates:** Streamlit auto-refresh or Flask-SocketIO

#### 7.2.2 Backend Technologies

- **Framework:** FastAPI (Python)
- **Database:** SQLite (development) / PostgreSQL (production)
- **Vector Store:** FAISS or Chroma (Python libraries)
- **Caching:** Python dictionaries or Redis (optional)
- **Task Queue:** Python asyncio or Celery

#### 7.2.3 AI/ML Technologies (Python Libraries)

- **LLM:** Transformers library (Hugging Face) or API calls via requests
- **Voice Cloning:** Coqui TTS (Python library)
- **Avatar Animation:** Wav2Lip (Python implementation)
- **STT:** OpenAI Whisper (Python library)
- **Vector Search:** FAISS or LangChain (Python)
- **NLP Processing:** spaCy, NLTK (Python libraries)

#### 7.2.4 Infrastructure

- **Containerization:** Docker with Python base images
- **Environment Management:** venv, conda, or poetry
- **Monitoring:** Python logging + optional Prometheus
- **CI/CD:** GitHub Actions with Python workflows

### 7.3 Python Package Dependencies

```python
# Core ML/AI
torch>=2.0.0
transformers>=4.30.0
langchain>=0.0.200
faiss-cpu>=1.7.0
openai-whisper>=20230314

# Voice & Audio
coqui-tts>=0.13.0
sounddevice>=0.4.0
pyaudio>=0.2.11
librosa>=0.10.0

# Computer Vision
opencv-python>=4.7.0
mediapipe>=0.10.0
pillow>=9.0.0

# Web Framework
streamlit>=1.25.0
# OR
flask>=2.3.0
flask-socketio>=5.3.0

# Backend
fastapi>=0.100.0
uvicorn>=0.22.0
sqlalchemy>=2.0.0
pydantic>=2.0.0

# Utilities
numpy>=1.24.0
pandas>=2.0.0
requests>=2.31.0
python-dotenv>=1.0.0
```

---

## 8. Implementation Timeline

### Phase 1: Foundation Setup (Weeks 1-4)

- **Week 1-2:** Python environment setup, dependency installation, basic project structure
- **Week 3-4:** Basic RAG pipeline using LangChain and FAISS (Python)
- **Deliverable:** Working query-response system in Python

### Phase 2: Core Intelligence (Weeks 5-8)

- **Week 5-6:** Response refinement module in Python
- **Week 7-8:** LLM integration using Python libraries/APIs
- **Deliverable:** Intelligent teaching-style responses

### Phase 3: Voice Integration (Weeks 9-12)

- **Week 9-10:** Coqui TTS voice cloning implementation
- **Week 11-12:** Audio processing pipeline with Python libraries
- **Deliverable:** System generates speech in cloned voice

### Phase 4: Avatar Development (Weeks 13-16)

- **Week 13-14:** Wav2Lip integration for avatar animation
- **Week 15-16:** OpenCV video streaming and lip-sync
- **Deliverable:** Complete avatar with synchronized speech

### Phase 5: Python Frontend Development (Weeks 17-20)

- **Week 17-18:** Streamlit interface development with chat, audio, and video components
- **Week 19-20:** Full system integration and Python-to-Python communication
- **Deliverable:** Complete working Python application

### Phase 6: Testing & Documentation (Weeks 21-24)

- **Week 21-22:** Comprehensive testing and Python-specific optimizations
- **Week 23-24:** Documentation, demo preparation, deployment
- **Deliverable:** Production-ready Python system with documentation

---

## Key Advantages of All-Python Stack

### 💡 Development Benefits

- **Single Language:** Both team members work in Python
- **Rapid Prototyping:** Streamlit enables quick UI development
- **Rich ML Ecosystem:** Access to all major AI/ML Python libraries
- **Easy Debugging:** Consistent debugging tools across the entire stack

### 🚀 Technical Benefits

- **Memory Efficiency:** Shared objects between frontend and backend
- **Performance:** Direct Python library calls without API overhead
- **Integration:** Seamless data flow between components
- **Deployment:** Single Python container deployment

---

## Updated Team Division

- **Person A (NLP Engineer):** RAG pipeline, refinement module, knowledge base, FastAPI backend
- **Person B (Voice/Avatar Engineer):** Voice cloning (Coqui TTS), avatar animation (Wav2Lip), Streamlit frontend

---

## Appendices

### A. Glossary

- **RAG:** Retrieval-Augmented Generation
- **TTS:** Text-to-Speech (Coqui TTS)
- **STT:** Speech-to-Text (Whisper)
- **FAISS:** Facebook AI Similarity Search
- **OpenCV:** Open Source Computer Vision Library

### B. Risk Assessment

- **High Risk:** Voice cloning quality with Coqui TTS
- **Medium Risk:** Real-time avatar rendering with OpenCV
- **Low Risk:** Basic Streamlit interface development

### C. Success Metrics

- **Technical:** <3s response time, >90% voice similarity, smooth video at 30fps
- **User Experience:** Intuitive Streamlit interface, natural interactions
- **Academic:** Successful project demonstration and evaluation

---

**Document End**

*This SRS is a living document and will be updated as the project evolves.*
