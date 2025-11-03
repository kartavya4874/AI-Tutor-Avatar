# AI Virtual Teacher - Complete Implementation Guide
## Self-Contained Development Roadmap

**Version:** 2.0  
**Last Updated:** November 03, 2025  
**Critical:** This document is the ONLY resource for building this project. Read completely before starting.

---

## 📋 DOCUMENT USAGE INSTRUCTIONS

### How to Use This Document

1. **Start Fresh Each Time**: When resuming work, read from the beginning to understand context
2. **Follow Sequentially**: Complete checkpoints in exact order - dependencies are critical
3. **Update Progress**: Mark checkpoints as you complete them
4. **Test Everything**: Each checkpoint has explicit testing instructions
5. **Document Issues**: Add notes in the "Issues Encountered" section
6. **Integration Focus**: Pay special attention to "Integration Points" in each checkpoint

### Status Legend
- ⬜ **Not Started**: Checkpoint not begun
- 🔄 **In Progress**: Currently working on this
- ✅ **Completed**: Checkpoint finished and tested
- ⚠️ **Blocked**: Cannot proceed due to dependency/issue
- 🔧 **Needs Revision**: Completed but requires fixes

---

## 🎯 PROJECT OVERVIEW

### What We're Building
An AI-powered virtual teacher system that:
1. Accepts text or voice queries from students
2. Retrieves relevant knowledge using RAG (Retrieval-Augmented Generation)
3. Generates educational responses using LLM
4. Refines responses for teaching clarity
5. Converts text to speech using cloned voice (Coqui TTS)
6. Animates an avatar with lip-sync (Wav2Lip)
7. Displays everything in a Streamlit web interface

### System Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   FastAPI    │────▶│ LLM Service │
│  Frontend   │◀────│   Backend    │◀────│  (OpenAI)   │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
             ┌──────▼────┐  ┌─────▼──────┐
             │   FAISS   │  │ Coqui TTS  │
             │  Vector   │  │   Voice    │
             │   Store   │  │  Cloning   │
             └───────────┘  └────────────┘
                                  │
                            ┌─────▼──────┐
                            │  Wav2Lip   │
                            │   Avatar   │
                            │ Animation  │
                            └────────────┘
```

### Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **LLM**: OpenAI GPT-4 (or GPT-3.5-turbo)
- **Vector Store**: FAISS
- **TTS**: Coqui TTS
- **Avatar**: Wav2Lip + OpenCV
- **STT**: OpenAI Whisper
- **Database**: SQLite (session storage)

---

## 📊 MASTER PROGRESS TRACKER

| Phase | Checkpoint | Status | Date Started | Date Completed | Blocker |
|-------|-----------|--------|--------------|----------------|---------|
| **PHASE 0: Pre-Setup** | | | | | |
| 0.1 | System Requirements Check | ⬜ | | | |
| 0.2 | API Keys & Accounts Setup | ⬜ | | | |
| **PHASE 1: Foundation** | | | | | |
| 1.1 | Python Environment Setup | ⬜ | | | |
| 1.2 | Project Structure Creation | ⬜ | | | |
| 1.3 | Core Dependencies Installation | ⬜ | | | |
| 1.4 | Basic Configuration Setup | ⬜ | | | |
| **PHASE 2: Knowledge Base** | | | | | |
| 2.1 | Document Ingestion System | ⬜ | | | |
| 2.2 | FAISS Vector Store Setup | ⬜ | | | |
| 2.3 | Retrieval Pipeline Testing | ⬜ | | | |
| 2.4 | Knowledge Base API Endpoints | ⬜ | | | |
| **PHASE 3: LLM Integration** | | | | | |
| 3.1 | OpenAI API Integration | ⬜ | | | |
| 3.2 | RAG Pipeline Implementation | ⬜ | | | |
| 3.3 | Response Refinement Module | ⬜ | | | |
| 3.4 | Query Processing API | ⬜ | | | |
| **PHASE 4: Voice Cloning** | | | | | |
| 4.1 | Coqui TTS Installation | ⬜ | | | |
| 4.2 | Voice Sample Recording | ⬜ | | | |
| 4.3 | Voice Cloning Training | ⬜ | | | |
| 4.4 | TTS API Endpoints | ⬜ | | | |
| **PHASE 5: Avatar Animation** | | | | | |
| 5.1 | Wav2Lip Setup | ⬜ | | | |
| 5.2 | Avatar Preparation | ⬜ | | | |
| 5.3 | Lip-Sync Generation | ⬜ | | | |
| 5.4 | Avatar API Endpoints | ⬜ | | | |
| **PHASE 6: Frontend Development** | | | | | |
| 6.1 | Basic Streamlit Interface | ⬜ | | | |
| 6.2 | Chat Component | ⬜ | | | |
| 6.3 | Voice Recording Component | ⬜ | | | |
| 6.4 | Avatar Display Component | ⬜ | | | |
| **PHASE 7: Integration** | | | | | |
| 7.1 | Frontend-Backend Connection | ⬜ | | | |
| 7.2 | Complete Flow Testing | ⬜ | | | |
| 7.3 | Error Handling & Polish | ⬜ | | | |
| **PHASE 8: Finalization** | | | | | |
| 8.1 | Performance Optimization | ⬜ | | | |
| 8.2 | Documentation | ⬜ | | | |
| 8.3 | Demo Preparation | ⬜ | | | |

**Overall Completion: 0/38 Checkpoints (0%)**

---

## 🚀 PHASE 0: PRE-SETUP (Week 0 - Before Coding)

### ⬜ CHECKPOINT 0.1: System Requirements Check

#### Purpose
Verify that your development machine meets minimum requirements before starting.

#### Requirements Checklist
- [ ] **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 20.04+
- [ ] **Python**: Version 3.10 or 3.11 (NOT 3.12 - compatibility issues)
- [ ] **RAM**: Minimum 16GB (32GB recommended)
- [ ] **Storage**: 50GB free space
- [ ] **GPU**: NVIDIA GPU with 6GB+ VRAM (optional but recommended for Wav2Lip)
- [ ] **Internet**: Stable connection for API calls and model downloads

#### Verification Commands
```bash
# Check Python version
python --version
# Output should be: Python 3.10.x or Python 3.11.x

# Check available RAM (Linux/Mac)
free -h

# Check available storage
df -h

# Check GPU (if available)
nvidia-smi
```

#### Success Criteria
- [ ] All requirements met or noted for workarounds
- [ ] Python version confirmed compatible

#### Next Step
Proceed to Checkpoint 0.2: API Keys & Accounts Setup

---

### ⬜ CHECKPOINT 0.2: API Keys & Accounts Setup

#### Purpose
Obtain all necessary API keys and create accounts before development.

#### Required Accounts & Keys

##### 1. OpenAI API Key
- **URL**: https://platform.openai.com/signup
- **Steps**:
  1. Create account
  2. Go to https://platform.openai.com/api-keys
  3. Click "Create new secret key"
  4. Copy and save key (starts with `sk-`)
  5. Add $5-10 credits to account
- **Save As**: `OPENAI_API_KEY=sk-xxx...`

##### 2. Hugging Face Token (Optional)
- **URL**: https://huggingface.co/join
- **Steps**:
  1. Create account
  2. Go to Settings → Access Tokens
  3. Create new token with read access
  4. Copy and save token
- **Save As**: `HF_TOKEN=hf_xxx...`

##### 3. Git Repository
- **URL**: https://github.com (or GitLab/Bitbucket)
- **Steps**:
  1. Create new repository: "ai-virtual-teacher"
  2. Initialize as private repository
  3. Copy repository URL
- **Save As**: `REPO_URL=https://github.com/yourusername/ai-virtual-teacher.git`

#### API Keys Checklist
- [ ] OpenAI API key obtained and tested
- [ ] Hugging Face token obtained (optional)
- [ ] Git repository created
- [ ] Keys stored securely (NOT in code)

#### Testing API Keys
```bash
# Test OpenAI API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY"

# Should return list of available models
```

#### Success Criteria
- [ ] All API keys obtained
- [ ] Keys tested and working
- [ ] Keys securely stored for next steps

#### Next Step
Proceed to Checkpoint 1.1: Python Environment Setup

---

## 🔧 PHASE 1: FOUNDATION SETUP (Week 1)

### ⬜ CHECKPOINT 1.1: Python Environment Setup

#### Purpose
Create an isolated Python environment with correct version and tools.

#### Step-by-Step Instructions

##### Step 1: Install Python 3.10/3.11
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# macOS (using Homebrew)
brew install python@3.10

# Windows - Download from python.org
# https://www.python.org/downloads/release/python-3100/
```

##### Step 2: Create Project Directory
```bash
# Create main project folder
mkdir ai-virtual-teacher
cd ai-virtual-teacher

# Verify you're in the right directory
pwd
# Should output: /path/to/ai-virtual-teacher
```

##### Step 3: Create Virtual Environment
```bash
# Create virtual environment named 'venv'
python3.10 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show (venv) at start of prompt)
which python
# Should output: /path/to/ai-virtual-teacher/venv/bin/python
```

##### Step 4: Upgrade pip and Install Basic Tools
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install development tools
pip install ipython pytest black flake8
```

##### Step 5: Initialize Git Repository
```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.env

# IDE
.vscode/
.idea/
*.swp

# Data
data/knowledge_base/*
data/voice_models/*
data/avatar_models/*
*.db
*.sqlite

# Temporary files
*.log
*.wav
*.mp4
*.avi
temp_*

# Models (large files)
checkpoints/
models/
*.pth
*.pt
*.ckpt
EOF

# Make first commit
git add .gitignore
git commit -m "Initial commit: .gitignore"
```

#### Verification Checklist
- [ ] Python 3.10/3.11 installed
- [ ] Virtual environment created and activated
- [ ] Pip upgraded to latest version
- [ ] Git initialized
- [ ] .gitignore created

#### Testing Commands
```bash
# Test Python version
python --version

# Test pip
pip --version

# Test git
git status
```

#### Success Criteria
- [ ] Virtual environment active (venv) in prompt
- [ ] Python 3.10/3.11 confirmed
- [ ] Git repository initialized

#### Issues Encountered
```
[Document issues here as you encounter them]
```

#### Next Step
Proceed to Checkpoint 1.2: Project Structure Creation

---

### ⬜ CHECKPOINT 1.2: Project Structure Creation

#### Purpose
Create the complete directory structure for the entire project.

#### Complete Directory Structure
```
ai-virtual-teacher/
├── backend/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application entry point
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── query.py            # Query processing endpoints
│   │       ├── voice.py            # TTS endpoints
│   │       └── avatar.py           # Avatar generation endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py       # FAISS vector store manager
│   │   ├── rag_pipeline.py         # RAG implementation
│   │   └── refinement.py           # Response refinement logic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py          # OpenAI API wrapper
│   │   ├── tts_service.py          # Coqui TTS wrapper
│   │   ├── stt_service.py          # Whisper STT wrapper
│   │   └── avatar_service.py       # Wav2Lip integration
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic models
│   └── utils/
│       ├── __init__.py
│       ├── logger.py               # Logging configuration
│       └── config.py               # Configuration loader
├── frontend/
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit application
│   ├── components/
│   │   ├── __init__.py
│   │   ├── chat_interface.py      # Chat UI component
│   │   ├── voice_input.py         # Voice recording component
│   │   └── avatar_display.py      # Avatar video player
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py         # Audio processing utilities
│       └── api_client.py          # Backend API client
├── data/
│   ├── knowledge_base/            # Educational documents (PDFs, TXT)
│   ├── voice_models/              # Cloned voice models
│   └── avatar_models/             # Avatar images/videos
├── tests/
│   ├── __init__.py
│   ├── test_knowledge_base.py
│   ├── test_rag_pipeline.py
│   ├── test_llm_service.py
│   ├── test_tts_service.py
│   └── test_avatar_service.py
├── scripts/
│   ├── setup_knowledge_base.py    # Script to populate knowledge base
│   ├── train_voice_model.py      # Script to train voice cloning
│   └── prepare_avatar.py         # Script to prepare avatar assets
├── config/
│   ├── __init__.py
│   └── settings.yaml              # Configuration file
├── logs/                          # Application logs
├── temp/                          # Temporary files
├── .env.example                   # Example environment variables
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── docker-compose.yml             # Docker configuration (optional)
```

#### Step-by-Step Creation

##### Step 1: Create All Directories
```bash
# Navigate to project root
cd ai-virtual-teacher

# Create backend structure
mkdir -p backend/api/routes
mkdir -p backend/core
mkdir -p backend/services
mkdir -p backend/models
mkdir -p backend/utils

# Create frontend structure
mkdir -p frontend/components
mkdir -p frontend/utils

# Create data directories
mkdir -p data/knowledge_base
mkdir -p data/voice_models
mkdir -p data/avatar_models

# Create other directories
mkdir -p tests
mkdir -p scripts
mkdir -p config
mkdir -p logs
mkdir -p temp
```

##### Step 2: Create All __init__.py Files
```bash
# Backend __init__.py files
touch backend/__init__.py
touch backend/api/__init__.py
touch backend/api/routes/__init__.py
touch backend/core/__init__.py
touch backend/services/__init__.py
touch backend/models/__init__.py
touch backend/utils/__init__.py

# Frontend __init__.py files
touch frontend/__init__.py
touch frontend/components/__init__.py
touch frontend/utils/__init__.py

# Tests __init__.py
touch tests/__init__.py

# Config __init__.py
touch config/__init__.py
```

##### Step 3: Create Placeholder Python Files
```bash
# Backend files
touch backend/api/main.py
touch backend/api/routes/query.py
touch backend/api/routes/voice.py
touch backend/api/routes/avatar.py
touch backend/core/knowledge_base.py
touch backend/core/rag_pipeline.py
touch backend/core/refinement.py
touch backend/services/llm_service.py
touch backend/services/tts_service.py
touch backend/services/stt_service.py
touch backend/services/avatar_service.py
touch backend/models/schemas.py
touch backend/utils/logger.py
touch backend/utils/config.py

# Frontend files
touch frontend/app.py
touch frontend/components/chat_interface.py
touch frontend/components/voice_input.py
touch frontend/components/avatar_display.py
touch frontend/utils/audio_utils.py
touch frontend/utils/api_client.py

# Test files
touch tests/test_knowledge_base.py
touch tests/test_rag_pipeline.py
touch tests/test_llm_service.py
touch tests/test_tts_service.py
touch tests/test_avatar_service.py

# Script files
touch scripts/setup_knowledge_base.py
touch scripts/train_voice_model.py
touch scripts/prepare_avatar.py
```

##### Step 4: Create Configuration Files

**Create .env.example**:
```bash
cat > .env.example << 'EOF'
# OpenAI API
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4

# Hugging Face
HF_TOKEN=hf_your_huggingface_token_here

# Database
DATABASE_URL=sqlite:///./virtual_teacher.db

# Paths
KNOWLEDGE_BASE_PATH=./data/knowledge_base
VOICE_MODEL_PATH=./data/voice_models
AVATAR_MODEL_PATH=./data/avatar_models
TEMP_PATH=./temp
LOG_PATH=./logs

# API Settings
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=8501

# Model Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TTS_MODEL=tts_models/en/ljspeech/tacotron2-DDC
AVATAR_CHECKPOINT=checkpoints/wav2lip_gan.pth

# Performance
MAX_TOKENS=2000
TEMPERATURE=0.7
TOP_K_RESULTS=3
EOF
```

**Create .env (copy from .env.example)**:
```bash
cp .env.example .env
# Now edit .env and add your actual API keys
```

**Create config/settings.yaml**:
```bash
cat > config/settings.yaml << 'EOF'
app:
  name: "AI Virtual Teacher"
  version: "1.0.0"
  debug: true

knowledge_base:
  chunk_size: 1000
  chunk_overlap: 200
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 2000

tts:
  model: "tts_models/en/ljspeech/tacotron2-DDC"
  sample_rate: 22050
  
avatar:
  fps: 25
  resolution: [512, 512]
  
api:
  cors_origins:
    - "http://localhost:8501"
    - "http://127.0.0.1:8501"
EOF
```

**Create README.md**:
```bash
cat > README.md << 'EOF'
# AI Virtual Teacher

An AI-powered virtual teacher with personalized avatar and voice cloning.

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables: Copy `.env.example` to `.env` and add API keys
3. Run backend: `python -m uvicorn backend.api.main:app --reload`
4. Run frontend: `streamlit run frontend/app.py`

## Project Status

See IMPLEMENTATION_GUIDE.md for detailed progress tracking.

## Documentation

- [Implementation Guide](IMPLEMENTATION_GUIDE.md) - Step-by-step development guide
- [API Documentation](http://localhost:8000/docs) - Auto-generated API docs
EOF
```

##### Step 5: Verify Structure
```bash
# Display directory tree
tree -I 'venv|__pycache__|*.pyc' -L 3

# Or use find if tree is not available
find . -type d -not -path '*/venv/*' -not -path '*/__pycache__/*' | sort
```

#### Verification Checklist
- [ ] All directories created
- [ ] All __init__.py files present
- [ ] All placeholder Python files created
- [ ] .env.example created
- [ ] .env created with API keys
- [ ] config/settings.yaml created
- [ ] README.md created

#### Success Criteria
- [ ] Directory structure matches specification exactly
- [ ] Can navigate to any directory
- [ ] All configuration files in place

#### Integration Point
This structure will be used throughout development. Each module file will be populated in subsequent checkpoints.

#### Issues Encountered
```
[Document issues here]
```

#### Next Step
Proceed to Checkpoint 1.3: Core Dependencies Installation

---

### ⬜ CHECKPOINT 1.3: Core Dependencies Installation

#### Purpose
Install all Python packages required for the entire project in the correct order.

#### Dependencies Overview

The project requires packages in 5 categories:
1. **Core ML/AI**: PyTorch, Transformers, LangChain
2. **Voice & Audio**: Coqui TTS, audio processing libraries
3. **Computer Vision**: OpenCV, video processing
4. **Web Framework**: Streamlit, FastAPI
5. **Utilities**: Database, logging, testing

#### Step-by-Step Installation

##### Step 1: Create requirements.txt

```bash
cat > requirements.txt << 'EOF'
# Core ML/AI
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
transformers==4.30.2
langchain==0.0.267
sentence-transformers==2.2.2
faiss-cpu==1.7.4
openai==0.28.0
tiktoken==0.4.0

# Voice & Audio
TTS==0.15.6
sounddevice==0.4.6
pyaudio==0.2.13
librosa==0.10.0.post2
scipy==1.11.2
pydub==0.25.1
noisereduce==2.0.1

# Computer Vision & Avatar
opencv-python==4.8.0.76
opencv-contrib-python==4.8.0.76
Pillow==10.0.0
imageio==2.31.3
imageio-ffmpeg==0.4.9
mediapipe==0.10.3

# Web Framework
streamlit==1.26.0
fastapi==0.103.0
uvicorn[standard]==0.23.2
python-multipart==0.0.6
aiofiles==23.2.1
websockets==11.0.3

# Database & Storage
sqlalchemy==2.0.20
alembic==1.12.0

# Utilities
python-dotenv==1.0.0
pydantic==2.3.0
pydantic-settings==2.0.3
pyyaml==6.0.1
requests==2.31.0
httpx==0.24.1
aiohttp==3.8.5

# Testing & Development
pytest==7.4.2
pytest-asyncio==0.21.1
pytest-cov==4.1.0
black==23.9.1
flake8==6.1.0
mypy==1.5.1

# Logging & Monitoring
loguru==0.7.0
python-json-logger==2.0.7

# Additional utilities
numpy==1.24.3
pandas==2.0.3
tqdm==4.66.1
colorama==0.4.6
EOF
```

##### Step 2: Install PyTorch First (Important!)

PyTorch must be installed before other packages to ensure correct CUDA version.

**For CPU-only (most students)**:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

**For NVIDIA GPU (if available)**:
```bash
# CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

##### Step 3: Install Audio Dependencies (Platform-Specific)

**On Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-pyaudio ffmpeg
```

**On macOS**:
```bash
brew install portaudio ffmpeg
```

**On Windows**:
```bash
# Download PyAudio wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
# Then install: pip install PyAudio‑0.2.11‑cp310‑cp310‑win_amd64.whl
```

##### Step 4: Install All Other Dependencies
```bash
# Install from requirements.txt (skip torch as already installed)
pip install -r requirements.txt --no-deps torch torchvision torchaudio

# Or install all at once if torch installation was successful
pip install -r requirements.txt
```

##### Step 5: Install Whisper for STT
```bash
pip install openai-whisper==20230314
```

##### Step 6: Download Wav2Lip Repository

```bash
# Clone Wav2Lip in project root
cd ai-virtual-teacher
git clone https://github.com/Rudrabha/Wav2Lip.git

# Download pre-trained model
cd Wav2Lip
mkdir -p checkpoints

# Download model (use wget or curl)
wget "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZdQ7g_7udxPZLxGKQ?download=1" -O "checkpoints/wav2lip_gan.pth"

cd ..
```

##### Step 7: Verify All Installations

```bash
# Create verification script
cat > scripts/verify_installation.py << 'EOF'
"""
Verification script to check all dependencies are installed correctly.
"""

import sys

def check_import(package_name, import_name=None):
    """Try to import a package and report status."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}: OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: FAILED - {e}")
        return False

def main():
    print("Checking Python dependencies...\n")
    
    packages = [
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("LangChain", "langchain"),
        ("FAISS", "faiss"),
        ("OpenAI", "openai"),
        ("Coqui TTS", "TTS"),
        ("sounddevice", "sounddevice"),
        ("librosa", "librosa"),
        ("OpenCV", "cv2"),
        ("Pillow", "PIL"),
        ("Streamlit", "streamlit"),
        ("FastAPI", "fastapi"),
        ("Uvicorn", "uvicorn"),
        ("SQLAlchemy", "sqlalchemy"),
        ("Pydantic", "pydantic"),
        ("python-dotenv", "dotenv"),
        ("pytest", "pytest"),
        ("Whisper", "whisper"),
    ]
    
    results = []
    for package_name, import_name in packages:
        results.append(check_import(package_name, import_name))
    
    print(f"\n{'='*50}")
    print(f"Total: {sum(results)}/{len(results)} packages installed successfully")
    
    if all(results):
        print("✅ All dependencies installed correctly!")
        return 0
    else:
        print("❌ Some dependencies failed to install. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Run verification
python scripts/verify_installation.py
```

#### Common Installation Issues & Solutions

**Issue 1: PyAudio fails on Windows**
```
Solution: Download pre-compiled wheel from 
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

**Issue 2: FAISS installation fails**
```bash
Solution: Use conda instead
conda install -c pytorch faiss-cpu
```

**Issue 3: Coqui TTS fails**
```bash
Solution: Install specific version
pip install TTS==0.15.6 --no-deps
pip install numpy scipy librosa soundfile
```

**Issue 4: OpenCV issues**
```bash
Solution: Reinstall
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.76
```

#### Verification Checklist
- [ ] PyTorch installed and CUDA detected (if GPU available)
- [ ] All ML/AI packages installed
- [ ] Audio packages installed
- [ ] Computer vision packages installed
- [ ] Web framework packages installed
- [ ] Wav2Lip repository cloned
- [ ] Wav2Lip checkpoint downloaded
- [ ] Verification script runs successfully

#### Testing Commands
```bash
# Test PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test key packages
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import langchain; print(f'LangChain: {langchain.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "from TTS.api import TTS; print('Coqui TTS: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Run full verification
python scripts/verify_installation.py
```

#### Success Criteria
- [ ] All packages import without errors
- [ ] Verification script shows 100% success
- [ ] No dependency conflicts
- [ ] Wav2Lip checkpoint file exists (size ~150MB)

#### Package Size Warning
Total installation size: ~5-8 GB
Time required: 15-30 minutes depending on internet speed

#### Issues Encountered
```
[Document specific installation issues here]
```

#### Integration Point
These packages will be imported in subsequent modules:
- **knowledge_base.py**: langchain, faiss, sentence-transformers
- **llm_service.py**: openai, transformers
- **tts_service.py**: TTS
- **avatar_service.py**: cv2, Wav2Lip
- **app.py**: streamlit

#### Next Step
Proceed to Checkpoint 1.4: Basic Configuration Setup

---

### ⬜ CHECKPOINT 1.4: Basic Configuration Setup

#### Purpose
Set up configuration management, logging, and environment loading that will be used by all modules.

#### Step-by-Step Implementation

##### Step 1: Implement Configuration Loader

**File: `backend/utils/config.py`**
```python
"""
Configuration management for the Virtual Teacher system.
Loads settings from .env file and settings.yaml
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for accessing all settings."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent
        self.settings = self._load_yaml_settings()
        
    def _load_yaml_settings(self) -> Dict[str, Any]:
        """Load settings from YAML file."""
        config_path = self.base_dir / "config" / "settings.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    # API Keys
    @property
    def openai_api_key(self) -> str:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return key
    
    @property
    def hf_token(self) -> str:
        return os.getenv("HF_TOKEN", "")
    
    # Database
    @property
    def database_url(self) -> str:
        return os.getenv("DATABASE_URL", "sqlite:///./virtual_teacher.db")
    
    # Paths
    @property
    def knowledge_base_path(self) -> Path:
        path = Path(os.getenv("KNOWLEDGE_BASE_PATH", "./data/knowledge_base"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def voice_model_path(self) -> Path:
        path = Path(os.getenv("VOICE_MODEL_PATH", "./data/voice_models"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def avatar_model_path(self) -> Path:
        path = Path(os.getenv("AVATAR_MODEL_PATH", "./data/avatar_models"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def temp_path(self) -> Path:
        path = Path(os.getenv("TEMP_PATH", "./temp"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def log_path(self) -> Path:
        path = Path(os.getenv("LOG_PATH", "./logs"))
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # API Settings
    @property
    def backend_host(self) -> str:
        return os.getenv("BACKEND_HOST", "0.0.0.0")
    
    @property
    def backend_port(self) -> int:
        return int(os.getenv("BACKEND_PORT", "8000"))
    
    @property
    def frontend_port(self) -> int:
        return int(os.getenv("FRONTEND_PORT", "8501"))
    
    # Model Settings
    @property
    def embedding_model(self) -> str:
        return self.settings.get("knowledge_base", {}).get(
            "embedding_model", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    
    @property
    def openai_model(self) -> str:
        return self.settings.get("llm", {}).get("model", "gpt-4")
    
    @property
    def llm_temperature(self) -> float:
        return float(self.settings.get("llm", {}).get("temperature", 0.7))
    
    @property
    def llm_max_tokens(self) -> int:
        return int(self.settings.get("llm", {}).get("max_tokens", 2000))
    
    @property
    def tts_model(self) -> str:
        return self.settings.get("tts", {}).get(
            "model", 
            "tts_models/en/ljspeech/tacotron2-DDC"
        )
    
    @property
    def chunk_size(self) -> int:
        return int(self.settings.get("knowledge_base", {}).get("chunk_size", 1000))
    
    @property
    def chunk_overlap(self) -> int:
        return int(self.settings.get("knowledge_base", {}).get("chunk_overlap", 200))
    
    @property
    def top_k_results(self) -> int:
        return int(os.getenv("TOP_K_RESULTS", "3"))
    
    # Wav2Lip Settings
    @property
    def wav2lip_checkpoint(self) -> str:
        return os.getenv(
            "AVATAR_CHECKPOINT", 
            "./Wav2Lip/checkpoints/wav2lip_gan.pth"
        )
    
    @property
    def avatar_fps(self) -> int:
        return int(self.settings.get("avatar", {}).get("fps", 25))
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get any setting by key path (e.g., 'llm.model')."""
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value if value is not None else default

# Global config instance
config = Config()
```

##### Step 2: Implement Logger

**File: `backend/utils/logger.py`**
```python
"""
Logging configuration for the Virtual Teacher system.
Provides consistent logging across all modules.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

from backend.utils.config import config


class Logger:
    """Custom logger with file and console output."""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str, level: Optional[int] = None) -> logging.Logger:
        """
        Get or create a logger with the given name.
        
        Args:
            name: Logger name (usually __name__ of the module)
            level: Logging level (defaults to INFO)
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level or logging.INFO)
        logger.propagate = False
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = config.log_path / f"{name.replace('.', '_')}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


# Convenience function
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return Logger.get_logger(name)
```

##### Step 3: Create Pydantic Models

**File: `backend/models/schemas.py`**
```python
"""
Pydantic models for request/response validation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# Request Models
class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    complexity: str = Field(default="intermediate", description="Response complexity level")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is photosynthesis?",
                "complexity": "beginner"
            }
        }


class TTSRequest(BaseModel):
    """Request model for text-to-speech generation."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice_model: Optional[str] = Field(default=None, description="Voice model ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, I am your virtual teacher.",
                "voice_model": "default"
            }
        }


class AvatarRequest(BaseModel):
    """Request model for avatar generation."""
    audio_path: str = Field(..., description="Path to audio file")
    avatar_id: Optional[str] = Field(default="default", description="Avatar identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_path": "/temp/audio_123.wav",
                "avatar_id": "teacher_01"
            }
        }


# Response Models
class QueryResponse(BaseModel):
    """Response model for query processing."""
    query: str
    response: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None
    sources: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is photosynthesis?",
                "response": "Photosynthesis is the process by which plants...",
                "audio_url": "/audio/response_123.wav",
                "video_url": "/video/avatar_123.mp4",
                "sources": ["Biology Textbook, Chapter 3"],
                "timestamp": "2025-11-03T10:30:00"
            }
        }


class TTSResponse(BaseModel):
    """Response model for TTS generation."""
    audio_url: str
    duration: float
    sample_rate: int = 22050
    
    class Config:
        json_schema_extra = {
            "example": {
                "audio_url": "/audio/tts_123.wav",
                "duration": 5.2,
                "sample_rate": 22050
            }
        }


class AvatarResponse(BaseModel):
    """Response model for avatar generation."""
    video_url: str
    duration: float
    fps: int = 25
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_url": "/video/avatar_123.mp4",
                "duration": 5.2,
                "fps": 25
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid query",
                "detail": "Query cannot be empty",
                "timestamp": "2025-11-03T10:30:00"
            }
        }


# Internal Models
class Document(BaseModel):
    """Document model for knowledge base."""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class RefinedResponse(BaseModel):
    """Refined response structure."""
    introduction: str
    main_explanation: str
    examples: List[str]
    step_by_step: List[str]
    summary: str
    follow_up_questions: List[str]
```

##### Step 4: Test Configuration and Logging

**Create: `tests/test_config.py`**
```python
"""
Test configuration and logging setup.
"""

import pytest
from backend.utils.config import config
from backend.utils.logger import get_logger


def test_config_loads():
    """Test that configuration loads successfully."""
    assert config is not None
    assert config.base_dir.exists()


def test_api_keys():
    """Test that API keys are accessible."""
    try:
        api_key = config.openai_api_key
        assert api_key.startswith("sk-")
    except ValueError:
        pytest.skip("OpenAI API key not set")


def test_paths_created():
    """Test that all required paths are created."""
    assert config.knowledge_base_path.exists()
    assert config.voice_model_path.exists()
    assert config.avatar_model_path.exists()
    assert config.temp_path.exists()
    assert config.log_path.exists()


def test_model_settings():
    """Test that model settings are accessible."""
    assert config.embedding_model is not None
    assert config.openai_model in ["gpt-4", "gpt-3.5-turbo"]
    assert 0 <= config.llm_temperature <= 2
    assert config.llm_max_tokens > 0


def test_logger():
    """Test that logger works."""
    logger = get_logger("test_logger")
    
    logger.info("Test info message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    
    # Check log file created
    log_file = config.log_path / "test_logger.log"
    assert log_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

##### Step 5: Run Configuration Tests

```bash
# Run configuration tests
pytest tests/test_config.py -v

# Expected output:
# tests/test_config.py::test_config_loads PASSED
# tests/test_config.py::test_api_keys PASSED
# tests/test_config.py::test_paths_created PASSED
# tests/test_config.py::test_model_settings PASSED
# tests/test_config.py::test_logger PASSED
```

#### Verification Checklist
- [ ] config.py created and loads without errors
- [ ] logger.py created and produces logs
- [ ] schemas.py created with all models
- [ ] All directories created by config
- [ ] API keys accessible from config
- [ ] Configuration tests pass
- [ ] Log files appear in logs/ directory

#### Testing Commands
```bash
# Test config loading
python -c "from backend.utils.config import config; print(config.openai_model)"

# Test logger
python -c "from backend.utils.logger import get_logger; logger = get_logger('test'); logger.info('Test message')"

# Test schemas
python -c "from backend.models.schemas import QueryRequest; req = QueryRequest(query='test'); print(req.query)"

# Run all tests
pytest tests/test_config.py -v
```

#### Success Criteria
- [ ] Config loads all settings correctly
- [ ] Logger creates log files
- [ ] All Pydantic models validate correctly
- [ ] No import errors
- [ ] Tests pass

#### Integration Point
These utilities will be used by ALL subsequent modules:
- **Every service** will import `config` for settings
- **Every module** will import `get_logger` for logging
- **All API endpoints** will use Pydantic models for validation

#### Issues Encountered
```
[Document configuration issues here]
```

#### Next Step
Proceed to Phase 2: Checkpoint 2.1 - Document Ingestion System

---

## 📚 PHASE 2: KNOWLEDGE BASE (Week 2)

### ⬜ CHECKPOINT 2.1: Document Ingestion System

#### Purpose
Build system to load, process, and chunk educational documents for the knowledge base.

#### What This Module Does
- Loads documents from data/knowledge_base/ directory
- Supports PDF, TXT, MD, DOCX formats
- Splits documents into manageable chunks
- Prepares documents for vector store

#### Step-by-Step Implementation

##### Step 1: Install Additional Dependencies

```bash
# Install document processing libraries
pip install pypdf==3.15.5
pip install python-docx==0.8.11
pip install markdown==3.4.4
pip install beautifulsoup4==4.12.2
```

##### Step 2: Implement Document Ingestion

**File: `backend/core/knowledge_base.py`**
```python
"""
Knowledge Base management with document ingestion and vector storage.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

from backend.utils.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBase:
    """
    Manages educational content storage and retrieval.
    
    This class handles:
    1. Loading documents from various formats
    2. Chunking documents for processing
    3. Creating and managing FAISS vector store
    4. Searching for relevant content
    """
    
    def __init__(self):
        """Initialize knowledge base with embeddings model."""
        logger.info("Initializing Knowledge Base...")
        
        self.data_path = config.knowledge_base_path
        self.vector_store_path = self.data_path / "faiss_index"
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {config.embedding_model}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Vector store (will be loaded/created)
        self.vector_store: Optional[FAISS] = None
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("Knowledge Base initialized")
    
    def load_documents(self, file_types: List[str] = None) -> List[Document]:
        """
        Load documents from the knowledge base directory.
        
        Args:
            file_types: List of file extensions to load (default: all supported)
            
        Returns:
            List of loaded documents
        """
        if file_types is None:
            file_types = ['.txt', '.pdf', '.md', '.docx']
        
        logger.info(f"Loading documents from {self.data_path}")
        documents = []
        
        for file_path in self.data_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_types:
                try:
                    docs = self._load_single_document(file_path)
                    documents.extend(docs)
                    logger.info(f"Loaded: {file_path.name} ({len(docs)} chunks)")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {str(e)}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _load_single_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on its file type.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks
        """
        suffix = file_path.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif suffix == '.docx':
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        
        documents = loader.load()
        
        # Add metadata
        for doc in documents:
            doc.metadata['source'] = file_path.name
            doc.metadata['file_type'] = suffix
        
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.
        
        Args:
            documents: List of document chunks
            
        Returns:
            FAISS vector store
        """
        logger.info("Creating FAISS vector store...")
        
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        logger.info(f"Vector store created with {len(documents)} documents")
        return self.vector_store
    
    def save_vector_store(self):
        """Save vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save")
        
        logger.info(f"Saving vector store to {self.vector_store_path}")
        self.vector_store.save_local(str(self.vector_store_path))
        logger.info("Vector store saved successfully")
    
    def load_vector_store(self) -> bool:
        """
        Load existing vector store from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.vector_store_path.exists():
            logger.warning(f"Vector store not found at {self.vector_store_path}")
            return False
        
        try:
            logger.info(f"Loading vector store from {self.vector_store_path}")
            self.vector_store = FAISS.load_local(
                str(self.vector_store_path),
                self.embeddings
            )
            logger.info("Vector store loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def search(
        self, 
        query: str, 
        k: int = None,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return (default from config)
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with scores
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if k is None:
            k = config.top_k_results
        
        logger.info(f"Searching for: '{query}' (k={k})")
        
        # Perform similarity search with scores
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query, 
            k=k
        )
        
        # Filter by score threshold and format results
        results = []
        for doc, score in docs_and_scores:
            # FAISS returns distance, lower is better
            # Convert to similarity score (higher is better)
            similarity = 1 / (1 + score)
            
            if similarity >= score_threshold:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': similarity
                })
        
        logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def ingest_documents(self, force_rebuild: bool = False) -> int:
        """
        Complete ingestion pipeline: load, chunk, and index documents.
        
        Args:
            force_rebuild: Force rebuild even if vector store exists
            
        Returns:
            Number of chunks indexed
        """
        # Check if vector store already exists
        if not force_rebuild and self.load_vector_store():
            logger.info("Using existing vector store")
            return 0
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_path}")
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        # Save vector store
        self.save_vector_store()
        
        return len(chunks)
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """
        Add a single document to the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        if metadata is None:
            metadata = {}
        
        # Create document
        doc = Document(page_content=content, metadata=metadata)
        
        # Chunk if necessary
        chunks = self.text_splitter.split_documents([doc])
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        logger.info(f"Added document with {len(chunks)} chunks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        stats = {
            'data_path': str(self.data_path),
            'vector_store_exists': self.vector_store is not None,
            'embedding_model': config.embedding_model,
            'chunk_size': config.chunk_size,
            'chunk_overlap': config.chunk_overlap
        }
        
        if self.vector_store:
            stats['num_documents'] = self.vector_store.index.ntotal
        
        return stats


# Global knowledge base instance
kb = KnowledgeBase()
```

##### Step 3: Create Sample Educational Content

**Create sample documents for testing:**

```bash
# Create sample science document
cat > data/knowledge_base/photosynthesis.txt << 'EOF'
Photosynthesis

Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water. Photosynthesis in plants generally involves the green pigment chlorophyll and generates oxygen as a byproduct.

The Process
Plants absorb light energy from the sun through chlorophyll in their leaves. This energy is used to convert carbon dioxide from the air and water from the soil into glucose (sugar) and oxygen. The oxygen is released into the atmosphere as a waste product.

Chemical Equation
The overall chemical equation for photosynthesis is:
6CO2 + 6H2O + light energy → C6H12O6 + 6O2

This means six molecules of carbon dioxide plus six molecules of water, in the presence of light energy, produce one molecule of glucose and six molecules of oxygen.

Importance
Photosynthesis is crucial for life on Earth. It is the primary source of organic compounds and oxygen. Almost all organisms depend on photosynthesis either directly or indirectly for their survival.
EOF

# Create sample math document
cat > data/knowledge_base/quadratic_equations.txt << 'EOF'
Quadratic Equations

A quadratic equation is a second-order polynomial equation in a single variable x, with the form:
ax² + bx + c = 0

where a, b, and c are constants, and a ≠ 0.

Solving Methods

1. Factoring
If the quadratic can be factored, express it as (x - r1)(x - r2) = 0, where r1 and r2 are the roots.

2. Quadratic Formula
The most reliable method uses the quadratic formula:
x = (-b ± √(b² - 4ac)) / 2a

The discriminant (b² - 4ac) determines the nature of roots:
- If positive: two distinct real roots
- If zero: one repeated real root
- If negative: two complex roots

3. Completing the Square
Rewrite the equation in the form (x - h)² = k to find the solution.

Example
Solve: x² - 5x + 6 = 0
Using the quadratic formula with a=1, b=-5, c=6:
x = (5 ± √(25-24)) / 2 = (5 ± 1) / 2
Therefore: x = 3 or x = 2
EOF

# Create sample history document
cat > data/knowledge_base/world_war_2.txt << 'EOF'
World War II Overview

World War II (1939-1945) was a global war involving most of the world's nations, including all great powers, eventually forming two opposing military alliances: the Allies and the Axis.

Key Events

The war began with Germany's invasion of Poland on September 1, 1939. Britain and France declared war on Germany two days later. The conflict expanded globally, involving nations from every continent.

Major Turning Points:
- Battle of Britain (1940): RAF defended Britain from German air attacks
- Pearl Harbor (1941): US entered the war after Japanese attack
- Battle of Stalingrad (1942-43): Turning point on Eastern Front
- D-Day (1944): Allied invasion of Normandy
- Atomic bombings (1945): Japan surrendered after Hiroshima and Nagasaki

The war ended in Europe on May 8, 1945 (V-E Day) and in Asia on September 2, 1945 (V-J Day) after Japan's surrender.

Impact
World War II resulted in an estimated 70-85 million fatalities, making it the deadliest conflict in human history. It led to the establishment of the United Nations and significantly influenced the geopolitical landscape of the modern world.
EOF
