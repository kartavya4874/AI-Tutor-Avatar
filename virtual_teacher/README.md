# AI Virtual Teacher - Setup Guide

## Project Structure
```
virtual_teacher/
├── avatar_module.py          # 3D Avatar handling
├── voice_cloning_module.py   # Voice cloning and TTS
├── llm_response_module.py    # LLM integration and response refinement
├── integration_demo.py       # Complete system integration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    ├── voice_samples/        # Voice training samples
    ├── avatar_models/        # Avatar configuration files
    └── knowledge_base/       # Educational content database
```

## Requirements.txt

```txt
# Core ML/AI Libraries
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0

# Audio Processing
librosa>=0.10.0
soundfile>=0.12.0
sounddevice>=0.4.6
pyaudio>=0.2.11

# Computer Vision & Avatar
opencv-python>=4.8.0
mediapipe>=0.10.0
Pillow>=9.5.0

# Voice Cloning (Coqui TTS)
coqui-tts>=0.13.0
# Alternative: TTS>=0.13.0

# LLM and NLP
openai>=0.27.0
requests>=2.31.0
aiohttp>=3.8.0
langchain>=0.0.200

# Vector Database (choose one)
faiss-cpu>=1.7.4
# OR chromadb>=0.4.0

# Speech Recognition
openai-whisper>=20230314

# Web Framework Options
streamlit>=1.25.0
# OR flask>=2.3.0

# Backend API
fastapi>=0.100.0
uvicorn>=0.22.0
pydantic>=2.0.0

# Database
sqlalchemy>=2.0.0
sqlite3  # Built-in with Python

# Utilities
python-dotenv>=1.0.0
asyncio  # Built-in with Python 3.7+
threading  # Built-in
logging  # Built-in
pathlib  # Built-in
dataclasses  # Built-in with Python 3.7+
enum  # Built-in
time  # Built-in
json  # Built-in
re  # Built-in
```

## Installation Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv_teacher

# Activate virtual environment
# On Windows:
venv_teacher\Scripts\activate
# On macOS/Linux:
source venv_teacher/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 2. Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# For CUDA support (if you have GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Additional Setup

#### Audio System Setup
```bash
# On Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio

# On macOS with Homebrew
brew install portaudio
pip install pyaudio

# On Windows
# PyAudio should install automatically with pip
```

#### Coqui TTS Setup
```bash
# Install Coqui TTS
pip install coqui-tts[all]

# Test TTS installation
tts --text "Hello, I am your AI teacher" --out_path test_output.wav
```

### 4. Environment Variables

Create a `.env` file in your project root:

```env
# OpenAI API (if using external LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Alternative LLM APIs
HUGGINGFACE_API_KEY=your_huggingface_key_here

# System Configuration
DEVICE=cpu  # or cuda if GPU available
LOG_LEVEL=INFO
```

### 5. Test Installation

```bash
# Test individual modules
python avatar_module.py
python voice_cloning_module.py
python llm_response_module.py

# Test complete integration
python integration_demo.py
```

## Quick Start Guide

### 1. Basic Usage

```python
from integration_demo import create_virtual_teacher

# Create teacher instance
teacher = create_virtual_teacher()

# Start session
session_id = teacher.start_session("student_001", "intermediate")

# Process query
response_data = teacher.process_query("Explain photosynthesis")

# Play response (audio + avatar)
teacher.play_response(response_data)

# End session
teacher.end_session()
```

### 2. Interactive Demo

```bash
python integration_demo.py
# Select option 1 for interactive demo
```

### 3. Custom Configuration

```python
config = {
    "avatar": {
        "width": 800,
        "height": 600,
        "animation_fps": 30
    },
    "llm": {
        "model_name": "gpt-4",
        "use_local_model": False,
        "api_key": "your_api_key"
    },
    "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
    "device": "cuda"  # if GPU available
}

teacher = create_virtual_teacher(config)
```

## Module Details

### Avatar Module (`avatar_module.py`)
- Creates 3D-style avatar with facial features
- Handles lip synchronization with audio
- Supports emotion expression
- Real-time animation capabilities

**Key Features:**
- Mouth movement sync with audio amplitude
- Idle animations (blinking, subtle movements)
- Customizable appearance
- OpenCV-based rendering

### Voice Cloning Module (`voice_cloning_module.py`)
- Voice recording and sample management
- Text-to-speech with voice cloning
- Audio preprocessing and optimization
- Real-time TTS capabilities

**Key Features:**
- Coqui TTS integration
- Voice similarity analysis
- Audio amplitude extraction for lip sync
- Multiple voice sample support

### LLM Response Module (`llm_response_module.py`)
- Query classification and processing
- Response refinement for teaching context
- Session management
- Multi-modal response generation

**Key Features:**
- Teaching-focused prompt templates
- Response quality refinement
- Conversation context tracking
- Multiple LLM provider support

## Troubleshooting

### Common Issues

#### 1. Audio Issues
```bash
# If sounddevice fails to find audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# If PortAudio errors occur
pip uninstall pyaudio
pip install --upgrade pyaudio
```

#### 2. TTS Issues
```bash
# If Coqui TTS model download fails
python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/en/ljspeech/tacotron2-DDC')"
```

#### 3. OpenCV Issues
```bash
# If OpenCV window doesn't display
pip uninstall opencv-python
pip install opencv-python-headless==4.8.0.74
pip install opencv-python==4.8.0.74
```

#### 4. Memory Issues
- Reduce avatar resolution in config
- Use smaller TTS models
- Enable GPU acceleration if available

### Performance Optimization

#### For CPU-only systems:
```python
config = {
    "device": "cpu",
    "avatar": {"animation_fps": 15},  # Reduce FPS
    "llm": {"max_tokens": 300}  # Shorter responses
}
```

#### For GPU systems:
```python
config = {
    "device": "cuda",
    "avatar": {"animation_fps": 30},
    "llm": {"use_local_model": True}  # Use local GPU model
}
```

## Development Notes

### For 3-Person Team Division:
- **Person A (NLP Engineer)**: Focus on `llm_response_module.py` and knowledge base
- **Person B (Voice/Avatar Engineer)**: Focus on `voice_cloning_module.py` and `avatar_module.py`
- **Person C (Integration Engineer)**: Focus on `integration_demo.py` and UI development

### Next Steps:
1. Implement Streamlit web interface
2. Add knowledge base with RAG pipeline
3. Implement more sophisticated avatar animations
4. Add gesture recognition
5. Create evaluation metrics

## License
This project is for educational purposes as part of a final year academic project.

## Support
For issues and questions, please check the troubleshooting section or create detailed error logs for debugging.
