# voice_cloning_module.py
"""
Voice Cloning Module for AI Virtual Teacher
Handles voice cloning, TTS, and audio processing using Coqui TTS
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
import threading
import time
import os
import tempfile
from typing import Optional, List, Dict, Any, Union, Callable
import logging
from pathlib import Path

# Note: In production, uncomment these imports
# import TTS
# from TTS.api import TTS
# from TTS.config import load_config
# from TTS.utils.manage import ModelManager

class VoiceCloner:
    """
    Voice cloning class using Coqui TTS for generating personalized speech
    """
    
    def __init__(self, model_name: str = "tts_models/en/ljspeech/tacotron2-DDC", 
                 device: str = "cpu"):
        """
        Initialize Voice Cloner
        
        Args:
            model_name: TTS model name from Coqui TTS
            device: Device to run inference ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.tts_model = None
        self.is_initialized = False
        self.sample_rate = 22050
        self.voice_samples = {}
        
        # Audio processing settings
        self.audio_config = {
            "sample_rate": self.sample_rate,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "f_min": 0,
            "f_max": 8000
        }
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize TTS model
        self._initialize_tts()
    
    def _initialize_tts(self):
        """Initialize TTS model"""
        try:
            # In production, use actual Coqui TTS
            # self.tts_model = TTS(model_name=self.model_name, progress_bar=True)
            # self.tts_model.to(self.device)
            
            # For demo/development, create mock model
            self.tts_model = self._create_mock_tts()
            self.is_initialized = True
            self.logger.info(f"TTS model {self.model_name} initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS model: {e}")
            self.is_initialized = False
    
    def _create_mock_tts(self):
        """Create mock TTS for development/testing"""
        class MockTTS:
            def tts(self, text, speaker_wav=None):
                # Generate mock audio (sine wave for demo)
                duration = len(text) * 0.1  # 0.1 seconds per character
                t = np.linspace(0, duration, int(22050 * duration))
                # Simple sine wave with varying frequency based on text
                freq = 200 + len(text) % 100
                audio = np.sin(2 * np.pi * freq * t) * 0.3
                return audio.astype(np.float32)
        
        return MockTTS()
    
    def record_voice_sample(self, duration: float = 10.0, 
                           sample_name: str = "default") -> bool:
        """
        Record voice sample for cloning
        
        Args:
            duration: Recording duration in seconds
            sample_name: Name to store the sample
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Recording voice sample '{sample_name}' for {duration} seconds...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate), 
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32'
            )
            
            print("Recording... Speak clearly!")
            sd.wait()  # Wait until recording is finished
            print("Recording completed!")
            
            # Store the sample
            self.voice_samples[sample_name] = {
                'audio': audio_data.flatten(),
                'duration': duration,
                'sample_rate': self.sample_rate
            }
            
            self.logger.info(f"Voice sample '{sample_name}' recorded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record voice sample: {e}")
            return False
    
    def load_voice_sample(self, file_path: str, 
                         sample_name: str = "default") -> bool:
        """
        Load voice sample from file
        
        Args:
            file_path: Path to audio file
            sample_name: Name to store the sample
            
        Returns:
            bool: Success status
        """
        try:
            audio_data, sample_rate = librosa.load(file_path, sr=self.sample_rate)
            
            self.voice_samples[sample_name] = {
                'audio': audio_data,
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate
            }
            
            self.logger.info(f"Voice sample '{sample_name}' loaded from {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load voice sample from {file_path}: {e}")
            return False
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio for better TTS quality
        
        Args:
            audio: Raw audio array
            
        Returns:
            np.ndarray: Processed audio
        """
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Remove silence from beginning and end
        audio = librosa.effects.trim(audio, top_db=20)[0]
        
        # Apply light noise reduction
        audio = librosa.effects.preemphasis(audio)
        
        return audio
    
    def clone_voice_tts(self, text: str, 
                       voice_sample_name: str = "default",
                       output_path: Optional[str] = None) -> Union[np.ndarray, str]:
        """
        Generate speech using cloned voice
        
        Args:
            text: Text to convert to speech
            voice_sample_name: Name of voice sample to use
            output_path: Optional output file path
            
        Returns:
            np.ndarray or str: Generated audio array or file path
        """
        if not self.is_initialized:
            self.logger.error("TTS model not initialized")
            return None
        
        try:
            # Get voice sample
            if voice_sample_name not in self.voice_samples:
                self.logger.warning(f"Voice sample '{voice_sample_name}' not found, using default synthesis")
                speaker_wav = None
            else:
                speaker_wav = self.voice_samples[voice_sample_name]['audio']
            
            # Generate speech
            # In production with Coqui TTS:
            # audio = self.tts_model.tts(text=text, speaker_wav=speaker_wav)
            
            # For demo:
            audio = self.tts_model.tts(text=text, speaker_wav=speaker_wav)
            
            # Post-process audio
            audio = self.preprocess_audio(audio)
            
            # Save to file if requested
            if output_path:
                sf.write(output_path, audio, self.sample_rate)
                return output_path
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to generate TTS: {e}")
            return None
    
    def play_audio(self, audio: Union[np.ndarray, str]):
        """
        Play audio array or file
        
        Args:
            audio: Audio array or file path
        """
        try:
            if isinstance(audio, str):
                audio_data, _ = librosa.load(audio, sr=self.sample_rate)
            else:
                audio_data = audio
            
            sd.play(audio_data, self.sample_rate)
            
        except Exception as e:
            self.logger.error(f"Failed to play audio: {e}")
    
    def get_audio_amplitude(self, audio: np.ndarray, 
                           frame_size: int = 1024) -> List[float]:
        """
        Extract audio amplitude for lip sync
        
        Args:
            audio: Audio array
            frame_size: Frame size for amplitude calculation
            
        Returns:
            List[float]: Amplitude values for each frame
        """
        amplitudes = []
        for i in range(0, len(audio), frame_size):
            frame = audio[i:i+frame_size]
            amplitude = np.sqrt(np.mean(frame**2))  # RMS amplitude
            amplitudes.append(amplitude)
        
        return amplitudes
    
    def real_time_tts(self, text_stream: Callable[[], str], 
                     voice_sample_name: str = "default",
                     callback: Optional[Callable[[np.ndarray], None]] = None):
        """
        Real-time TTS for streaming text
        
        Args:
            text_stream: Function that returns text chunks
            voice_sample_name: Voice sample to use
            callback: Callback function for generated audio chunks
        """
        def tts_worker():
            while True:
                text_chunk = text_stream()
                if text_chunk is None:
                    break
                
                audio_chunk = self.clone_voice_tts(text_chunk, voice_sample_name)
                if audio_chunk is not None and callback:
                    callback(audio_chunk)
        
        worker_thread = threading.Thread(target=tts_worker)
        worker_thread.daemon = True
        worker_thread.start()
        return worker_thread
    
    def analyze_voice_similarity(self, sample1_name: str, 
                                sample2_name: str) -> float:
        """
        Analyze similarity between two voice samples
        
        Args:
            sample1_name: First voice sample name
            sample2_name: Second voice sample name
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        try:
            if sample1_name not in self.voice_samples or sample2_name not in self.voice_samples:
                return 0.0
            
            audio1 = self.voice_samples[sample1_name]['audio']
            audio2 = self.voice_samples[sample2_name]['audio']
            
            # Extract MFCC features
            mfcc1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=13)
            mfcc2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate similarity using cosine similarity
            mfcc1_mean = np.mean(mfcc1, axis=1)
            mfcc2_mean = np.mean(mfcc2, axis=1)
            
            similarity = np.dot(mfcc1_mean, mfcc2_mean) / (
                np.linalg.norm(mfcc1_mean) * np.linalg.norm(mfcc2_mean)
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Failed to analyze voice similarity: {e}")
            return 0.0
    
    def get_voice_characteristics(self, sample_name: str) -> Dict[str, Any]:
        """
        Extract voice characteristics from sample
        
        Args:
            sample_name: Voice sample name
            
        Returns:
            Dict: Voice characteristics
        """
        if sample_name not in self.voice_samples:
            return {}
        
        audio = self.voice_samples[sample_name]['audio']
        
        try:
            characteristics = {
                'duration': len(audio) / self.sample_rate,
                'fundamental_freq': float(np.mean(librosa.yin(audio, fmin=50, fmax=400))),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))),
                'spectral_rolloff': float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
                'energy': float(np.mean(audio**2))
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Failed to extract voice characteristics: {e}")
            return {}


# Factory and utility functions
def create_voice_cloner(model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
                       device: str = "cpu") -> VoiceCloner:
    """
    Factory function to create VoiceCloner instance
    
    Args:
        model_name: TTS model name
        device: Device for inference
        
    Returns:
        VoiceCloner: Configured voice cloner instance
    """
    return VoiceCloner(model_name=model_name, device=device)


def test_voice_cloning():
    """Test function for voice cloning module"""
    print("Testing Voice Cloning Module...")
    
    # Create voice cloner
    voice_cloner = create_voice_cloner()
    
    # Test TTS generation
    test_text = "Hello, I am your AI virtual teacher. How can I help you today?"
    audio = voice_cloner.clone_voice_tts(test_text)
    
    if audio is not None:
        print(f"Generated audio shape: {audio.shape}")
        print("TTS generation test passed")
        
        # Test amplitude extraction for lip sync
        amplitudes = voice_cloner.get_audio_amplitude(audio)
        print(f"Extracted {len(amplitudes)} amplitude frames")
        print("Amplitude extraction test passed")
    else:
        print("TTS generation test failed")
    
    print("Voice Cloning Module test completed!")


if __name__ == "__main__":
    test_voice_cloning()
