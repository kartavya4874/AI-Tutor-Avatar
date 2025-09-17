# integration_demo.py
"""
Integration Demo for AI Virtual Teacher System
Shows how to integrate Avatar, Voice Cloning, and LLM Response modules
"""

import asyncio
import threading
import time
import cv2
import numpy as np
from typing import Optional, Dict, Any
import logging

# Import our custom modules
from avatar_module import create_avatar_instance, Avatar3D
from voice_cloning_module import create_voice_cloner, VoiceCloner
from llm_response_module import (
    create_llm_handler, create_query_context, 
    LLMResponseHandler, QueryContext, TeachingResponse
)

class VirtualTeacher:
    """
    Integrated Virtual Teacher System
    Combines 3D Avatar, Voice Cloning, and LLM Response capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Virtual Teacher
        
        Args:
            config: Configuration dictionary for all components
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.avatar = create_avatar_instance(self.config.get("avatar", {}))
        self.voice_cloner = create_voice_cloner(
            model_name=self.config.get("tts_model", "tts_models/en/ljspeech/tacotron2-DDC"),
            device=self.config.get("device", "cpu")
        )
        self.llm_handler = create_llm_handler(self.config.get("llm", {}))
        
        # System state
        self.is_active = False
        self.current_session_id = None
        self.user_id = None
        
        # Threading for real-time components
        self.avatar_thread = None
        self.audio_thread = None
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for Virtual Teacher"""
        return {
            "avatar": {
                "width": 640,
                "height": 480,
                "animation_fps": 30
            },
            "tts_model": "tts_models/en/ljspeech/tacotron2-DDC",
            "device": "cpu",
            "llm": {
                "use_local_model": True,
                "refinement_enabled": True,
                "model_name": "gpt-3.5-turbo"
            },
            "learning_level": "intermediate",
            "voice_sample_name": "teacher_voice"
        }
    
    def start_session(self, user_id: str, learning_level: str = "intermediate") -> str:
        """
        Start a new teaching session
        
        Args:
            user_id: User identifier
            learning_level: User's learning level
            
        Returns:
            str: Session ID
        """
        self.user_id = user_id
        self.current_session_id = f"session_{user_id}_{int(time.time())}"
        self.config["learning_level"] = learning_level
        self.is_active = True
        
        # Start avatar animation
        self.avatar.start_idle_animation()
        
        self.logger.info(f"Started session {self.current_session_id} for user {user_id}")
        return self.current_session_id
    
    def end_session(self):
        """End the current teaching session"""
        self.is_active = False
        
        # Stop avatar animation
        self.avatar.stop_idle_animation()
        
        # Get session summary
        if self.current_session_id:
            summary = self.llm_handler.get_session_summary(self.current_session_id)
            self.logger.info(f"Session ended. Summary: {summary}")
        
        self.current_session_id = None
        self.user_id = None
    
    def setup_voice(self, voice_sample_path: Optional[str] = None) -> bool:
        """
        Setup voice cloning
        
        Args:
            voice_sample_path: Path to voice sample file (optional)
            
        Returns:
            bool: Success status
        """
        sample_name = self.config["voice_sample_name"]
        
        if voice_sample_path:
            # Load voice sample from file
            success = self.voice_cloner.load_voice_sample(voice_sample_path, sample_name)
        else:
            # Record voice sample
            print("Voice setup required. Please speak for 10 seconds when recording starts...")
            success = self.voice_cloner.record_voice_sample(duration=10.0, sample_name=sample_name)
        
        if success:
            self.logger.info(f"Voice setup completed successfully")
            return True
        else:
            self.logger.error("Voice setup failed")
            return False
    
    async def process_query_async(self, query: str) -> Dict[str, Any]:
        """
        Process user query asynchronously
        
        Args:
            query: User query/question
            
        Returns:
            Dict: Response data including text, audio, and video
        """
        if not self.is_active or not self.current_session_id:
            return {"error": "No active session"}
        
        try:
            # Create query context
            context = create_query_context(
                user_id=self.user_id,
                session_id=self.current_session_id,
                learning_level=self.config["learning_level"]
            )
            
            # Generate LLM response
            self.logger.info(f"Processing query: {query}")
            teaching_response = await self.llm_handler.generate_response_async(query, context)
            
            # Generate voice audio
            self.logger.info("Generating speech...")
            audio = self.voice_cloner.clone_voice_tts(
                text=teaching_response.content,
                voice_sample_name=self.config["voice_sample_name"]
            )
            
            if audio is None:
                self.logger.error("Failed to generate audio")
                return {"error": "Audio generation failed"}
            
            # Extract audio amplitudes for lip sync
            amplitudes = self.voice_cloner.get_audio_amplitude(audio)
            
            # Generate avatar video with lip sync
            self.logger.info("Generating avatar animation...")
            video_frames = self._generate_avatar_video(audio, amplitudes)
            
            return {
                "response": teaching_response,
                "audio": audio,
                "video_frames": video_frames,
                "amplitudes": amplitudes,
                "session_id": self.current_session_id
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            return {"error": str(e)}
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query synchronously
        
        Args:
            query: User query/question
            
        Returns:
            Dict: Response data
        """
        return asyncio.run(self.process_query_async(query))
    
    def _generate_avatar_video(self, audio: np.ndarray, 
                              amplitudes: list) -> list:
        """
        Generate avatar video frames synchronized with audio
        
        Args:
            audio: Audio array
            amplitudes: Audio amplitude values
            
        Returns:
            list: Video frames
        """
        frames = []
        frame_duration = len(audio) / len(amplitudes)
        
        for i, amplitude in enumerate(amplitudes):
            # Sync avatar mouth with audio amplitude
            self.avatar.sync_with_audio(amplitude)
            
            # Generate frame
            frame = self.avatar.render_frame()
            frames.append(frame)
        
        return frames
    
    def play_response(self, response_data: Dict[str, Any]):
        """
        Play the complete response (audio + video)
        
        Args:
            response_data: Response data from process_query
        """
        if "error" in response_data:
            print(f"Error: {response_data['error']}")
            return
        
        audio = response_data["audio"]
        video_frames = response_data["video_frames"]
        
        # Start audio playback in separate thread
        audio_thread = threading.Thread(
            target=self.voice_cloner.play_audio, 
            args=(audio,)
        )
        audio_thread.start()
        
        # Display video frames
        for frame in video_frames:
            cv2.imshow("AI Virtual Teacher", frame)
            if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
                break
        
        audio_thread.join()
        cv2.destroyAllWindows()
    
    def interactive_demo(self):
        """Run interactive demo of the Virtual Teacher"""
        print("=== AI Virtual Teacher Interactive Demo ===")
        print("Starting system initialization...")
        
        # Setup voice (optional)
        setup_voice = input("Do you want to setup voice cloning? (y/n): ").lower() == 'y'
        if setup_voice:
            voice_success = self.setup_voice()
            if not voice_success:
                print("Voice setup failed, using default voice")
        
        # Start session
        user_id = input("Enter your user ID: ") or "demo_user"
        learning_level = input("Enter learning level (beginner/intermediate/advanced): ") or "intermediate"
        
        session_id = self.start_session(user_id, learning_level)
        print(f"Session started: {session_id}")
        
        print("\nYou can now ask questions! Type 'quit' to exit.")
        print("Example questions:")
        print("- Explain photosynthesis")
        print("- Give me an example of Newton's laws")
        print("- Create a quiz about Python programming")
        print("- What is machine learning?")
        
        try:
            while True:
                query = input("\nYour question: ").strip()
                
                if query.lower() in ['quit', 'exit', 'bye']:
                    break
                
                if not query:
                    continue
                
                print("Processing your question...")
                
                # Process query
                response_data = self.process_query(query)
                
                if "error" in response_data:
                    print(f"Error: {response_data['error']}")
                    continue
                
                # Display text response
                teaching_response = response_data["response"]
                print(f"\nTeacher Response ({teaching_response.response_type.value}):")
                print("-" * 50)
                print(teaching_response.content)
                print("-" * 50)
                print(f"Key Concepts: {', '.join(teaching_response.key_concepts[:3])}")
                print(f"Confidence: {teaching_response.confidence:.2f}")
                print(f"Reading Time: {teaching_response.estimated_reading_time} seconds")
                
                # Ask if user wants to see avatar demo
                show_avatar = input("\nShow avatar demo? (y/n): ").lower() == 'y'
                if show_avatar:
                    print("Displaying avatar... Press 'q' to close video window")
                    self.play_response(response_data)
                
                # Show follow-up questions
                if teaching_response.follow_up_questions:
                    print("\nFollow-up questions:")
                    for i, question in enumerate(teaching_response.follow_up_questions, 1):
                        print(f"{i}. {question}")
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            # End session
            self.end_session()
            print("Session ended. Goodbye!")
    
    def batch_demo(self):
        """Run batch demo with predefined queries"""
        print("=== AI Virtual Teacher Batch Demo ===")
        
        # Start session
        session_id = self.start_session("demo_user", "intermediate")
        
        # Test queries
        test_queries = [
            "Explain how photosynthesis works in plants",
            "Give me a practical example of machine learning",
            "I'm confused about quantum mechanics, can you clarify?",
            "Create a quiz question about Python programming"
        ]
        
        try:
            for i, query in enumerate(test_queries, 1):
                print(f"\n=== Test {i}: {query} ===")
                
                response_data = self.process_query(query)
                
                if "error" not in response_data:
                    teaching_response = response_data["response"]
                    print(f"Response Type: {teaching_response.response_type.value}")
                    print(f"Content: {teaching_response.content[:200]}...")
                    print(f"Key Concepts: {teaching_response.key_concepts[:3]}")
                    print(f"Audio generated: {len(response_data['audio'])} samples")
                    print(f"Video frames: {len(response_data['video_frames'])}")
                else:
                    print(f"Error: {response_data['error']}")
                
                time.sleep(1)  # Brief pause between tests
        
        finally:
            self.end_session()
            print("Batch demo completed!")


# Utility functions
def create_virtual_teacher(config: Optional[Dict[str, Any]] = None) -> VirtualTeacher:
    """
    Factory function to create VirtualTeacher instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        VirtualTeacher: Configured virtual teacher instance
    """
    return VirtualTeacher(config)

def run_quick_test():
    """Quick test of all modules integration"""
    print("=== Quick Integration Test ===")
    
    # Create virtual teacher
    teacher = create_virtual_teacher()
    
    # Start session
    session_id = teacher.start_session("test_user")
    print(f"Session started: {session_id}")
    
    # Test query
    test_query = "Explain what artificial intelligence is"
    print(f"Testing query: {test_query}")
    
    response_data = teacher.process_query(test_query)
    
    if "error" not in response_data:
        response = response_data["response"]
        print(f"✓ LLM Response: {len(response.content)} characters")
        print(f"✓ Audio: {len(response_data['audio'])} samples")
        print(f"✓ Video: {len(response_data['video_frames'])} frames")
        print(f"✓ Response Type: {response.response_type.value}")
        print("Integration test PASSED!")
    else:
        print(f"✗ Integration test FAILED: {response_data['error']}")
    
    teacher.end_session()

def main():
    """Main function to run demos"""
    print("AI Virtual Teacher - Module Integration")
    print("=" * 50)
    
    mode = input("Select demo mode:\n1. Interactive Demo\n2. Batch Demo\n3. Quick Test\nChoice (1-3): ").strip()
    
    teacher = create_virtual_teacher()
    
    if mode == "1":
        teacher.interactive_demo()
    elif mode == "2":
        teacher.batch_demo()
    elif mode == "3":
        run_quick_test()
    else:
        print("Invalid choice. Running quick test...")
        run_quick_test()

if __name__ == "__main__":
    main()
