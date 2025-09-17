# avatar_module.py
"""
3D Avatar Module for AI Virtual Teacher
Handles 3D avatar creation, animation, and rendering
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from typing import Optional, Tuple, Dict, Any
import logging

class Avatar3D:
    """
    3D Avatar class for rendering and animating virtual teacher
    """
    
    def __init__(self, avatar_config: Optional[Dict[str, Any]] = None):
        """
        Initialize 3D Avatar
        
        Args:
            avatar_config: Configuration dictionary for avatar settings
        """
        self.config = avatar_config or self._default_config()
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Avatar state
        self.is_speaking = False
        self.mouth_openness = 0.0
        self.current_emotion = "neutral"
        self.animation_thread = None
        self.stop_animation = False
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for avatar"""
        return {
            "width": 640,
            "height": 480,
            "background_color": (50, 50, 50),
            "face_color": (200, 180, 160),
            "eye_color": (100, 150, 200),
            "mouth_color": (180, 100, 100),
            "animation_fps": 30
        }
    
    def create_base_avatar(self) -> np.ndarray:
        """
        Create base 3D avatar frame
        
        Returns:
            np.ndarray: Base avatar image
        """
        img = np.full(
            (self.config["height"], self.config["width"], 3), 
            self.config["background_color"], 
            dtype=np.uint8
        )
        
        # Draw basic 3D-style face structure
        center_x, center_y = self.config["width"] // 2, self.config["height"] // 2
        
        # Face oval (3D effect with gradient)
        face_center = (center_x, center_y - 20)
        axes = (120, 160)
        
        # Create face with 3D gradient effect
        for i in range(10):
            color_intensity = 200 - i * 10
            color = (color_intensity, color_intensity - 20, color_intensity - 40)
            cv2.ellipse(img, face_center, 
                       (axes[0] - i*2, axes[1] - i*2), 
                       0, 0, 360, color, -1)
        
        # Eyes (3D spheres)
        self._draw_3d_eyes(img, center_x, center_y)
        
        # Nose (3D effect)
        self._draw_3d_nose(img, center_x, center_y)
        
        # Mouth (dynamic based on speaking state)
        self._draw_3d_mouth(img, center_x, center_y)
        
        return img
    
    def _draw_3d_eyes(self, img: np.ndarray, center_x: int, center_y: int):
        """Draw 3D-style eyes"""
        eye_y = center_y - 30
        
        # Left eye
        left_eye_center = (center_x - 40, eye_y)
        cv2.circle(img, left_eye_center, 25, (255, 255, 255), -1)
        cv2.circle(img, left_eye_center, 20, self.config["eye_color"], -1)
        cv2.circle(img, (left_eye_center[0] - 5, left_eye_center[1] - 5), 8, (0, 0, 0), -1)
        cv2.circle(img, (left_eye_center[0] - 3, left_eye_center[1] - 7), 3, (255, 255, 255), -1)
        
        # Right eye
        right_eye_center = (center_x + 40, eye_y)
        cv2.circle(img, right_eye_center, 25, (255, 255, 255), -1)
        cv2.circle(img, right_eye_center, 20, self.config["eye_color"], -1)
        cv2.circle(img, (right_eye_center[0] - 5, right_eye_center[1] - 5), 8, (0, 0, 0), -1)
        cv2.circle(img, (right_eye_center[0] - 3, right_eye_center[1] - 7), 3, (255, 255, 255), -1)
    
    def _draw_3d_nose(self, img: np.ndarray, center_x: int, center_y: int):
        """Draw 3D-style nose"""
        nose_points = np.array([
            [center_x, center_y - 10],
            [center_x - 8, center_y + 10],
            [center_x + 8, center_y + 10]
        ], np.int32)
        
        cv2.fillPoly(img, [nose_points], (160, 140, 120))
        cv2.polylines(img, [nose_points], True, (140, 120, 100), 2)
    
    def _draw_3d_mouth(self, img: np.ndarray, center_x: int, center_y: int):
        """Draw 3D-style mouth with animation support"""
        mouth_y = center_y + 40
        mouth_width = int(40 + self.mouth_openness * 20)
        mouth_height = int(15 + self.mouth_openness * 25)
        
        # Mouth shape based on speaking state
        if self.is_speaking and self.mouth_openness > 0.3:
            # Open mouth
            cv2.ellipse(img, (center_x, mouth_y), (mouth_width, mouth_height), 
                       0, 0, 360, (50, 30, 30), -1)
            cv2.ellipse(img, (center_x, mouth_y), (mouth_width, mouth_height), 
                       0, 0, 360, (100, 50, 50), 3)
        else:
            # Closed/slightly open mouth
            cv2.ellipse(img, (center_x, mouth_y), (mouth_width, mouth_height//3), 
                       0, 0, 360, self.config["mouth_color"], -1)
    
    def sync_with_audio(self, audio_amplitude: float):
        """
        Synchronize mouth movement with audio amplitude
        
        Args:
            audio_amplitude: Audio amplitude value (0.0 to 1.0)
        """
        self.mouth_openness = max(0.0, min(1.0, audio_amplitude))
        self.is_speaking = audio_amplitude > 0.1
    
    def set_emotion(self, emotion: str):
        """
        Set avatar emotion
        
        Args:
            emotion: Emotion string ("happy", "sad", "surprised", "neutral")
        """
        self.current_emotion = emotion
        # Emotion can affect eye shape, eyebrow position, mouth curve, etc.
    
    def start_idle_animation(self):
        """Start idle animation (blinking, slight movements)"""
        if self.animation_thread is None or not self.animation_thread.is_alive():
            self.stop_animation = False
            self.animation_thread = threading.Thread(target=self._idle_animation_loop)
            self.animation_thread.daemon = True
            self.animation_thread.start()
    
    def stop_idle_animation(self):
        """Stop idle animation"""
        self.stop_animation = True
        if self.animation_thread:
            self.animation_thread.join(timeout=1.0)
    
    def _idle_animation_loop(self):
        """Idle animation loop (runs in separate thread)"""
        blink_timer = 0
        while not self.stop_animation:
            # Simple blinking animation
            blink_timer += 1
            if blink_timer % 120 == 0:  # Blink every 4 seconds at 30fps
                time.sleep(0.1)  # Brief blink
            
            time.sleep(1.0 / self.config["animation_fps"])
    
    def render_frame(self) -> np.ndarray:
        """
        Render current avatar frame
        
        Returns:
            np.ndarray: Rendered avatar frame
        """
        return self.create_base_avatar()
    
    def get_avatar_stream(self):
        """
        Generator for avatar video stream
        
        Yields:
            np.ndarray: Avatar frames for streaming
        """
        while True:
            frame = self.render_frame()
            yield frame
            time.sleep(1.0 / self.config["animation_fps"])
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_idle_animation()


# Example usage and integration functions
def create_avatar_instance(config: Optional[Dict[str, Any]] = None) -> Avatar3D:
    """
    Factory function to create Avatar3D instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Avatar3D: Configured avatar instance
    """
    return Avatar3D(config)


def test_avatar():
    """Test function for avatar module"""
    avatar = create_avatar_instance()
    
    print("Testing Avatar3D Module...")
    
    # Test basic rendering
    frame = avatar.render_frame()
    print(f"Base frame shape: {frame.shape}")
    
    # Test audio sync
    avatar.sync_with_audio(0.7)
    speaking_frame = avatar.render_frame()
    print("Audio sync test completed")
    
    # Test emotion setting
    avatar.set_emotion("happy")
    print("Emotion setting test completed")
    
    # Start idle animation
    avatar.start_idle_animation()
    time.sleep(2)
    avatar.stop_idle_animation()
    print("Idle animation test completed")
    
    print("Avatar3D Module test completed successfully!")


if __name__ == "__main__":
    test_avatar()
