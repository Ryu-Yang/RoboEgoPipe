import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from roboegopipe.mediapipe.utils import draw_landmarks_on_image, visualize_landmarks_3d, create_3d_visualization_figure


model_path = './models/mediapipe/hand_landmarker.task'


class Detector():
    def __init__(self):
        self.base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=self.base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2
            )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)

    def detect(self, frame, timestamp_ns_unix):
        # Convert frame to uint8 if it's float32
        if frame.dtype == np.float32:
            # Check if image is in 0-1 range or 0-255 range
            if frame.max() <= 1.0:
                # Convert from 0-1 float range to 0-255 uint8
                frame = (frame * 255).astype(np.uint8)
            else:
                # Convert from float to uint8 (clipping values outside 0-255)
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Convert timestamp from nanoseconds to milliseconds (MediaPipe expects int)
        # timestamp_ns_unix is in nanoseconds, convert to ms as int
        timestamp_ms_int = int(timestamp_ns_unix / 1_000_000)

        hand_landmarker_result = self.landmarker.detect_for_video(mp_image, timestamp_ms_int)

        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)

        return annotated_image

    # def visualize_landmarks_3d(
    #         hand_landmarker_result, 
    #         ax=ax_3d, 
    #         title=f"Hand Landmarks 3D - Frame {frame_count}"
    #     )
