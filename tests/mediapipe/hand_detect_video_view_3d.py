import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from roboegopipe.mediapipe.utils import draw_landmarks_on_image, visualize_landmarks_3d, create_3d_visualization_figure


model_path = './models/mediapipe/hand_landmarker.task'
video_path = './data/baai_example_03.mp4'


base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2
    )
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("错误：无法打开视频文件")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"视频帧率: {fps} fps")

# Setup matplotlib for interactive 3D visualization
plt.ion()  # Turn on interactive mode
fig_3d, ax_3d = create_3d_visualization_figure()
fig_3d.canvas.manager.set_window_title("3D Hand Landmarks View")
fig_3d.subplots_adjust(left=0, right=1, bottom=0, top=1)  # Adjust layout

# Variables for 3D visualization update
last_3d_update_frame = 0
update_3d_every_n_frames = 1  # Update 3D view every N frames for performance

frame_count = 0
while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    timestamp_ms = int(frame_count * (1000 / fps))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    hand_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
    
    if frame_count % 30 == 0:
        print(f"Frame {frame_count}, timestamp: {timestamp_ms}ms")
        if hand_landmarker_result.hand_landmarks:
            print(f"检测到 {len(hand_landmarker_result.hand_landmarks)} 只手")
    
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
    
    # Display 2D annotated image
    cv2.imshow("Hand Landmark Detection (2D)", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    # Update 3D visualization periodically for performance
    if frame_count - last_3d_update_frame >= update_3d_every_n_frames and hand_landmarker_result.hand_landmarks:
        # Update 3D plot with new landmarks
        visualize_landmarks_3d(
            hand_landmarker_result, 
            ax=ax_3d, 
            title=f"Hand Landmarks 3D - Frame {frame_count}"
        )
        
        # Redraw the 3D plot
        fig_3d.canvas.draw()
        fig_3d.canvas.flush_events()
        
        last_3d_update_frame = frame_count
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    elif key == ord('3'):
        # Force update 3D view
        if hand_landmarker_result.hand_landmarks:
            visualize_landmarks_3d(
                hand_landmarker_result, 
                ax=ax_3d, 
                title=f"Hand Landmarks 3D - Frame {frame_count} (Manual Update)"
            )
            fig_3d.canvas.draw()
            fig_3d.canvas.flush_events()
            print(f"手动更新3D视图 - 帧 {frame_count}")
    elif key == ord('s'):
        # Save current 3D view as image
        if hand_landmarker_result.hand_landmarks:
            filename = f"hand_3d_frame_{frame_count}.png"
            fig_3d.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"3D视图已保存为: {filename}")
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
landmarker.close()

# Keep 3D window open for inspection
print(f"处理完成，共处理 {frame_count} 帧")
print("3D窗口保持打开，可以旋转查看。按任意键关闭3D窗口...")

# Turn off interactive mode and show final plot
plt.ioff()
plt.show()

print("程序结束")
