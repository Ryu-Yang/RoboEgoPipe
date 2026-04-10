import cv2
import mediapipe as mp

from roboegopipe.mediapipe.utils import draw_landmarks_on_image


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
    
    cv2.imshow("Hand Landmark Detection", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(0)
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
landmarker.close()

print(f"处理完成，共处理 {frame_count} 帧")