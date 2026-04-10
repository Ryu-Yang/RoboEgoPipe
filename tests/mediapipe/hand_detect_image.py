import cv2
import mediapipe as mp
from pprint import pprint
from roboegopipe.mediapipe.utils import draw_landmarks_on_image


model_path = './models/mediapipe/hand_landmarker.task'
image_path = './data/test_mediapipe.png'


base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2
    )
landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

mp_image = mp.Image.create_from_file(image_path)

hand_landmarker_result = landmarker.detect(mp_image)

# print("result: ", hand_landmarker_result)
print("result: ")
pprint(vars(hand_landmarker_result))

annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), hand_landmarker_result)
cv2.imshow("test", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)

cv2.destroyAllWindows()
landmarker.close()