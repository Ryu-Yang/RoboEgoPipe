import mediapipe as mp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    mp_drawing.draw_landmarks(
      annotated_image,
      hand_landmarks,
      mp_hands.HAND_CONNECTIONS,
      mp_drawing_styles.get_default_hand_landmarks_style(),
      mp_drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def visualize_landmarks_3d(detection_result, ax=None, title="Hand Landmarks 3D"):
    """
    Visualize hand landmarks in 3D space.
    
    Args:
        detection_result: MediaPipe hand detection result
        ax: Matplotlib 3D axis (if None, creates new figure)
        title: Title for the plot
    
    Returns:
        Matplotlib figure and axis objects
    """
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        created_new = True
    else:
        fig = ax.figure
        created_new = False
    
    # Clear previous plot
    ax.clear()
    
    # Define colors for different hands
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # Plot each hand
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]
        
        # Extract coordinates
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]
        zs = [landmark.z for landmark in hand_landmarks]
        
        # Normalize for better visualization (z is depth, so invert for intuitive view)
        # MediaPipe: x and y are normalized [0, 1], z is relative depth
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)
        
        # Plot landmarks as scatter points
        color = colors[idx % len(colors)]
        ax.scatter(xs, ys, zs, c=color, s=50, alpha=0.8, label=f"{handedness[0].category_name}")
        
        # Draw connections between landmarks
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection.start
            end_idx = connection.end
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            ax.plot([start_point.x, end_point.x],
                    [start_point.y, end_point.y],
                    [start_point.z, end_point.z],
                    c=color, linewidth=2, alpha=0.6)
    
    # Set labels and title
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    ax.set_zlabel('Z (depth)')
    ax.set_title(title)
    
    # Invert Y axis to match image coordinates (top-left origin)
    ax.invert_yaxis()
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    if hand_landmarks_list:
        ax.legend()
    
    # Set view angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    if created_new:
        plt.tight_layout()
        return fig, ax
    else:
        return ax


def create_3d_visualization_figure():
    """
    Create a reusable figure for 3D visualization.
    
    Returns:
        Matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax
