import cv2 as cv
import mediapipe as mp
import numpy as np

# Mediapipe tools
BaseOptions = mp.tasks.BaseOptions # Set up the model, what file .task to use, or GPU/CPU
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions # Set up how it will work : number of hands, confidence threshold
VisionRunningMode = mp.tasks.vision.RunningMode # Set up the mode : IMAGE, VIDEO or LIVE_STREAM

mp_hands = mp.tasks.vision.HandLandmarksConnections # Get the links between hand points
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles 

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO, num_hands = 2)

MARGIN = 10 

def detect_hands(rgb_frame, timestamp_ms, landmarker) :
    # Convert image into mediapipe format
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)

    # Apply the hand detection
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    return result

def draw_hand_result(rgb_image, result) :
    #List of list of the 21 points of the hand
    hand_landmarks_list = result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    height, width, _ = annotated_image.shape

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks.
        for landmark in hand_landmarks :
            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv.circle(img = annotated_image, center = (x, y), radius = 5, color = (255,29,141), thickness = -1)

        for connection in mp_hands.HAND_CONNECTIONS :
            # Get the index of the 2 landmarks to link, 'connection' is a tuple of 2 landmarks
            start_index = connection.start
            end_index = connection.end

            # Get their  real coordinates
            start = hand_landmarks[start_index]
            end = hand_landmarks[end_index]

            # Convert into pixels
            x1, y1 = int(start.x * width), int(start.y * height)
            x2, y2 = int(end.x * width), int(end.y * height)

            cv.line(img = annotated_image, pt1 = (x1, y1), pt2 = (x2, y2), color = (135, 206, 235), thickness = 2)

        # Get the top left corner of the detected hand's bounding box.
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        x_min = max(int(min(x_coordinates) * width) - MARGIN, 0)
        y_min = max(int(min(y_coordinates) * height) - MARGIN, 0)
        x_max = min(int(max(x_coordinates) * width) + MARGIN, width)
        y_max = min(int(max(y_coordinates) * height) + MARGIN, height)
        cv.rectangle(annotated_image,(x_min, y_min), (x_max, y_max), (200), 2)

    return annotated_image

