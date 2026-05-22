import time

import cv2 as cv


def open_camera():
    video = cv.VideoCapture(1) # Change the index depending on your camera
    video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    if not video.isOpened():
        print("Can't open camera !")
        exit()

    return video

def read_frame(video) :
    # Capture frame-by-frame and ret is bool
    # to make sure the capture is working
    # video.read() returns a boolean and
    # the image captured by the camera
    ret, frame = video.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame.")
        return False, None

    # frame = cv.flip(frame, 1)

    return ret, frame

def convert_to_RGB(frame) :
    # Convert OpenCV BGR image to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    return rgb_frame

def get_timestamp() :
    # Create timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000)

    return timestamp_ms

def release_video(video) :
    # When everything done, release the capture
    video.release()
    cv.destroyAllWindows()