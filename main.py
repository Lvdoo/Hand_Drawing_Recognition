from collections import Counter

import cv2 as cv
import torch
import torch.nn as nn

import camera
import drawing_detection
import model
import preprocessing
from hand_detection import *

prediction_list = []
prediction_done = False
final_prediction = None
softmax = nn.Softmax(dim = 1)
with HandLandmarker.create_from_options(options) as landmarker: 
    video = camera.open_camera()
    while True :
        ret, frame = camera.read_frame(video)
        timestamp = camera.get_timestamp()
        if not ret : 
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        rgb_frame = camera.convert_to_RGB(frame)
        result = detect_hands(rgb_frame, timestamp, landmarker)
    
        if len(result.hand_landmarks) == 0 : 
            annoted_image = rgb_frame.copy()
            cropped_image, box = drawing_detection.locate_drawing(annoted_image)
            if box is not None:
                x_min, y_min, x_max, y_max = box
                cv.rectangle(
                    annoted_image,
                    (x_min, y_min),
                    (x_max, y_max),
                    (255, 0, 0),
                    2)

            if cropped_image is None:
                print("You draw nothing")
                prediction_list.clear()

            else:
                final_image = preprocessing.preprocess(cropped_image)
                with torch.no_grad():
                    if prediction_done == False :
                        outputs = model.model(final_image)
                        probabilities = softmax(outputs)
                        max_probability, prediction = torch.max(probabilities, 1)
                        max_probability = max_probability.item()
                        prediction = prediction.item()
                        if max_probability > 0.8 :
                            prediction_list.append(prediction)
                        if len(prediction_list) > 30 :
                            del prediction_list[:-30]
                        if len(prediction_list) == 0 :
                            print("Low confidence")
                            continue
                        count = Counter(prediction_list)
                        most_predicted_number = count.most_common(1)[0][0]
                        count_most_predicted = count.most_common(1)[0][1]  
                        if count_most_predicted / len(prediction_list) * 100 > 80 and len(prediction_list) > 25 :
                            prediction_done = True
                            final_prediction = most_predicted_number
                            print(most_predicted_number)
                            print("Analyze finished")

                        else : 
                            print("Analyzing")
                    if final_prediction is not None : 
                        cv.putText(img = annoted_image, text = f"You draw {final_prediction}", org = (50,100), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (135,50,202), thickness = 2)

        else :
            annoted_image = draw_hand_result(rgb_frame, result)
            print('Hand detected : no prediction')
            prediction_list.clear()
            prediction_done = False
            final_prediction = None

        annoted_image = cv.cvtColor(annoted_image, cv.COLOR_RGB2BGR)
        cv.imshow('frame', annoted_image)

        if cv.waitKey(1) == ord('q'):
            break
    camera.release_video(video)

