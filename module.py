import cv2
import mediapipe as mp
import pyautogui
import screen_brightness_control as sbc
import numpy as np
from math import hypot
from google.protobuf.json_format import MessageToDict
Draw = mp.solutions.drawing_utils
# Initializing the Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
min_detection_confidence = 0.75,
                           min_tracking_confidence = 0.75,
                                                     max_num_hands = 2)

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read video frame by frame
    success, img = cap.read()

    # Flip the image(frame)
    img = cv2.flip(img, 1)

    # Convert BGR image to RGB image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the RGB image
    results = hands.process(imgRGB)
    # If hands are present in image(frame)
    if results.multi_hand_landmarks:

        # Both Hands are present in image(frame)
        if len(results.multi_handedness) == 2:
            # Display 'Both Hands' on the image
            cv2.putText(img, 'Both Hands are there', (250, 50),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.9, (0, 255, 0), 2)

        # If any hand present
        else:
            for i in results.multi_handedness:

                # Return whether it is Right or Left Hand
                label = MessageToDict(i)['classification'][0]['label']

                if label == 'Left':
                    landmarkList = []
                    # if hands are present in image(frame)
                    if results.multi_hand_landmarks:
                        # detect handmarks
                        for handlm in results.multi_hand_landmarks:
                            for _id, landmarks in enumerate(handlm.landmark):
                                # store height and width of image
                                height, width, color_channels = img.shape

                                # calculate and append x, y coordinates
                                # of handmarks from image(frame) to lmList
                                x, y = int(landmarks.x * width), int(landmarks.y * height)
                                landmarkList.append([_id, x, y])

                            # draw Landmarks
                            Draw.draw_landmarks(img, handlm,
                                                mpHands.HAND_CONNECTIONS)

                    # If landmarks list is not empty
                    if landmarkList != []:
                        # store x,y coordinates of (tip of) thumb
                        x_1, y_1 = landmarkList[4][1], landmarkList[4][2]

                        # store x,y coordinates of (tip of) index finger
                        x_2, y_2 = landmarkList[8][1], landmarkList[8][2]

                        # draw circle on thumb and index finger tip
                        cv2.circle(img, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
                        cv2.circle(img, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)

                        # draw line from tip of thumb to tip of index finger
                        cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

                        # calculate square root of the sum of
                        # squares of the specified arguments.
                    L = hypot(x_2 - x_1, y_2 - y_1)
                    if L>50:
                        pyautogui.press("volumeup")
                    else:
                        pyautogui.press("volumedown")

                if label == 'Right':
                        landmarkList = []
                        if results.multi_hand_landmarks:
                            for handlm in results.multi_hand_landmarks:
                                for _id, landmarks in enumerate(handlm.landmark):
                                    height, width, color_channels = img.shape
                                    x, y = int(landmarks.x * width), int(landmarks.y * height)
                                    landmarkList.append([_id, x, y])
                                Draw.draw_landmarks(img, handlm,
                                                    mpHands.HAND_CONNECTIONS)
                        if landmarkList != []:
                            x_1, y_1 = landmarkList[4][1], landmarkList[4][2]
                            x_2, y_2 = landmarkList[8][1], landmarkList[8][2]
                            cv2.circle(img, (x_1, y_1), 7, (0, 255, 0), cv2.FILLED)
                            cv2.circle(img, (x_2, y_2), 7, (0, 255, 0), cv2.FILLED)
                            cv2.line(img, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)
                            L = hypot(x_2 - x_1, y_2 - y_1)

                            # 1-D linear interpolant to a function
                            # with given discrete data points
                            # (Hand range 15 - 220, Brightness
                            # range 0 - 100), evaluated at length.
                            b_level = np.interp(L, [15, 220], [0, 100])

                            # set brightness
                            sbc.set_brightness(int(b_level))
    #press q to quit the program
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
