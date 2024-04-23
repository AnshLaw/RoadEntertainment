
import cv2
import pyautogui
import mediapipe as mp
import argparse
import subprocess

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

# Face model
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['2', '5', '10', '18', '25', '40', '50', '60+']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
padding=20
    


    
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,max_num_hands = 1,
    min_tracking_confidence=0.7) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
 
    image,faceBoxes=highlightFace(faceNet,image)
    if not faceBoxes:
        print("No face detected")
        
    for faceBox in faceBoxes:
        face=image[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, image.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[0]} years')

        cv2.putText(image, f' {age} years', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
    h, w, c = image.shape
    framergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:                #Hand Gesture Model
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            index_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            pinky_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            pinky_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
            ring_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            ring_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
            middle_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            middle_pip_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            thumb_tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            thumb_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
            thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
            thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
            for hand_lm in hand_landmarks.landmark:
                x, y = int(hand_lm.x * w), int(hand_lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if thumb_tip_y < thumb_mcp_y:
            pyautogui.press('volumeup')
            
        if thumb_tip_y > thumb_mcp_y:
            pyautogui.press('volumedown')
            
        if pinky_tip_y > pinky_pip_y and ring_tip_y > ring_pip_y and middle_tip_y < middle_pip_y and index_tip_y < index_pip_y and thumb_tip_x < thumb_mcp_x:
            zoom_path = "C:\\Users\\anshr\\AppData\\Roaming\\Zoom\\bin\\Zoom.exe"
            process = subprocess.Popen(zoom_path)
        
        if pinky_tip_y < pinky_pip_y and ring_tip_y > ring_pip_y and middle_tip_y > middle_pip_y and index_tip_y < index_pip_y and thumb_tip_x < thumb_mcp_x:
            board_path = "C:\\Program Files (x86)\\WellCraftedWhiteBoard\\WhiteBoard.exe"
            process1 = subprocess.Popen(board_path)
            
    cv2.imshow('Detection Window', image)
  
    
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()

