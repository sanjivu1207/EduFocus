import cv2
import numpy as np
import pygame
import threading

def play_alert_sound():
    pygame.mixer.init()
    try:
        sound = pygame.mixer.Sound("alert.mp3")  
        sound.play()
    except pygame.error:
        print("Error: Could not load or play alert sound!")  

DISTRACTION_THRESHOLD = 3  
FOCUS_DROP_ALERT = 70  
FPS = 30  
distraction_timer = 0
alert_triggered = False
focused_frames = 100  
total_frames = 100  
scanner_pos = 0
scanner_direction = 1

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
if eye_cascade.empty():
    print("Warning: Default eye cascade not found, trying local backup.")
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye_tree_eyeglasses.xml')

print("Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    is_focused = False
    total_frames += 1  

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        scanner_pos += scanner_direction * 5
        if scanner_pos > h or scanner_pos < 0:
            scanner_direction *= -1  

        scanner_y = y + scanner_pos
        cv2.line(frame, (x, scanner_y), (x + w, scanner_y), (0, 255, 0), 3)


        eye_region_y1 = int(y + h * 0.2)  
        eye_region_y2 = int(y + h * 0.6)  
        roi_gray = gray[eye_region_y1:eye_region_y2, x:x + w]
        roi_color = frame[eye_region_y1:eye_region_y2, x:x + w]


        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            is_focused = True  


    if is_focused:
        focused_frames = min(focused_frames + 5, total_frames)  
        distraction_timer = 0  
        alert_triggered = False  
    else:
        focused_frames = max(0, focused_frames - 1)  
        distraction_timer += 1 / FPS  


        if distraction_timer > DISTRACTION_THRESHOLD and not alert_triggered:
            print("Stay focused!")
            threading.Thread(target=play_alert_sound).start()
            alert_triggered = True


    focus_percentage = min((focused_frames / total_frames) * 100, 100)
    cv2.putText(frame, f"Focus: {focus_percentage:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    if focus_percentage < FOCUS_DROP_ALERT and not alert_triggered:
        print(f"Focus dropped below {FOCUS_DROP_ALERT}%. Playing alert!")
        threading.Thread(target=play_alert_sound).start()
        alert_triggered = True


    cv2.imshow('Attention Monitor', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
