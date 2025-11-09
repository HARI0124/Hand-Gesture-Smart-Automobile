import cv2
import mediapipe as mp
import socket
import time
import os
from collections import deque

# ===============================
# ESP32 UDP Configuration
# ===============================
UDP_IP = "10.223.167.195"  # Replace with your ESP32 IP
CONTROL_PORT = 1234

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(0.05)

# ===============================
# Check Connection Function
# ===============================
def check_connection():
    response = os.system(f"ping -c 1 {UDP_IP} > /dev/null 2>&1")
    return response == 0

esp32_connected = check_connection()
print(f"{'✅' if esp32_connected else '❌'} ESP32 {'' if esp32_connected else 'not '}reachable at {UDP_IP}")

# ===============================
# MediaPipe Setup
# ===============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# ===============================
# Gesture Tracking
# ===============================
gesture_buffer = deque(maxlen=5)
manual_override = ""
motor_speed = 200
FPS_LIMIT = 15
FRAME_DELAY = 1.0 / FPS_LIMIT
last_sent_cmd = ""
last_sent_speed = -1
COMMAND_TIMEOUT = 0.25
last_command_time = 0
last_check_time = 0
CHECK_INTERVAL = 5  # seconds

colors = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255)]

# ===============================
# Gesture Classification
# ===============================
def classify_gesture(hand_landmarks):
    fingers = []
    tips = [4,8,12,16,20]
    fingers.append(hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x)
    for tip in tips[1:]:
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y)
    if all(fingers): return "FORWARD"
    if fingers[0] and not any(fingers[1:]): return "BACKWARD"
    if fingers[0] and fingers[1] and not any(fingers[2:]): return "LEFT"
    if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]: return "RIGHT"
    return "STOP"

# ===============================
# Speed Trackbar
# ===============================
def update_speed(x):
    global motor_speed
    motor_speed = int(x)

cv2.namedWindow("Hand Gesture Control", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Speed", "Hand Gesture Control", motor_speed, 255, update_speed)

# ===============================
# Main Loop
# ===============================
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    prev_time = 0
    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: continue

        # Auto-reconnect check every 5 seconds
        if time.time() - last_check_time > CHECK_INTERVAL:
            esp32_connected = check_connection()
            last_check_time = time.time()

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        command = "STOP"

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            command = classify_gesture(hand)
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            for idx, tip in enumerate([4,8,12,16,20]):
                lm = hand.landmark[tip]
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(frame, (cx, cy), 6, colors[idx], cv2.FILLED)

        # Manual Override
        k = cv2.waitKey(1) & 0xFF
        if k != 255:
            if k in (ord('w'), ord('W')): manual_override = "FORWARD"
            elif k in (ord('s'), ord('S')): manual_override = "BACKWARD"
            elif k in (ord('a'), ord('A')): manual_override = "LEFT"
            elif k in (ord('d'), ord('D')): manual_override = "RIGHT"
            elif k in (ord('x'), ord('X')): manual_override = "STOP"
            elif k == 27: break

        gesture_buffer.append(command)
        most_common = max(set(gesture_buffer), key=gesture_buffer.count)
        to_send = manual_override if manual_override else most_common

        # Send UDP Command
        if esp32_connected:
            current_time = time.time()
            if (to_send != last_sent_cmd or motor_speed != last_sent_speed or
                (current_time - last_command_time) > COMMAND_TIMEOUT):
                msg = f"CMD:{to_send} SPD:{motor_speed}"
                try:
                    sock.sendto(msg.encode(), (UDP_IP, CONTROL_PORT))
                    last_sent_cmd = to_send
                    last_sent_speed = motor_speed
                    last_command_time = current_time
                except OSError:
                    esp32_connected = False

        fps = 1 / (time.time() - prev_time) if prev_time else 0
        prev_time = time.time()

        cv2.putText(frame, f'Gesture: {most_common}', (10,40), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        cv2.putText(frame, f'Speed: {motor_speed}', (10,70), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv2.putText(frame, f'FPS: {int(fps)}', (10,100), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        cv2.putText(frame, f'ESP32: {"🟢 Connected" if esp32_connected else "🔴 Not Reachable"}',
                    (10,130), cv2.FONT_HERSHEY_SIMPLEX,0.7,
                    (0,255,0) if esp32_connected else (0,0,255), 2)

        cv2.imshow("Hand Gesture Control", frame)

        elapsed = time.time() - loop_start
        if FRAME_DELAY - elapsed > 0: time.sleep(FRAME_DELAY - elapsed)

cap.release()
cv2.destroyAllWindows()
