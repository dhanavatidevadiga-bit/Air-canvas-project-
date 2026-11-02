import cv2
import mediapipe as mp
import numpy as np
import time
import os
import platform
import subprocess

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

save_folder = "My_Drawings"
os.makedirs(save_folder, exist_ok=True)

canvas = None
prev_x, prev_y = 0, 0

undo_stack = []
redo_stack = []
last_undo_time = 0 
colors = [
    (255, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 0)
]
color_names = ["PURPLE", "BLUE", "GREEN", "YELLOW", "ERASER"]
current_color = colors[0]

def fingers_state(hand):
    lm = hand.landmark
    fingers = []
    fingers.append(lm[4].x > lm[3].x)
    fingers.append(lm[8].y < lm[6].y)
    fingers.append(lm[12].y < lm[10].y)
    fingers.append(lm[16].y < lm[14].y)
    fingers.append(lm[20].y < lm[18].y)
    return fingers

def draw_palette(img):
    h, w, _ = img.shape
    box_w = w // len(colors)
    for i, col in enumerate(colors):
        x1 = i * box_w
        x2 = (i + 1) * box_w
        cv2.rectangle(img, (x1, 0), (x2, 60), col, -1)
        cv2.putText(img, color_names[i], (x1 + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def open_folder(path):
    # Works on Windows, Mac, Linux
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        print("Cannot open folder:", e)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    draw_palette(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode = "NONE"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        thumb, index, middle, ring, pinky = fingers_state(hand)

        x = int(hand.landmark[8].x * w)
        y = int(hand.landmark[8].y * h)

        # UNDO with cooldown
        if thumb and not index and not middle:
            if time.time() - last_undo_time > 0.8:
                if undo_stack:
                    redo_stack.append(canvas.copy())
                    canvas = undo_stack.pop()
                last_undo_time = time.time()
            mode = "UNDO"

        elif not thumb and not index and not middle and not ring and not pinky:
            if time.time() - last_undo_time > 0.8:
                if redo_stack:
                    undo_stack.append(canvas.copy())
                    canvas = redo_stack.pop()
                last_undo_time = time.time()
            mode = "REDO"

        elif index and middle and not ring:
            prev_x, prev_y = 0, 0
            mode = "SELECT"
            if y < 60:
                box_w = w // len(colors)
                idx = x // box_w
                if idx < len(colors):
                    current_color = colors[idx]
            cv2.circle(frame, (x, y), 15, current_color, cv2.FILLED)

        elif index and not middle:
            mode = "DRAW"
            cv2.circle(frame, (x, y), 8, current_color, cv2.FILLED)
            if prev_x == 0 and prev_y == 0:
                undo_stack.append(canvas.copy())
                redo_stack.clear()
                prev_x, prev_y = x, y
            thickness = 40 if current_color == (0, 0, 0) else 6
            cv2.line(canvas, (prev_x, prev_y), (x, y), current_color, thickness)
            prev_x, prev_y = x, y

        elif index and middle and ring:
            mode = "FILL"
            h2, w2 = canvas.shape[:2]
            mask = np.zeros((h2 + 2, w2 + 2), np.uint8)
            undo_stack.append(canvas.copy())
            redo_stack.clear()
            cv2.floodFill(canvas, mask, (x, y), current_color)

        else:
            prev_x, prev_y = 0, 0

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.putText(frame, f"Mode: {mode}", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame,
                "Index=Draw | Index+Middle=Select | 3 Fingers=Fill | Thumb=Undo | Fist=Redo | S=Save",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow("Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        filename = f"air_canvas_{int(time.time())}.png"
        filepath = os.path.join(save_folder, filename)
        cv2.imwrite(filepath, canvas)
        print(f"Drawing saved in {filepath}")
        open_folder(save_folder)

    if key == ord('c'):
        undo_stack.append(canvas.copy())
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
