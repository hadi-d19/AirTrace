import cv2                     # OpenCV for computer vision tasks
import numpy as np            # NumPy for array and image operations
import mediapipe as mp        # MediaPipe for hand and face tracking
import pyautogui              # PyAutoGUI for screen control via eye tracking
from collections import deque # Deque for efficient append/pop from both ends (used for drawing)

# Initialize paint color buffers for each color
bpoints = [deque(maxlen=1024)]  # Blue
gpoints = [deque(maxlen=1024)]  # Green
rpoints = [deque(maxlen=1024)]  # Red
ypoints = [deque(maxlen=1024)]  # Yellow

# Index counters for each color
blue_index = green_index = red_index = yellow_index = 0

# List of BGR color values
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0                # Start with blue

mode = 'draw'                # Start in drawing mode

# Initialize a white canvas for drawing
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255

# Define UI button areas with labels and positions
buttons = [
    ("CLEAR", (40, 1, 140, 65)),
    ("BLUE", (160, 1, 255, 65)),
    ("GREEN", (275, 1, 370, 65)),
    ("RED", (390, 1, 485, 65)),
    ("YELLOW", (505, 1, 600, 65))
]

# Draw buttons on the canvas
for text, (x1, y1, x2, y2) in buttons:
    cv2.rectangle(paintWindow, (x1, y1), (x2, y2), (0, 0, 0), 2)
    cv2.putText(paintWindow, text, (x1 + 10, y1 + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Initialize MediaPipe hand and face mesh models
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils  # For drawing landmarks

# Get screen dimensions for eye tracking
screen_w, screen_h = pyautogui.size()

# Start video capture
cap = cv2.VideoCapture(0)

# Helper function to count fingers up (based on landmark positions)
def fingers_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]  # Indexes of fingertip landmarks
    fingers = []
    for tip in finger_tips:
        # A finger is considered up if its tip is higher than the lower joint
        fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)
    return fingers.count(True)  # Count how many fingers are up

# Main loop
while True:
    ret, frame = cap.read()       # Read frame from webcam
    if not ret:
        break                     # Break if no frame is read

    frame = cv2.flip(frame, 1)    # Flip frame horizontally for mirror view
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    result_hands = hands.process(frame_rgb)             # Process hand landmarks
    result_face = mp_face_mesh.process(frame_rgb)       # Process face landmarks
    frame_h, frame_w, _ = frame.shape                   # Get frame dimensions

    # If hands are detected
    if result_hands.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result_hands.multi_hand_landmarks, result_hands.multi_handedness):
            label = handedness.classification[0].label  # Left or Right hand

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)  # Draw hand connections

            # Control mode switching with left hand gestures
            if label == 'Left':
                if fingers_up(hand_landmarks) >= 4:
                    mode = 'eye'   # Switch to eye tracking mode
                elif fingers_up(hand_landmarks) <= 1:
                    mode = 'draw'  # Switch to drawing mode

            # Right hand controls drawing
            elif label == 'Right' and mode == 'draw':
                landmarks = []
                for lm in hand_landmarks.landmark:
                    lmx, lmy = int(lm.x * 640), int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                fore_finger = tuple(landmarks[8])  # Index finger tip
                thumb = tuple(landmarks[4])        # Thumb tip

                cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)  # Draw dot at fingertip

                # Start a new stroke when fingers are close (gesture to lift pen)
                if (thumb[1] - fore_finger[1]) < 30:
                    bpoints.append(deque(maxlen=512)); blue_index += 1
                    gpoints.append(deque(maxlen=512)); green_index += 1
                    rpoints.append(deque(maxlen=512)); red_index += 1
                    ypoints.append(deque(maxlen=512)); yellow_index += 1

                # If finger is near top buttons (within y <= 65)
                elif fore_finger[1] <= 65:
                    x = fore_finger[0]
                    if 40 <= x <= 140:  # Clear canvas
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]
                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :, :] = 255  # Clear drawing area
                    elif 160 <= x <= 255: colorIndex = 0  # Blue
                    elif 275 <= x <= 370: colorIndex = 1  # Green
                    elif 390 <= x <= 485: colorIndex = 2  # Red
                    elif 505 <= x <= 600: colorIndex = 3  # Yellow

                # Otherwise, draw with the selected color
                else:
                    if colorIndex == 0:
                        bpoints[blue_index].appendleft(fore_finger)
                    elif colorIndex == 1:
                        gpoints[green_index].appendleft(fore_finger)
                    elif colorIndex == 2:
                        rpoints[red_index].appendleft(fore_finger)
                    elif colorIndex == 3:
                        ypoints[yellow_index].appendleft(fore_finger)

    # If no hands, prepare to start new stroke
    else:
        if mode == 'draw':
            bpoints.append(deque(maxlen=512)); blue_index += 1
            gpoints.append(deque(maxlen=512)); green_index += 1
            rpoints.append(deque(maxlen=512)); red_index += 1
            ypoints.append(deque(maxlen=512)); yellow_index += 1

    # Eye tracking logic
    if result_face.multi_face_landmarks and mode == 'eye':
        landmarks = result_face.multi_face_landmarks[0].landmark

        # Use specific eye tracking landmarks
        for id, lm in enumerate(landmarks[474:478]):
            x, y = int(lm.x * frame_w), int(lm.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Move mouse using landmark 475
            if id == 1:
                screen_x = screen_w * lm.x
                screen_y = screen_h * lm.y
                pyautogui.moveTo(screen_x, screen_y)

        # Blink detection for click (distance between upper/lower eyelid)
        left_eye = [landmarks[145], landmarks[159]]
        if abs(left_eye[0].y - left_eye[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)  # Delay after click to prevent spamming

    # Redraw buttons on canvas (paintWindow) each loop
    for text, (x1, y1, x2, y2) in buttons:
        cv2.rectangle(paintWindow, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(paintWindow, text, (x1 + 10, y1 + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Draw lines on canvas from saved points
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    # Show current mode on the output frame
    cv2.putText(frame, f'Mode: {mode.upper()}', (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # Display the paint canvas and webcam frame
    cv2.imshow("Paint", paintWindow)
    cv2.imshow("Output", frame)

    # Exit on ESC or 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
