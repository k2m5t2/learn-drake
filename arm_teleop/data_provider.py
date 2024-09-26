# NOTE: caution - poorly written code.
#       refactor later, if needed.

import sqlite3
import time
import random
import mediapipe as mp
import cv2
import numpy as np

# Connect to SQLite database
conn = sqlite3.connect('shared_data.db')
c = conn.cursor()

# Create a table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS cube_positions
             (id INTEGER PRIMARY KEY, cube_id INTEGER, x REAL, y REAL, z REAL)''')

# Function to clear the table
def clear_table():
    c.execute("DELETE FROM cube_positions")
    conn.commit()

clear_table()

def provide_random_data(num_cubes=3):
# Insert random positions for cubes every second
    while True:
        for cube_id in range(num_cubes):  # Assuming 3 cubes
            x = random.uniform(-5, 5)
            y = random.uniform(-5, 5)
            z = random.uniform(-5, 5)
            c.execute("INSERT INTO cube_positions (cube_id, x, y, z) VALUES (?, ?, ?, ?)", (cube_id, x, y, z))
            conn.commit()
        print("Updated cube positions with random values.")
        time.sleep(1)

# utils/common

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def results_to_dict(results, type="world"):
    landmarks = []
    if type == "normalizd":
        data = results.pose_landmarks.landmark
    if type == "world":
        data = results.pose_world_landmarks.landmark
    for data_point in data:
        landmarks.append({
                            'X': data_point.x,
                            'Y': data_point.y,
                            'Z': data_point.z,
                            'Visibility': data_point.visibility,
                            })
    # print(landmarks) # DEBUG
    return landmarks

def provide_hands_only():
    left_hand_indices = [16, 18, 20, 22]
    right_hand_indices = [15, 17, 19, 21]

    num_cubes = 2

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks != None:
                landmarks = results_to_dict(results, type="world")
                # for cube_id in range(num_cubes):
                for cube_id in range(len(landmarks)):
                    if (cube_id == 16) or (cube_id == 15):
                    # if (cube_id in left_hand_indices) or (cube_id in right_hand_indices):
                        # x = landmarks[cube_id]['X']
                        # y = landmarks[cube_id]['Y']
                        # z = landmarks[cube_id]['Z']
                        x = landmarks[cube_id]['Z'] * -1
                        y = landmarks[cube_id]['X']
                        z = landmarks[cube_id]['Y'] * -1
                        c.execute("INSERT INTO cube_positions (cube_id, x, y, z) VALUES (?, ?, ?, ?)", (cube_id, x, y, z))
                        conn.commit()
                print("Updated cube positions with random values.")
                # time.sleep(1)
                time.sleep(0.1)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()

def provide_full_pose():
    num_cubes = 33

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks != None:
                landmarks = results_to_dict(results, type="world")
                for cube_id in range(num_cubes):
                    # x = landmarks[cube_id]['X']
                    # y = landmarks[cube_id]['Y']
                    # z = landmarks[cube_id]['Z']
                    x = landmarks[cube_id]['Z'] * -1
                    y = landmarks[cube_id]['X']
                    z = landmarks[cube_id]['Y'] * -1
                    c.execute("INSERT INTO cube_positions (cube_id, x, y, z) VALUES (?, ?, ?, ?)", (cube_id, x, y, z))
                    conn.commit()
                print("Updated cube positions with random values.")
                # time.sleep(1)
                time.sleep(0.5)
                # time.sleep(0.1)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


def main():
    # provide_random_data(3) # DEBUG
    # provide_hands_only()
    provide_full_pose()
    # NOTE: it's better to filter from the subscriber side, not provider side.

    # Close the connection (not reached in this loop)
    conn.close()

if __name__ == "__main__": main()

