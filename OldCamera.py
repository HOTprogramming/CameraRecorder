import cv2
import os
from datetime import datetime
from networktables import NetworkTables
import time
from Utils import RingBuffer
import threading
import tkinter as tk

test = False
    
def main():
    # if(test):
    #     NetworkTables.initialize(server="127.0.0.1")
    # else:
    NetworkTables.initialize(server="10.0.67.2")

    sd = NetworkTables.getTable("SmartDashboard")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        cap.release()
        return

    frame_height, frame_width = frame.shape[:2]

    file_name = f"output_{datetime.now().timestamp()}.mp4"
    output = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    buffer_size = 100
    ring_buffer = RingBuffer(buffer_size)

    recording = False

    frames = []

    prevRecording = False

    startTime = time.time()

    manual = False

    buffer = False
    prevBuffer = False

    previous_time = 0

    print("Press 'q' to quit.")
    while True:
        
        en = sd.getBoolean("Teleop", False)

        if not manual:
            recording = en
        else:
            en = recording

        key = cv2.waitKey(1)

        if key & 0xFF == ord('r'):
            recording = True
            manual = True
        if key & 0xFF == ord('s'):
            recording = False
            manual = False

        ret, mainFrame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)
        thickness = 2
        position = (50, 50)

        cv2.putText(mainFrame, str(en), position, font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(mainFrame, str(time.time() - startTime), (500, 50), font, font_scale, color, thickness, cv2.LINE_AA)

        if recording:
            frames.append(mainFrame)
        else:
            if buffer and (time.time() - bufferTime):
                frames.append(mainFrame)

                if recording:
                    buffer = False

                if time.time() - bufferTime > 2:
                    buffer = False
            else:
                ring_buffer.add_frame(mainFrame)

        if recording == True and prevRecording != recording:           
            startTime = time.time()
        
            frames = ring_buffer.get_frames()

            print("Starting Recording")

        cv2.imshow("Camera Feed", mainFrame)

        current_time = time.time()
        # Calculate FPS for the last frame processed
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        print(fps)

        if recording == False and prevRecording != recording:
            print("Ending Recording")

            buffer = True
            bufferTime = time.time()

        if buffer == False and prevBuffer != buffer:
            thread = threading.Thread(target=writeandplay, args=(frames, output, file_name), daemon=True)
            thread.start()

            frames = []

            file_name = f"output_{datetime.now().timestamp()}.mp4"
            output = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
            
        if key & 0xFF == ord('q'):
            break

        prevRecording = recording
        prevBuffer = buffer

    cap.release()   
    output.release()
    cv2.destroyAllWindows()

    os.remove(file_name)


def writeandplay(frames, output, file_name):
    for frame in frames:
        output.write(frame)

    output.release()
    os.startfile(file_name)


if __name__ == "__main__":
    main()