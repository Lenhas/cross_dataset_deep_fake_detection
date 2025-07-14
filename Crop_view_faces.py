import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt


def zoom_and_crop_face(video_path):
    # Initialize the MTCNN face detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while cap.isOpened() and frame_count < 10:
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to RGB (as OpenCV loads BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and get bounding boxes
        boxes, _ = mtcnn.detect(rgb_frame)

        if boxes is not None and len(boxes) > 0:
            # Get the bounding box of the first detected face
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]

            # Ensure bounding box is within the image size
            x1, y1 = max(0, x1), max(0, y1)
            cropped_face = rgb_frame[y1:y2, x1:x2]

            # Optionally apply zoom (scaling the face)
            zoom_factor = 1.2
            width = x2 - x1
            height = y2 - y1
            new_w = int(width * zoom_factor)
            new_h = int(height * zoom_factor)
            resized_face = cv2.resize(cropped_face, (new_w, new_h))

            # Display the cropped and zoomed face
            plt.imshow(resized_face)
            plt.axis('off')
            plt.show()

            frame_count += 1
        else:
            print("No face detected in frame", frame_count)

    cap.release()


# Example usage:
video_path = 'C:/Users/José Marques/Desktop/celebDF/Celeb-synthesis/id3_id2_0001.mp4'
zoom_and_crop_face(video_path)



