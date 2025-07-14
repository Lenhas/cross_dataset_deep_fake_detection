import cv2
import dlib
import numpy as np
import os
import json
from matplotlib import pyplot as plt

def visualize_landmarks(cropped_face, landmarks):
    """Visualize landmarks on a cropped face image."""
    if landmarks is not None and len(landmarks) > 0:
        for (x, y) in landmarks:
            cv2.circle(cropped_face, (x, y), 1, (0, 255, 0), -1)
    plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def adjust_and_crop_bbox(rect, frame, expand_factor=1.2, target_size=None):
    """Adjusts the bounding box by an expand factor and crops the face from the frame."""
    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
    width, height = right - left, bottom - top
    center_x, center_y = left + width / 2, top + height / 2

    # Expand the bounding box
    new_width = width * expand_factor
    new_height = height * expand_factor

    new_left = int(max(center_x - new_width / 2, 0))
    new_top = int(max(center_y - new_height / 2, 0))
    new_right = int(new_left + new_width)
    new_bottom = int(new_top + new_height)

    # Ensure the new bounding box does not exceed frame boundaries
    frame_height, frame_width = frame.shape[:2]
    new_right = min(new_right, frame_width)
    new_bottom = min(new_bottom, frame_height)

    # Crop the face from the frame
    cropped_face = frame[new_top:new_bottom, new_left:new_right]

    # Resize if target_size is specified
    if target_size is not None:
        cropped_face = cv2.resize(cropped_face, target_size, interpolation=cv2.INTER_AREA)

    new_box = (new_left, new_top, new_right, new_bottom)
    return cropped_face, new_box

def extract_landmarks(frame, detector, predictor, expand_factor=1.2, target_size=None):
    """Extract facial landmarks for a given frame, with optional cropping."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    landmarks_mapped_list = []
    landmarks_relative_list = []
    cropped_faces = []
    boxes = []

    for rect in rects:
        cropped_face, new_box = adjust_and_crop_bbox(rect, frame, expand_factor, target_size)
        cropped_faces.append(cropped_face)
        boxes.append(new_box)

        cropped_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        cropped_rect = dlib.rectangle(0, 0, cropped_face.shape[1], cropped_face.shape[0])
        shape = predictor(cropped_gray, cropped_rect)
        landmarks_relative = np.array([[part.x, part.y] for part in shape.parts()], dtype=int)
        landmarks_relative_list.append(landmarks_relative)

        offset_x, offset_y = new_box[0], new_box[1]
        landmarks_mapped = landmarks_relative + np.array([offset_x, offset_y])
        landmarks_mapped_list.append(landmarks_mapped)

    return landmarks_mapped_list, landmarks_relative_list, cropped_faces, boxes

def process_video(video_path, detector, predictor, max_frames=None, expand_factor=1.2, target_size=None, visualize=True):
    """
    Processes a video to extract facial landmarks from frames 260 to 270 and visualize them on cropped faces.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return {}

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_landmarks = {}
    num_landmarks = 68  # dlib uses 68 landmark points

    # Skip frames until frame 260
    for i in range(90):
        ret, _ = cap.read()
        if not ret:
            print(f"Video ended before reaching frame 260 at frame {i}.")
            # Return a list of zeros for all skipped frames
            for j in range(i, 100):
                frame_landmarks[f'frame{j:03d}'] = [[0, 0] for _ in range(num_landmarks)]
            return frame_landmarks

    # Process frames 260-270
    for i in range(90, min(100, frame_count)):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i} could not be read. Stopping.")
            break

        landmarks_mapped_list, landmarks_relative_list, cropped_faces, boxes = extract_landmarks(
            frame, detector, predictor, expand_factor, target_size
        )

        # If no landmarks were detected, return a list of zeros
        if not landmarks_mapped_list:
            frame_landmarks[f'frame{i:03d}'] = [[0, 0] for _ in range(num_landmarks)]
        else:
            frame_landmarks[f'frame{i:03d}'] = [landmarks.tolist() for landmarks in landmarks_mapped_list]

        # Visualize landmarks on each cropped face
        if visualize:
            for cropped_face, landmarks_relative in zip(cropped_faces, landmarks_relative_list):
                visualize_landmarks(cropped_face.copy(), landmarks_relative)

    # Pad the remaining frames if the video ends before frame 270
    for j in range(i+1, 100):
        frame_landmarks[f'frame{j:03d}'] = [[0, 0] for _ in range(num_landmarks)]

    cap.release()
    return frame_landmarks

def main(video_folder, output_folder, shape_predictor_path, expand_factor=1.2, target_size=(256, 256), visualize=True):
    """
    Main function to process all videos in a folder and extract landmarks from frames 260-270.
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(video_folder, video_file)
            landmarks = process_video(
                video_path, detector, predictor,
                expand_factor=expand_factor,
                target_size=target_size,
                visualize=visualize
            )
            if landmarks:
                output_file = os.path.splitext(video_file)[0] + ".json"
                output_path = os.path.join(output_folder, output_file)

                with open(output_path, 'w') as f:
                    json.dump(landmarks, f)

                print(f"Processed {video_file} and saved landmarks to {output_file}")
            else:
                print(f"No landmarks extracted for {video_file}")

if __name__ == "__main__":
    real_input_dir = 'C:/Users/José Marques/Desktop/celebDF/Celeb-real'
    real_output_dir = 'C:/Users/José Marques/Desktop/celebDF/CelebReal2/landmarks'
    fake_input_dir = 'C:/Users/José Marques/Desktop/celebDF/Celeb-synthesis'
    fake_output_dir = 'C:/Users/José Marques/Desktop/celebDF/CelebFake2/landmarks'

    shape_predictor_path = 'shape_predictor_68_face_landmarks.dat'
    expand_factor = 1.2
    target_size = (256, 256)
    visualize = False

    main(real_input_dir, real_output_dir, shape_predictor_path, expand_factor, target_size, visualize)
    #main(fake_input_dir, fake_output_dir, shape_predictor_path, expand_factor, target_size, visualize)
