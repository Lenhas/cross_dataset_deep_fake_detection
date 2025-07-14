import os
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.fftpack import fft2, fftshift
import matplotlib.pyplot as plt


def adjust_and_crop_bbox(frame, box, target_size=256):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    center_x, center_y = left + width / 2, top + height / 2

    # Adjust bounding box to focus more on the face, reduce unnecessary background
    expand_factor = 1.2
    new_width = width * expand_factor
    new_height = height * expand_factor

    # Determine new boundaries centered on the face with expanded area
    new_left = max(int(center_x - new_width / 2), 0)
    new_top = max(int(center_y - new_height / 2), 0)
    new_right = new_left + int(new_width)
    new_bottom = new_top + int(new_height)

    # Ensure the crop doesn't exceed the frame boundaries
    frame_width, frame_height = frame.size
    if new_right > frame_width:
        new_left = frame_width - int(new_width)
        new_right = frame_width
    if new_bottom > frame_height:
        new_top = frame_height - int(new_height)
        new_bottom = frame_height

    # Crop and resize to the target size
    cropped_face = frame.crop((new_left, new_top, new_right, new_bottom)).resize((target_size, target_size))
    return cropped_face


def detect_and_crop_face(frame, mtcnn, target_size=256):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    boxes, _ = mtcnn.detect(frame_pil)
    cropped_faces = []
    if boxes is not None:
        for box in boxes:
            face = adjust_and_crop_bbox(frame_pil, box, target_size=target_size)
            cropped_faces.append(face)
    return cropped_faces


# Function to convert image to frequency domain using Fourier transform
def convert_to_frequency_domain(image):
    if image.mode == 'RGB':
        image = image.convert('L')

    image_np = np.array(image)

    f_transform = fft2(image_np)
    f_transform_shifted = fftshift(f_transform)

    magnitude_spectrum = np.abs(f_transform_shifted)

    magnitude_spectrum = np.log1p(magnitude_spectrum)
    magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)

    return magnitude_spectrum


# Function to display cropped and FFT-transformed images
def display_images(cropped_image, fft_image):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(cropped_image)
    axes[0].set_title("Cropped Face")
    axes[0].axis("off")

    axes[1].imshow(fft_image, cmap='gray')
    axes[1].set_title("FFT-Transformed Image")
    axes[1].axis("off")

    plt.show()


def preprocess_and_save_videos(input_dir, output_dir, target_size=256):
    mtcnn = MTCNN(keep_all=False, post_process=False)
    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(input_dir) if f.endswith('.mp4')]
    for filename in tqdm(video_files, desc=f"Processing videos in {input_dir}"):
        video_path = os.path.join(input_dir, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video {filename}. Skipping.")
            continue

        frames_collected = []
        frame_idx = 0
        last_valid_frame = None

        # Skip frames until frame 260
        while frame_idx < 90:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video {filename} reached at frame {frame_idx}.")
                break
            frame_idx += 1

        # Process frames 260-270
        while 90 <= frame_idx <= 100:
            ret, frame = cap.read()
            if not ret:
                print(f"End of video {filename} reached at frame {frame_idx}.")
                break

            cropped_faces = detect_and_crop_face(frame, mtcnn, target_size=target_size)
            if cropped_faces:
                face = cropped_faces[0]
                last_valid_frame = face

                # Convert face to frequency domain
                frequency_features = convert_to_frequency_domain(face)

                # Display the cropped face and its FFT-transformed image
                #display_images(face, frequency_features)

                frames_collected.append(frequency_features)
            else:
                print(f"No faces detected in frame {frame_idx} of video {filename}.")
                if last_valid_frame is not None:
                    frequency_features = convert_to_frequency_domain(last_valid_frame)
                else:
                    blank_image = np.zeros((target_size, target_size), dtype=np.float32)
                    frequency_features = blank_image
                frames_collected.append(frequency_features)

            frame_idx += 1

        # If fewer frames are collected, pad with the last valid frame or blank frames
        while len(frames_collected) < 11:  # Since we want to process 260-270 (11 frames)
            if last_valid_frame is not None:
                frequency_features = convert_to_frequency_domain(last_valid_frame)
            else:
                blank_image = np.zeros((target_size, target_size), dtype=np.float32)
                frequency_features = blank_image
            frames_collected.append(frequency_features)

        video_array = np.stack(frames_collected[:11], axis=0)

        base_filename = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_filename}.npy")

        np.save(output_file, video_array)

        cap.release()


if __name__ == "__main__":
    real_input_dir = 'C:/Users/José Marques/Desktop/celebDF/Celeb-real'
    real_output_dir = 'C:/Users/José Marques/Desktop/celebDF/CelebRealFreq2'
    fake_input_dir = 'C:/Users/José Marques/Desktop/celebDF/Celeb-synthesis'
    fake_output_dir = 'C:/Users/José Marques/Desktop/celebDF/CelebFakeFreq2'

    preprocess_and_save_videos(
        input_dir=real_input_dir,
        output_dir=real_output_dir,
        target_size=256
    )

    preprocess_and_save_videos(
        input_dir=fake_input_dir,
        output_dir=fake_output_dir,
        target_size=256
    )
