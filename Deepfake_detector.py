import json
import logging
import os

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

from triplet_attention import TripletAttention

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = {
    "checkpoint_filename": "Deepfake2_MHA_model_git_checkpoint_epoch_{epoch}.pth.tar",
    "metrics_filename": "Deepfake2_MHA_model_git_training_metrics_epoch_{epoch}.json",
    "model_filename": "Deepfake2_MHA_model_git_epoch_{epoch}.pth",
}


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Args:
        batch (list): List of tuples (videos, freq_videos, landmarks, labels, video_paths)

    Returns:
        tuple: Tuple of stacked tensors (videos, freq_videos, landmarks, labels, video_paths)
    """
    videos, freq_videos, landmarks, labels, video_paths = zip(*batch)
    videos = torch.stack(videos)
    freq_videos = torch.stack(freq_videos)
    landmarks = torch.stack(landmarks)
    labels = torch.tensor(labels)
    return videos, freq_videos, landmarks, labels, video_paths


def calculate_eer(targets, scores):
    """
    Calculate Equal Error Rate (EER) for binary classification.

    Args:
        targets (list or np.array): Ground truth binary labels (0 or 1).
        scores (list or np.array): Predicted scores or probabilities for the positive class.

    Returns:
        float: Equal Error Rate (EER) value between 0 and 1.
    """
    fpr, tpr, thresholds = roc_curve(targets, scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FPR and FNR are closest
    idx_eer = np.nanargmin(np.absolute((fpr - fnr)))
    eer = fpr[idx_eer]
    return eer


def convert_to_3_channels(data):
    """
    Convert single-channel data to 3-channel data by repeating along the channel dimension.

    Args:
        data (np.array): Input data of shape (timesteps, H, W)

    Returns:
        np.array: Output data of shape (timesteps, 3, H, W)
    """
    return np.repeat(data[:, np.newaxis, :, :], 3, axis=1)


def calculate_landmark_features(coords, expected_size=4556, normalize=True):
    """
    Calculate landmark features based on Euclidean distances and angles between landmark points.

    Args:
        coords (np.array): Landmark coordinates of shape (68, 2).
        expected_size (int): Expected size of the output feature vector.
        normalize (bool): Whether to normalize the features.

    Returns:
        np.array: Feature vector of size 'expected_size'.
    """
    # Ensure that coords have the shape (68, 2)
    coords = np.squeeze(coords)
    logger.debug(f"Coordinates after squeezing: {coords.shape}")

    if coords.shape[0] != 68 or np.all(coords == 0):
        logger.debug(f"Coordinates shape mismatch or all zeros: {coords.shape}")
        return np.zeros(expected_size)

    # Compute the differences between each pair of points and calculate distances
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
    ##print(f"Differences shape: {diffs.shape}, Distances shape: {distances.shape}")

    # Normalize the coordinates to get angles between points
    norms = np.linalg.norm(coords, axis=1)
    norm_coords = coords / norms[:, np.newaxis]
    cosine_similarity = np.dot(norm_coords, norm_coords.T)
    cosine_similarity = np.clip(cosine_similarity, -1, 1)
    #print(f"cos sim: {cosine_similarity}")
    angles = np.arccos(cosine_similarity) * (180 / np.pi)
    logger.debug(
        f"Cosine similarity shape: {cosine_similarity.shape}, Angles shape: {angles.shape}"
    )
    #print(f"angles {angles}")
    # Extract upper triangular indices to avoid duplicate calculations
    triu_indices = np.triu_indices_from(distances, k=1)
    euclidean_distances = distances[triu_indices]
    upper_triangular_angles = angles[triu_indices]
    logger.debug(
        f"Euclidean distances shape: {euclidean_distances.shape}, "
        f"Upper triangular angles shape: {upper_triangular_angles.shape}"
    )

    # Concatenate distances and angles into a single feature vector
    features = np.concatenate([euclidean_distances, upper_triangular_angles])
    logger.debug(f"Concatenated features size: {features.size}")

    # Normalize the features if requested
    if normalize:
        min_val = features.min() if features.size > 0 else 0
        max_val = features.max() if features.size > 0 else 1
        if max_val - min_val > 0:
            features = (features - min_val) / (max_val - min_val)  # Min-max normalization
        else:
            features = np.zeros_like(features)

    return features


class EfficientNetFeatureExtractor(nn.Module):
    """
    EfficientNet-based feature extractor that returns specified layers' outputs.
    """

    def __init__(self, base_model, return_nodes):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.feature_extractor = create_feature_extractor(
            base_model, return_nodes=return_nodes
        )

    def forward(self, x):
        outputs = self.feature_extractor(x)
        # Log the shapes of extracted features
        for name, output in outputs.items():
            logger.debug(f"EfficientNetFeatureExtractor - {name} output shape: {output.shape}")
        return outputs


class PositionalEncoding(nn.Module):
    """
    Implements the Positional Encoding as described in the Transformer paper.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependent on pos and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, ignore the last term of cos
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, d_model)
        x = x + self.pe[:, : x.size(1)]
        logger.debug(f"PositionalEncoding - output shape: {x.shape}")
        return x


class GatedMultimodalUnit(nn.Module):
    """
    Implements the Gated Multimodal Unit (GMU) for combining two modalities.
    """

    def __init__(self, input_dim):
        super(GatedMultimodalUnit, self).__init__()
        self.input_dim = input_dim
        self.fc_z = nn.Linear(input_dim * 2, input_dim)
        self.fc_h = nn.Linear(input_dim * 2, input_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        """
        Forward pass for the GMU.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, timesteps, feature_dim)
            y (torch.Tensor): Tensor of shape (batch_size, timesteps, feature_dim)

        Returns:
            torch.Tensor: Combined features of shape (batch_size, timesteps, feature_dim)
        """
        combined = torch.cat((x, y), dim=-1)
        z = self.sigmoid(self.fc_z(combined))
        h_tilde = self.tanh(self.fc_h(combined))
        h = z * h_tilde + (1 - z) * x
        return h



class TripleGatedMultimodalUnit(nn.Module):
    """
    Implements the Gated Multimodal Unit for combining three modalities.
    """

    def __init__(self, input_dim):
        super(TripleGatedMultimodalUnit, self).__init__()
        self.input_dim = input_dim

        # Gate weights
        self.fc_z1 = nn.Linear(input_dim * 3, 1)
        self.fc_z2 = nn.Linear(input_dim * 3, 1)
        self.fc_z3 = nn.Linear(input_dim * 3, 1)

        # Transformations for each modality
        self.fc_h1 = nn.Linear(input_dim, input_dim)
        self.fc_h2 = nn.Linear(input_dim, input_dim)
        self.fc_h3 = nn.Linear(input_dim, input_dim)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x1, x2, x3):
        """
        Forward pass for the Triple GMU.

        Args:
            x1, x2, x3 (torch.Tensor): Tensors of shape (batch_size, timesteps, input_dim)

        Returns:
            torch.Tensor: Combined features of shape (batch_size, timesteps, input_dim)
        """
        # Transform each modality
        h1 = self.tanh(self.fc_h1(x1))
        h2 = self.tanh(self.fc_h2(x2))
        h3 = self.tanh(self.fc_h3(x3))

        # Compute gates
        combined = torch.cat((x1, x2, x3), dim=-1)
        z1 = self.sigmoid(self.fc_z1(combined)).squeeze(-1)
        z2 = self.sigmoid(self.fc_z2(combined)).squeeze(-1)
        z3 = self.sigmoid(self.fc_z3(combined)).squeeze(-1)

        # Combine the features using the gates
        h = z1.unsqueeze(-1) * h1 + z2.unsqueeze(-1) * h2 + z3.unsqueeze(-1) * h3
        return h


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class VideoClassifier(nn.Module):
    """
    Video Classifier model that combines spatial, frequency, and landmark features
    for deepfake detection.
    """

    def __init__(self, num_classes, feature_dim=512, num_heads=8, num_layers=2):
        super(VideoClassifier, self).__init__()

        self.feature_dim = feature_dim

        # Separate EfficientNet backbones for spatial and frequency features
        base_model_spatial = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        base_model_freq = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        return_nodes = {'features.4': 'low_level', 'features.7': 'high_level'}

        self.spatial_feature_extractor = EfficientNetFeatureExtractor(
            base_model_spatial, return_nodes
        )
        self.freq_feature_extractor = EfficientNetFeatureExtractor(
            base_model_freq, return_nodes
        )

        # Separate TripletAttention modules
        self.triplet_attention_low_spatial = TripletAttention(kernel_size=5)
        self.triplet_attention_high_spatial = TripletAttention(kernel_size=5)
        self.triplet_attention_low_freq = TripletAttention(kernel_size=5)
        self.triplet_attention_high_freq = TripletAttention(kernel_size=5)

        # 1x1 convolutions for feature reduction
        self.feature_reduction_low_spatial = nn.Conv2d(80, feature_dim, kernel_size=1)
        self.feature_reduction_high_spatial = nn.Conv2d(320, feature_dim, kernel_size=1)
        self.feature_reduction_low_freq = nn.Conv2d(80, feature_dim, kernel_size=1)
        self.feature_reduction_high_freq = nn.Conv2d(320, feature_dim, kernel_size=1)

        # Assume landmark_feature_size is the size of features output by calculate_landmark_features
        landmark_feature_size = 4556

        self.landmark_feature_projection = nn.Linear(landmark_feature_size, feature_dim)

        # Binary GMUs to combine low and high-level features
        self.gmu_spatial = GatedMultimodalUnit(feature_dim)
        self.gmu_freq = GatedMultimodalUnit(feature_dim)

        # Fusion using Triple Gated Multimodal Unit (Triple GMU)
        self.triple_gmu = TripleGatedMultimodalUnit(feature_dim)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(feature_dim)

        # Transformer Encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=2048,
            dropout=0.2,  # Increased dropout rate
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Final classification layer
        self.fc = nn.Sequential(nn.Linear(feature_dim, num_classes))

    def forward(self, x, freq_videos, landmark_features):
        """
        Forward pass of the VideoClassifier.

        Args:
            x (torch.Tensor): Input video frames of shape (batch_size, timesteps, C, H, W).
            freq_videos (torch.Tensor): Frequency domain video frames of shape (batch_size, timesteps, C, H, W).
            landmark_features (torch.Tensor): Landmark features of shape (batch_size, timesteps, feature_size).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        logging.info(f"Input shape (spatial): {x.shape}")
        logging.info(f"Input shape (frequency): {freq_videos.shape}")
        logging.info(f"Input shape (landmarks): {landmark_features.shape}")

        batch_size, timesteps, C, H, W = x.size()
        # Reshape inputs for feature extraction
        x = x.view(batch_size * timesteps, C, H, W)
        freq_videos = freq_videos.view(batch_size * timesteps, C, H, W)
        logging.info(f"Reshaped spatial: {x.shape}, Reshaped frequency: {freq_videos.shape}")

        # Extract low and high-level features for spatial input
        spatial_features = self.spatial_feature_extractor(x)
        low_level_spatial = spatial_features['low_level']
        high_level_spatial = spatial_features['high_level']
        logging.info(f"Spatial low-level shape: {low_level_spatial.shape}, high-level shape: {high_level_spatial.shape}")

        # Extract low and high-level features for frequency input
        freq_features = self.freq_feature_extractor(freq_videos)
        low_level_freq = freq_features['low_level']
        high_level_freq = freq_features['high_level']
        logging.info(f"Frequency low-level shape: {low_level_freq.shape}, high-level shape: {high_level_freq.shape}")

        # Apply Triplet Attention separately
        low_level_spatial = self.triplet_attention_low_spatial(low_level_spatial)
        high_level_spatial = self.triplet_attention_high_spatial(high_level_spatial)
        low_level_freq = self.triplet_attention_low_freq(low_level_freq)
        high_level_freq = self.triplet_attention_high_freq(high_level_freq)

        # Feature reduction
        low_level_spatial = self.feature_reduction_low_spatial(low_level_spatial)
        high_level_spatial = self.feature_reduction_high_spatial(high_level_spatial)
        low_level_freq = self.feature_reduction_low_freq(low_level_freq)
        high_level_freq = self.feature_reduction_high_freq(high_level_freq)

        # Adaptive pooling
        low_level_spatial = F.adaptive_avg_pool2d(low_level_spatial, (1, 1)).view(
            batch_size, timesteps, -1
        )
        high_level_spatial = F.adaptive_avg_pool2d(high_level_spatial, (1, 1)).view(
            batch_size, timesteps, -1
        )
        low_level_freq = F.adaptive_avg_pool2d(low_level_freq, (1, 1)).view(
            batch_size, timesteps, -1
        )
        high_level_freq = F.adaptive_avg_pool2d(high_level_freq, (1, 1)).view(
            batch_size, timesteps, -1
        )
        logging.info(f"Shapes after pooling - Low spatial: {low_level_spatial.shape}, High spatial: {high_level_spatial.shape}, Low freq: {low_level_freq.shape}, High freq: {high_level_freq.shape}")

        # Combine low and high-level features using binary GMUs
        combined_spatial = self.gmu_spatial(low_level_spatial, high_level_spatial)
        combined_freq = self.gmu_freq(low_level_freq, high_level_freq)
        logging.info(f"Combined spatial shape: {combined_spatial.shape}, Combined freq shape: {combined_freq.shape}")

        # Process landmark features
        landmark_features = self.landmark_feature_projection(landmark_features)
        logging.info(f"Projected landmark features shape: {landmark_features.shape}")

        # Fusion using Triple GMU
        fused_features = self.triple_gmu(
            combined_spatial, combined_freq, landmark_features
        )
        logging.info(f"Fused features shape: {fused_features.shape}")

        # Apply positional encoding to fused features
        fused_features = self.positional_encoding(fused_features)
        logging.info(f"Fused features after positional encoding shape: {fused_features.shape}")

        # Temporal Modeling with Transformer Encoder
        temporal_features = self.temporal_transformer(fused_features)
        logging.info(f"Temporal features shape after transformer: {temporal_features.shape}")

        # Take the mean over time
        temporal_features = temporal_features.mean(dim=1)  # Shape: (batch_size, feature_dim)
        logging.info(f"Temporal features shape after mean pooling: {temporal_features.shape}")

        # Final classification
        out = self.fc(temporal_features)  # Shape: (batch_size, num_classes)
        logging.info(f"Output shape: {out.shape}")

        return out



class VideoDataset(Dataset):
    """
    Custom Dataset for loading video data along with frequency domain data and landmark features.
    """

    def __init__(
        self,
        real_dir,
        fake_dir,
        freq_dir_real,
        freq_dir_fake,
        landmark_dir_real,
        landmark_dir_fake,
        transform=None,
        frames_per_video=10,
        frame_start=0,
    ):
        self.real_videos = [
            (
                os.path.join(real_dir, f),
                0,
                os.path.join(landmark_dir_real, os.path.splitext(f)[0] + '.json'),
                os.path.join(freq_dir_real, f"{os.path.splitext(f)[0]}.npy"),
            )
            for f in os.listdir(real_dir)
            if f.endswith('.npy')
        ]
        logger.info(f"Loaded {len(self.real_videos)} real videos")

        self.fake_videos = [
            (
                os.path.join(fake_dir, f),
                1,
                os.path.join(landmark_dir_fake, os.path.splitext(f)[0] + '.json'),
                os.path.join(freq_dir_fake, f"{os.path.splitext(f)[0]}.npy"),
            )
            for f in os.listdir(fake_dir)
            if f.endswith('.npy')
        ]
        logger.info(f"Loaded {len(self.fake_videos)} fake videos")

        # Combine real and fake videos into one list
        self.videos = self.real_videos + self.fake_videos
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.frame_start = frame_start
        logger.info(f"Total videos: {len(self.videos)}")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path, label, landmark_path, freq_path = self.videos[idx]
        logger.debug(f"Processing video {video_path}, label: {label}")

        # Load the video and frequency data
        video_data = np.load(video_path).astype(np.float32) / 255.0  # Normalize the data
        logger.debug(f"Loaded video data of shape: {video_data.shape}")

        freq_data = np.load(freq_path).astype(np.float32)
        logger.debug(f"Loaded frequency data of shape: {freq_data.shape}")

        # Convert 1-channel frequency feature into 3-channel format
        freq_data_3_channel = convert_to_3_channels(freq_data)
        logger.debug(
            f"Converted frequency data to 3 channels, new shape: {freq_data_3_channel.shape}"
        )

        # Load the landmark data from the .json file
        logger.debug(f"Loading landmark data from {landmark_path}")
        with open(landmark_path, 'r') as f:
            landmark_data = json.load(f)

        # Dynamic frame start for training and testing
        frames, freq_frames, landmark_features = [], [], []

        for i in range(self.frames_per_video):
            video_idx = self.frame_start + i if self.frame_start > 0 else i

            # Load frames from video and frequency data
            if video_idx < video_data.shape[0]:
                frame = video_data[video_idx]
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            else:
                frames.append(torch.zeros(3, 256, 256, dtype=torch.float32))

            freq_frame = (
                torch.from_numpy(freq_data_3_channel[video_idx]).float()
                if video_idx < freq_data_3_channel.shape[0]
                else torch.zeros(3, 256, 256, dtype=torch.float32)
            )
            freq_frames.append(freq_frame)

            # Create the key dynamically for landmarks, formatted as 'frameXXX' with zero padding
            frame_key = f'frame{video_idx:03d}'
            landmarks = np.array(
                landmark_data.get(frame_key, np.zeros((68, 2), dtype=int))
            )
            features = calculate_landmark_features(landmarks)
            landmark_features.append(features)

        # Pad frames, frequency frames, and landmark features if necessary
        frames += [torch.zeros_like(frames[0])] * (self.frames_per_video - len(frames))
        freq_frames += [torch.zeros_like(freq_frames[0])] * (
            self.frames_per_video - len(freq_frames)
        )
        landmark_features += [np.zeros_like(landmark_features[0])] * (
            self.frames_per_video - len(landmark_features)
        )

        logger.debug(
            f"Final frame shape: {torch.stack(frames).shape}, "
            f"frequency frame shape: {torch.stack(freq_frames).shape}, "
            f"landmark feature shape: {np.stack(landmark_features).shape}"
        )

        return (
            torch.stack(frames).float(),
            torch.stack(freq_frames).float(),
            torch.tensor(np.stack(landmark_features), dtype=torch.float32),
            torch.tensor(label),
            video_path,
        )
def create_data_loaders(
    data_dir_real,
    data_dir_fake,
    freq_dir_real,
    freq_dir_fake,
    landmark_dir_real,
    landmark_dir_fake,
    transform,
    batch_size,
    val_split,
    max_train_samples=None,
):
    """
    Create DataLoaders for training and validation datasets.

    Args:
        data_dir_real (str): Directory containing real videos.
        data_dir_fake (str): Directory containing fake videos.
        freq_dir_real (str): Directory containing frequency data for real videos.
        freq_dir_fake (str): Directory containing frequency data for fake videos.
        landmark_dir_real (str): Directory containing landmarks for real videos.
        landmark_dir_fake (str): Directory containing landmarks for fake videos.
        transform (callable): Transformation to apply to the video frames.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of data to use for validation.
        max_train_samples (int, optional): Maximum number of training samples.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Initialize the dataset
    dataset = VideoDataset(
        data_dir_real,
        data_dir_fake,
        freq_dir_real,
        freq_dir_fake,
        landmark_dir_real,
        landmark_dir_fake,
        transform,
        frame_start=0,
    )

    # Separate real and fake videos
    real_videos = dataset.real_videos
    fake_videos = dataset.fake_videos

    # Shuffle real and fake videos separately to maintain balance
    np.random.seed(42)
    np.random.shuffle(real_videos)
    np.random.shuffle(fake_videos)

    # Split into validation and training sets
    val_size_real = int(len(real_videos) * val_split)
    val_size_fake = int(len(fake_videos) * val_split)

    val_videos = real_videos[:val_size_real] + fake_videos[:val_size_fake]
    train_videos = real_videos[val_size_real:] + fake_videos[val_size_fake:]

    # Shuffle training videos again after separating for validation
    np.random.shuffle(train_videos)

    # Ensure balanced training data if max_train_samples is provided
    if max_train_samples is not None:
        half_train_samples = max_train_samples // 2
        train_videos = (
            train_videos[:half_train_samples]
            + fake_videos[val_size_fake : val_size_fake + half_train_samples]
        )

    # Create training and validation subsets
    train_indices = [dataset.videos.index(video) for video in train_videos]
    val_indices = [dataset.videos.index(video) for video in val_videos]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def create_test_loader(
    data_dir_real,
    data_dir_fake,
    freq_dir_real,
    freq_dir_fake,
    landmark_dir_real,
    landmark_dir_fake,
    transform,
    batch_size,
):
    """
    Create DataLoader for the test dataset.

    Args:
        data_dir_real (str): Directory containing real videos.
        data_dir_fake (str): Directory containing fake videos.
        freq_dir_real (str): Directory containing frequency data for real videos.
        freq_dir_fake (str): Directory containing frequency data for fake videos.
        landmark_dir_real (str): Directory containing landmarks for real videos.
        landmark_dir_fake (str): Directory containing landmarks for fake videos.
        transform (callable): Transformation to apply to the video frames.
        batch_size (int): Batch size for DataLoader.

    Returns:
        DataLoader: Test DataLoader.
    """
    # Initialize the dataset
    dataset = VideoDataset(
        data_dir_real,
        data_dir_fake,
        freq_dir_real,
        freq_dir_fake,
        landmark_dir_real,
        landmark_dir_fake,
        transform,
        frame_start=0,
    )

    # Total number of videos in the combined dataset
    total_len = len(dataset)

    # Generate test indices (which will include all data)
    test_indices = list(range(total_len))

    # Create test loader using custom collate_fn
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
    )

    return test_loader


def save_top_mistakes(top_mistakes, filename):
    """
    Save the top mistakes to a JSON file.

    Args:
        top_mistakes (list): List of top mistake dictionaries.
        filename (str): Path to the file where the top mistakes will be saved.
    """
    with open(filename, 'w') as f:
        json.dump(top_mistakes, f, indent=4)


def test_model(model, test_loader, criterion, device, mistakes_filename='top_mistakes.json'):
    """
    Test the model on the test dataset.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for computation.
        mistakes_filename (str): Filename to save the top mistakes.

    Returns:
        dict: Dictionary containing test metrics.
    """
    logger.info("Testing the model on the test dataset...")

    total_loss = 0
    correct = 0
    total = 0
    predictions, targets, softmax_outputs, confidences = [], [], [], []
    mistakes = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc="Testing", unit="batch")

        for idx, (videos, freq_videos, landmarks, labels, video_paths) in enumerate(test_loader):
            videos, freq_videos, landmarks, labels = (
                videos.to(device),
                freq_videos.to(device),
                landmarks.to(device),
                labels.to(device),
            )

            outputs = model(videos, freq_videos, landmarks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            softmax_output = F.softmax(outputs, dim=1)
            softmax_outputs.extend(softmax_output[:, 1].cpu().numpy())

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            confidences.extend(softmax_output.max(dim=1)[0].cpu().numpy())

            # Track mistakes (incorrect predictions)
            for i in range(labels.size(0)):
                if predicted[i] != labels[i]:
                    mistakes.append(
                        {
                            'video_path': video_paths[i],
                            'predicted': predicted[i].item(),
                            'actual': labels[i].item(),
                            'confidence': softmax_output[i].max().item(),
                        }
                    )

            pbar.update(1)
        pbar.close()

    # Sort mistakes by confidence in descending order
    top_mistakes = sorted(mistakes, key=lambda x: x['confidence'], reverse=True)[:5]

    # Log top mistakes
    for mistake in top_mistakes:
        logger.info(
            f"Top mistake: Video Path {mistake['video_path']}, "
            f"Predicted {mistake['predicted']}, Actual {mistake['actual']}, "
            f"Confidence {mistake['confidence']:.4f}"
        )

    # Save top mistakes to a JSON file
    save_top_mistakes(top_mistakes, mistakes_filename)
    logger.info(f"Top mistakes saved to {mistakes_filename}")

    # Compute accuracy and other metrics
    accuracy = 100.0 * correct / total if total != 0 else 0
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)

    # Compute AUC
    if len(np.unique(targets)) == 2:
        auc = roc_auc_score(targets, softmax_outputs)
        eer = calculate_eer(targets, softmax_outputs)
    else:
        auc = roc_auc_score(targets, softmax_outputs, multi_class="ovr")
        eer = None

    error_rate = 1 - (correct / total) if total != 0 else 0
    balanced_acc = balanced_accuracy_score(targets, predictions)

    # Logging the metrics
    logger.info(
        f"Test - Loss: {total_loss / len(test_loader):.4f}, "
        f"Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, "
        f"Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}, "
        f"EER: {eer if eer is not None else 'N/A'}, Error Rate: {error_rate:.2f}, "
        f"Balanced Accuracy: {balanced_acc:.2f}"
    )

    return {
        'test_loss': total_loss / len(test_loader),
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_auc': auc,
        'test_eer': eer,
        'test_error_rate': error_rate,
        'test_balanced_accuracy': balanced_acc,
        'test_predictions': predictions,
        'test_targets': targets,
        'top_mistakes': top_mistakes,
    }


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): DataLoader for the dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for computation.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()
    total_loss = 0
    total = 0
    correct = 0
    predictions, targets, softmax_outputs = [], [], []

    with torch.no_grad():
        for videos, freq_videos, landmarks, labels, _ in dataloader:
            videos, freq_videos, landmarks, labels = (
                videos.to(device),
                freq_videos.to(device),
                landmarks.to(device),
                labels.to(device),
            )

            outputs = model(videos, freq_videos, landmarks)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            softmax_output = F.softmax(outputs, dim=1)
            softmax_outputs.extend(softmax_output[:, 1].cpu().numpy())

    accuracy = 100.0 * correct / total if total != 0 else 0
    precision = precision_score(targets, predictions, average='macro', zero_division=0)
    recall = recall_score(targets, predictions, average='macro', zero_division=0)
    f1 = f1_score(targets, predictions, average='macro', zero_division=0)

    # Compute AUC and EER
    if len(np.unique(targets)) == 2:
        auc = roc_auc_score(targets, softmax_outputs)
        eer = calculate_eer(targets, softmax_outputs)
    else:
        auc = roc_auc_score(targets, softmax_outputs, multi_class="ovr")
        eer = None

    error_rate = 1 - (correct / total) if total != 0 else 0
    balanced_acc = balanced_accuracy_score(targets, predictions)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'eer': eer,
        'error_rate': error_rate,
        'balanced_accuracy': balanced_acc,
        'predictions': predictions,
        'targets': targets,
    }


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    accumulation_steps=4,
):
    """
    Train the model.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for computation.
        num_epochs (int): Number of epochs to train.
        accumulation_steps (int): Number of steps to accumulate gradients.

    Returns:
        None
    """
    metrics = {
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_auc': [],
        'val_eer': [],
        'val_error_rate': [],
        'val_balanced_accuracy': [],
        'predictions': [],
        'targets': [],
    }

    best_accuracy = 0
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(
            total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit='batch'
        )

        optimizer.zero_grad()
        train_predictions, train_targets = [], []

        for i, (videos, freq_videos, landmarks, labels, _) in enumerate(train_loader):
            videos, freq_videos, landmarks, labels = (
                videos.to(device),
                freq_videos.to(device),
                landmarks.to(device),
                labels.to(device),
            )

            outputs = model(videos, freq_videos, landmarks)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            current_loss = total_loss / (i + 1)
            current_accuracy = 100.0 * total_correct / total_samples
            pbar.set_postfix(loss=f"{current_loss:.4f}", accuracy=f"{current_accuracy:.1f}%")
            pbar.update(1)

        if (i + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        pbar.close()

        # Compute training metrics
        train_accuracy = 100.0 * total_correct / total_samples if total_samples != 0 else 0
        train_loss = total_loss / len(train_loader)
        train_precision = precision_score(train_targets, train_predictions, average='macro', zero_division=0)
        train_recall = recall_score(train_targets, train_predictions, average='macro', zero_division=0)
        train_f1 = f1_score(train_targets, train_predictions, average='macro', zero_division=0)

        metrics['train_loss'].append(train_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['train_precision'].append(train_precision)
        metrics['train_recall'].append(train_recall)
        metrics['train_f1'].append(train_f1)

        # Evaluate on validation set
        val_results = evaluate(model, val_loader, criterion, device)
        metrics['val_loss'].append(val_results["loss"])
        metrics['val_accuracy'].append(val_results["accuracy"])
        metrics['val_precision'].append(val_results["precision"])
        metrics['val_recall'].append(val_results["recall"])
        metrics['val_f1'].append(val_results["f1"])
        metrics['val_auc'].append(val_results["auc"])
        metrics['val_eer'].append(val_results["eer"])
        metrics['val_error_rate'].append(val_results["error_rate"])
        metrics['val_balanced_accuracy'].append(val_results["balanced_accuracy"])
        metrics['predictions'].append(val_results["predictions"])
        metrics['targets'].append(val_results["targets"])

        # Logging and saving the best model
        logger.info(
            f'Epoch [{epoch + 1}/{num_epochs}] '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
            f'Train Precision: {train_precision:.2f}, Train Recall: {train_recall:.2f}, '
            f'Train F1 Score: {train_f1:.2f} | '
            f'Val Loss: {val_results["loss"]:.4f}, Val Accuracy: {val_results["accuracy"]:.2f}%, '
            f'Val Precision: {val_results["precision"]:.2f}, Val Recall: {val_results["recall"]:.2f}, '
            f'Val F1 Score: {val_results["f1"]:.2f}, Val AUC: {val_results["auc"]:.2f}, '
            f'Val EER: {val_results["eer"] if val_results["eer"] is not None else "N/A"}, '
            f'Val Error Rate: {val_results["error_rate"]:.2f}, '
            f'Val Balanced Accuracy: {val_results["balanced_accuracy"]:.2f}'
        )

        # Save the best model based on validation accuracy or loss
        if val_results["accuracy"] > best_accuracy or val_results["loss"] < best_loss:
            best_accuracy = val_results["accuracy"]
            best_loss = val_results["loss"]
            save_model(model, config["model_filename"].format(epoch=epoch + 1))
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_accuracy': best_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                filename=config["checkpoint_filename"].format(epoch=epoch + 1),
            )

        # Save metrics after each epoch
        save_metrics(metrics, filename=config["metrics_filename"].format(epoch=epoch + 1))

    # Save final model after training
    save_model(model, config["model_filename"] + "_final")


def save_checkpoint(state, filename):
    """
    Save the training checkpoint.

    Args:
        state (dict): State dictionary containing model and optimizer states.
        filename (str): Path to save the checkpoint.
    """
    torch.save(state, filename)


def save_model(model, filename):
    """
    Save the model state.

    Args:
        model (nn.Module): The model to save.
        filename (str): Path to save the model.
    """
    torch.save(model.state_dict(), filename)


def save_metrics(metrics, filename):
    """
    Save training metrics to a JSON file.

    Args:
        metrics (dict): Dictionary containing metrics.
        filename (str): Path to save the metrics.
    """
    def convert_to_native_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native_types(i) for i in obj]
        else:
            return obj

    metrics = convert_to_native_types(metrics)

    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VideoClassifier(num_classes=2).to(device)
    lr = 0.00008
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    data_dir_real_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFReal'
    data_dir_fake_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFFake'
    freq_dir_real_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFRealFreq'
    freq_dir_fake_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFFakeFreq'
    landmark_dir_real_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFReal/landmarks'
    landmark_dir_fake_ff = 'C:/Users/José Marques/Desktop/FF_c24/FFFake/landmarks'

    transform = transforms.Compose([transforms.ToTensor()])

    train_loader, val_loader = create_data_loaders(
        data_dir_real=data_dir_real_ff,
        data_dir_fake=data_dir_fake_ff,
        freq_dir_real=freq_dir_real_ff,
        freq_dir_fake=freq_dir_fake_ff,
        landmark_dir_real=landmark_dir_real_ff,
        landmark_dir_fake=landmark_dir_fake_ff,
        transform=transform,
        batch_size=2,
        val_split=0.1,
    )

    data_dir_real_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebReal'
    data_dir_fake_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebFake'
    freq_dir_real_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebRealFreq'
    freq_dir_fake_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebFakeFreq'
    landmark_dir_real_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebReal/landmarks'
    landmark_dir_fake_celebdf = 'C:/Users/José Marques/Desktop/tese/celebDF/CelebFake/landmarks'

    test_loader = create_test_loader(
        data_dir_real=data_dir_real_celebdf,
        data_dir_fake=data_dir_fake_celebdf,
        freq_dir_real=freq_dir_real_celebdf,
        freq_dir_fake=freq_dir_fake_celebdf,
        landmark_dir_real=landmark_dir_real_celebdf,
        landmark_dir_fake=landmark_dir_fake_celebdf,
        transform=transform,
        batch_size=2,
    )

    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        accumulation_steps=1,
        num_epochs=5,
    )

    # Test the model on the query set
    test_metrics = test_model(
        model, test_loader, criterion, device, mistakes_filename="top_mistakes_few_shot.json"
    )

    # Save the test metrics
    save_metrics(test_metrics, filename="Baseline_model_test_metrics_few_shot.json")