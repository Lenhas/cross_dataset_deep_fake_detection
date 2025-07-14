# Cross‑Dataset Deep Fake Detection

A research‑grade pipeline that detects deepfaked videos by **fusing spatial, frequency and facial‑landmark clues** inside a Transformer‑based video classifier enhanced with **Triplet Attention** and **Gated Multimodal Units (GMUs)**. The code accompanies the master’s thesis *Cross‑Dataset Deep Fake Detection* (included in this repo) and is structured so you can reproduce preprocessing, training and evaluation from raw videos to metrics.

---

## Table of Contents

1. [Features](#features)
2. [Repository Layout](#repository-layout)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Dataset Pre‑Processing](#dataset-pre-processing)
6. [Training & Evaluation](#training--evaluation)
7. [Results & Checkpoints](#results--checkpoints)
8. [Roadmap](#roadmap)
9. [Citation](#citation)
10. [License](#license)

---

## Features

- **End‑to‑end workflow** – from cropping faces to saving model checkpoints.
- **Multimodal fusion** – spatial RGB, FFT‑based frequency maps and 68‑point facial landmarks.
- **Attention everywhere** – Triplet Attention on backbone feature maps; Transformer encoder across time.
- **Cross‑dataset generalisation** – trained on FaceForensics++ (c23/c24) and evaluated on Celeb‑DF.

---

## Repository Layout

| Path                                                                                                        | Description                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `Crop_view_faces.py`                                                                                        | Quick sanity‑check script that opens a video, detects the first face with **MTCNN**, zooms ×1.2 and crops/plots the result for up to 10 frames. |
| `DeepFake_PreProcess.py`                                                                                    | Extracts **11 frames (≈260‑270)** per video, crops a 256×256 face stamp and saves a \`\`\*\* tensor\*\* per video.                              |
| `DeepFake_Preprocess_Frequency.py`                                                                          | Same sampling logic, but converts each cropped frame to a normalised **magnitude FFT** image before stacking into `.npy`.                       |
| `Deepfake_PreProcess_Landmarks.py`                                                                          | Uses **dlib** to locate 68 landmarks on the same frame range, saves them as a per‑video **JSON**. Includes optional visualisation.              |
| `triplet_attention.py`                                                                                      | Self‑contained PyTorch implementation of the **Triplet Attention** module from *Rotate to Attend (2021)*.                                       |
| `Deepfake_detector.py`                                                                                      | Houses everything else:                                                                                                                         |
|  • `VideoClassifier` – dual EfficientNet‑B0 backbones (RGB & FFT) → attention → GMUs → Transformer → logits |                                                                                                                                                 |
|  • `VideoDataset` – streams `(rgb, fft, landmarks)` triplets                                                |                                                                                                                                                 |
|  • training/validation/test loops, metric logging & checkpoint helpers.                                     |                                                                                                                                                 |
| `Cross_Dataset_Deep_Fake_Detection.pdf`                                                                     | Full thesis (background, method, experiments, discussion).                                                                                      |

Generated artifacts (not in git):

```
<dataset>/{real,fake}/             # .npy tensors from DeepFake_PreProcess
<dataset>/{real,fake}Freq/         # .npy tensors from DeepFake_Preprocess_Frequency
<dataset>/{real,fake}/landmarks/   # .json files from Deepfake_PreProcess_Landmarks
checkpoints/                       # .pth & .json produced while training
```

---

## Installation

```bash
# 1. Recommended: create a fresh venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Core dependencies
pip install --upgrade pip wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Adjust CUDA/CPU build
pip install facenet-pytorch dlib opencv-python numpy scipy scikit-learn tqdm matplotlib pillow

# (Optional) for PDF reading
pip install PyPDF2
```

> **Conda users**: alternatively run `conda env create -f environment.yml` if you maintain one.

---

## Quick Start

1. **Pre‑process a dataset (RGB)**
   ```bash
   python DeepFake_PreProcess.py \
       --input /path/to/videos \
       --output /path/to/RGB_npy
   ```
2. **Generate frequency tensors**
   ```bash
   python DeepFake_Preprocess_Frequency.py \
       --input /path/to/videos \
       --output /path/to/Freq_npy
   ```
3. **Extract landmarks**
   ```bash
   python Deepfake_PreProcess_Landmarks.py \
       --video_folder /path/to/videos \
       --output_folder /path/to/landmarks \
       --shape_predictor shape_predictor_68_face_landmarks.dat
   ```
4. **Train**
   ```bash
   python Deepfake_detector.py  # edit paths & hyper‑params inside
   ```
5. **Inference on a new folder** (coming soon – see *Roadmap*).

---

## Dataset Pre‑Processing

All three scripts sample the same **11 contiguous frames** (indices \~260‑270 by default) so that RGB, FFT and landmark features stay temporally aligned. Each script has hard‑coded paths for the thesis experiments; simply replace them or refactor to CLI arguments.

Pre‑processing outputs per video:

```
RGB   →  (11, 256, 256, 3)   .npy
FFT   →  (11, 256, 256)      .npy  (single‑channel later broadcast to 3)
LMK   →  dict{'frame090': [[x,y]×68], ...}  .json
```

---

## Training & Evaluation

`Deepfake_detector.py` defines:

```text
VideoDataset → DataLoader → VideoClassifier → Trainer
```

Key hyper‑parameters (feel free to override):

| Param         | Default                     |
| ------------- | --------------------------- |
| `feature_dim` | 512                         |
| `num_heads`   | 8                           |
| `num_layers`  | 2 Transformer blocks        |
| Optimiser     | AdamW, lr = 8e‑5, wd = 1e‑3 |
| Loss          | CrossEntropy                |

During training the script logs metrics (Accuracy, Precision, Recall, F1, AUC, EER, Balanced Accuracy) and saves a checkpoint whenever validation improves.

---

## Results & Checkpoints

The best model on **FaceForensics++ (c23)** reached an **AUC of 0.93** and transferred to **Celeb‑DF** with **AUC 0.87** under the default 10‑frame setting (see thesis §5). Pre‑trained weights will be uploaded under `releases/`.

---

## Roadmap

-

---

## Citation

If you use this project in academic work please cite:

```bibtex
@mastersthesis{marques2024deepfake,
  title  = {Cross‑Dataset Deep Fake Detection},
  author = {José Pedro da Costa Marques},
  school = {Universidade da Beira Interior},
  year   = {2024}
}
```

---

## License

[MIT](LICENSE) – feel free to use, modify and distribute for research & education.

