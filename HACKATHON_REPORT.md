# 🚀 Mission Control: AI for Space Station Safety
**AIT CV Hackathon 2026 Submission**

## 📄 Executive Summary
We developed an advanced Computer Vision system capable of detecting critical safety equipment on the International Space Station (ISS) with **high reliability**. Facing a challenging dataset with severe class imbalance, our solution leverages **YOLOv8**, **Copy-Paste Augmentation**, and **Test-Time Augmentation (TTA)** to achieve a **Peak Validation mAP of 80.7%** and a **Test Precision of 91.6%**. 

Our final deliverable includes a deployment-ready "Mission Control" dashboard that processes visual feeds in real-time, providing astronauts with instant situational awareness.

> **Note:** This repository uses **Git LFS** to store high-precision model weights (`.pt` files). Please run `git lfs install` and `git lfs pull` after cloning.

---

## 🎯 Problem Statement
The dataset provided (Space Station Interior) presented three critical challenges that standard training methods failed to address:

1.  **Severe Class Imbalance**: Common objects like `OxygenTank` (1,500+ instances) dominated rare but critical objects like `FirstAidBox` (323 instances).
2.  **Low Recall (The "Missing Object" Problem)**: Initial baselines missed ~30% of objects, unacceptable for safety-critical environments.
3.  **Visual Complexity**: Varied lighting, occlusion, and cluttered backgrounds typical of the ISS.

---

## 🛠️ Methodology: The Winning Formula

We moved beyond standard "train-and-pray" methods, implementing a targeted engineering strategy:

### 1. Architecture Selection
We selected **YOLOv8-Medium (YOLOv8m)** over the Nano/Small variants.
*   **Why?** The Medium architecture offers the best trade-off between feature extraction power (needed for small/rare objects) and inference speed (needed for real-time edge deployment).

### 2. Combating Imbalance: Copy-Paste Augmentation 🏆
To solve the `FirstAidBox` scarcity, we implemented **Copy-Paste Augmentation** (`copy_paste=0.35`).
*   **Mechanism:** The system extracts objects from one image and "pastes" them into others.
*   **Impact:** This artificially boosted the prevalence of rare classes, forcing the model to learn their features robustly.

### 3. Training Optimization
We fine-tuned our model using a high-precision strategy:
*   **Multi-Scale Training (`multi_scale=0.5`)**: Trained on varying resolutions (320px-960px) to detect objects at different distances.
*   **Cosine Learning Rate**: ensured smooth convergence into the global minimum.
*   **Recall-Focused Loss**: We increased the Class Loss Weight (`cls=1.2`) and lowered the IoU threshold (`iou=0.55`) to prioritize finding *every* object.

### 4. Post-Processing: Test-Time Augmentation (TTA)
During inference, we enabled TTA, which predicts on multiple augmented versions of the same image (flip, scale) and averages the results. This delivered a **final reliability boost** of +1-3% mAP.

---

## 📊 Results & Performance

Our system demonstrates exceptional reliability, favoring Precision (correctness) while maintaining strong Recall.

### Official Metrics (Test Set - 1,400 Unseen Images)

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **Peak Validation mAP@50** | **80.7%** | **Best performance during optimization** |
| **Test Precision** | **91.6%** | **Extremely low false alarm rate** (Crucial for user trust) |
| **Test mAP@50** | **79.2%** | Solid reduced-bias generalization |
| **Test mAP@50-95** | **64.6%** | High localization accuracy (tight bounding boxes) |

### Reliability Analysis
*   **91.6% Precision** means when our system alerts the crew to a `FireExtinguisher`, it is almost certainly there.
*   **Confusion Matrix**: Shows strong diagonal performance, indicating minimal confusion between visually similar classes (e.g., Oxygen vs. Nitrogen tanks).

---

## 💻 Product Demo: "Mission Control" UI

We went beyond code to build a refined **Streamlit Dashboard** for the end-user.

### Key Features:
*   **Real-time Inference**: Drag-and-drop analysis with <100ms latency on GPU.
*   **Interactive Controls**: Adjustable Confidence and IoU thresholds for different mission scopes.
*   **Analytics Tab**: Transparent visualizations of training progress, confusion matrices, and dataset challenges.
*   **Aesthetic Design**: Dark-mode "Space" theme optimized for low-light control rooms.

---

## 🚀 Future Work
Given more time, we would implement:
1.  **Ensembling**: Combining YOLOv8m with an EfficientDet model.
2.  **Slicing Aided Hyper-Inference (SAHI)**: To further improve detection of small objects in high-res feeds.
3.  **Video Tracking**: Adding object ID tracking (DeepSORT) for continuous video feeds.

---

**Developed for AIT CV Hackathon 2026**
