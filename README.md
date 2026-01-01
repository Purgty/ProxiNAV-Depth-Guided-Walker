# PROXI-NAV: Risk-Aware Assistive Navigation via Learned Spatial Risk Modeling

A real-time assistive navigation system for visually impaired users that combines obstacle detection, walkable surface detection, and monocular depth estimation to generate dense navigation risk heatmaps using learned spatial reasoning.

<img width="1080" height="640" alt="architecture_diagram" src="https://github.com/user-attachments/assets/49f18107-2384-468a-9390-ba912eb09a6f" />

## Overview

PROXI-NAV uses a single RGB camera to process scenes in real-time and provide navigation guidance through:
- **Parallel perception**: YOLOv8 for obstacles and walkability + MiDaS for depth
- **Learned risk fusion**: RiskCNN combines spatial cues into a unified risk map
- **Real-time deployment**: Optimized for NVIDIA Jetson Nano

## Repository Structure

```
PROXI-NAV/
├── Assets/                      # Test images and videos
├── Datasets/                    # Dataset preparation and YOLO datasets
│   ├── Pavement_Data/          # Pavement detection dataset (augmented + original)
│   └── WTour_Dataset/          # Walkability/obstacle dataset
├── Output/                      # All inference outputs
│   ├── plain_inference/        # Feature maps (depth, obstacle, walkability, risk)
│   └── processed_frames/       # Video frame outputs
├── Paper/                       # Research paper and architecture diagrams
├── Pavement_Dataset_Model/     # Trained pavement detection model (best.pt)
├── WOTR_Dataset_Model/         # Trained walkability/obstacle model (best.pt)
├── RiskHead Scripts/           # Risk estimation and spatial reasoning
│   ├── CNN/                    # RiskCNN model and training
│   ├── Single Image/           # Single-frame inference pipeline
│   └── Multiple Images/        # Batch/video inference pipeline
├── RiskheadCNN_Ablation/       # Ablation study results
└── RuleBased Scripts/          # Rule-based baseline methods
```

## Installation

```bash
git clone https://github.com/Purgty/ProxiNAV-Depth-Guided-Walker.git
cd ProxiNAV-Depth-Guided-Walker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install ultralytics torch torchvision opencv-python numpy timm
```

### For Jetson Nano

Follow NVIDIA JetPack setup, then install PyTorch for ARM64 and the dependencies above.

## Quick Start

### Single Image Inference

Generate risk map for a single image:

```bash
cd "RiskHead Scripts/Single Image"

# Step 1: Generate spatial maps (obstacle + walkability)
python spatialmap_gen_image.py \
  --image ../../Assets/pavement1.png \
  --pavement_model ../../Pavement_Dataset_Model/best.pt \
  --obstacle_model ../../WOTR_Dataset_Model/best.pt \
  --output_dir ./outputs

# Step 2: Generate depth map
python depthmap_gen.py \
  --image ../../Assets/pavement1.png \
  --output_dir ./outputs

# Step 3: Generate risk map (RiskCNN)
python riskmap_generation.py \
  --obstacle_map ./outputs/obstacle_map.npy \
  --walkability_map ./outputs/walkability_map.npy \
  --depth_map ./outputs/depth_map.npy \
  --risk_model ../CNN/risk_cnn.pt \
  --output ./outputs/risk_map.png
```

**Example output:**

![pavement3_output](https://github.com/user-attachments/assets/336e71a3-53c8-4ee6-b52e-1540aec20138)

*Left to right: Input image, obstacle map, walkability map, depth map, final risk map*

### Video/Batch Inference

Process multiple frames from a video:

```bash
cd "RiskHead Scripts/Multiple Images"

# Generate spatial maps for all frames
python spatialmap_gen_vid.py \
  --video ../../Assets/tnagar_ppl.mp4 \
  --pavement_model ../../Pavement_Dataset_Model/best.pt \
  --obstacle_model ../../WOTR_Dataset_Model/best.pt \
  --output_dir ../../Output/processed_frames/tnagar_ppl

# Generate depth maps
python depthmap_gen.py \
  --input_dir ../../Output/processed_frames/tnagar_ppl \
  --output_dir ../../Output/plain_inference/depth_maps
```

### Rule-Based Baseline

Run the rule-based fusion baseline (no learned RiskCNN):

```bash
cd "RuleBased Scripts"

python inference_single_frame_nocurb.py \
  --image ../Assets/pavement1.png \
  --pavement_model ../Pavement_Dataset_Model/best.pt \
  --obstacle_model ../WOTR_Dataset_Model/best.pt \
  --output_dir ../Output
```

## Ablation Study

Compare RiskCNN against baselines (rule-based, single-cue, etc.):

```bash
cd "RiskHead Scripts/Multiple Images"

python riskmap_ablation_study_folder.py \
  --input_dir ../../RiskheadCNN_Ablation/plain_inference \
  --risk_model ../CNN/risk_cnn.pt \
  --output_dir ../../RiskheadCNN_Ablation/plain_inference/ablation_results
```

**Ablation visualization:**

<img width="1080" height="640" alt="ablation" src="https://github.com/user-attachments/assets/f317875f-105f-4d5f-b13b-5ae6bc498416" />
*Comparison of different fusion strategies: obstacles-only, walkability-only, depth-only, rule-based, RiskCNN*

This generates side-by-side comparisons showing:
- Obstacle-only risk
- Walkability-only risk
- Depth-only risk
- Rule-based fusion
- **RiskCNN (learned fusion)** ← Best performance

## Training RiskCNN

If you want to retrain the risk head:

```bash
cd "RiskHead Scripts/CNN"

python RIskhead_CNN.py \
  --train_dir ../../Output/plain_inference \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.001 \
  --output_model risk_cnn.pt
```

Dataset structure expected:
```
train_dir/
├── obstacle_maps/     # .npy files
├── walkability_maps/  # .npy files
├── depth_maps/        # .npy files
└── risk_labels/       # .npy pseudo-labels
```

Validate the trained model:

```bash
python Riskhead_val.py \
  --test_dir ../../Output/plain_inference \
  --model risk_cnn.pt
```

## Dataset Preparation

### Pavement Detection Dataset

Located in `Datasets/Pavement_Data/`:
- **Original dataset**: 700 images (560 train / 105 val / 35 test)
- **Augmented dataset**: 3,500 images (5× augmentation)

To augment your own data:

```bash
cd Datasets/Pavement_Data

python augementations_img.py \
  --input_dir original_yolo_dataset \
  --output_dir augmented_yolo_dataset \
  --augment_factor 5
```

### Walkability/Obstacle Dataset

Located in `Datasets/WTour_Dataset/`:
- Test set: 990 images with YOLO labels

## Visualization & Outputs

All inference outputs are stored in `Output/`:

| Output Type | Location | Description |
|-------------|----------|-------------|
| Annotated frames | `Output/annotated_frames_pavement_val/` | YOLO detections overlaid |
| Feature maps | `Output/plain_inference/` | Obstacle, walkability, depth maps (.npy) |
| Risk maps | `Output/plain_inference/risk_labels/` | Final risk heatmaps |
| Processed videos | `Output/output_pavements/`, `output_obstacles/` | Segmented video outputs |
| Ablation results | `RiskheadCNN_Ablation/plain_inference/ablation_results/` | Comparison visualizations |

**Color scale for risk maps:**
- Blue/Cyan: Safe, navigable
- Green/Yellow: Moderate risk
- Orange/Red: High risk, avoid

## Models

Pre-trained models are included:

| Model | Location | Purpose |
|-------|----------|---------|
| Pavement detector | `Pavement_Dataset_Model/best.pt` | YOLOv8 for walkable surfaces |
| Obstacle detector | `WOTR_Dataset_Model/best.pt` | YOLOv8 for obstacles |
| RiskCNN | `RiskHead Scripts/CNN/risk_cnn.pt` | Learned risk fusion |

## Real-Time Deployment (Jetson Nano)

For embedded deployment:

1. Use the rule-based scripts for faster inference (no RiskCNN overhead)
2. Reduce input resolution to 416×416 or 320×320
3. Consider INT8 quantization for YOLOv8 models
4. Run at 10-15 FPS for real-time audio feedback

Example real-time script structure:
```python
# Capture frame → Run obstacle + pavement detection → Generate depth
# → Fuse with RiskCNN → Aggregate directional risk → Audio output
```

See `RuleBased Scripts/inference_enhanced.py` for a complete pipeline example.

## Results Summary

**Quantitative (Ablation Study):**
- RiskCNN outperforms rule-based fusion in safety-critical scenarios
- Better handling of overlapping cues (obstacles on non-walkable surfaces)
- Lower false positive rate compared to single-cue methods

**Qualitative:**
- Robust to diverse environments (rain, mist, urban, suburban)
- Correctly prioritizes near obstacles over distant ones
- Integrates walkability and depth appropriately

See sample outputs in `RiskheadCNN_Ablation/plain_inference/ablation_results/`

## Limitations

- Monocular depth is scale-ambiguous (relative depth only)
- Performance degrades in extreme low-light conditions
- Walkability detector trained on specific surface types (may misclassify novel terrain)
- Jetson Nano latency: ~175-250ms per frame (4-6 FPS full pipeline)

## Future Work

- Temporal smoothing with optical flow or LSTM
- Multi-sensor fusion (LiDAR, thermal camera)
- Haptic feedback integration
- User-adaptive risk thresholds
- End-to-end learning from RGB to risk

## License

MIT License - see LICENSE file for details.

## Contact

For questions or collaboration, please open an issue on GitHub.

---

**Architecture diagram created with:** `Paper/architecture_diag.py`  
**Directory structure generated with:** `pyhu.py`
