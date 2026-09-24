# Agentic Sentinel

**An Agentic Vision-RF Fusion System for Adaptive Autonomous Inspection of Toxic Gas Hazards in Visibility-Degraded Underground Mining and Confined Industrial Environments**

---

## What This Project Is

In underground coal mines and confined industrial spaces, a toxic or flammable gas leak can go undetected until a worker is exposed and unable to move — and a rescuer who then enters without checking the air first faces the same danger. This "second casualty" pattern is well-documented in real incidents.

Existing solutions fall short: point-based gas sensors give no spatial awareness, and camera-only inspection robots fail completely once smoke or dust obscures visibility.

This project builds an autonomous inspection system that fuses:

1. **WiFi Channel State Information (CSI) sensing** — detects presence, movement, and machine activity through visual occlusion, since radio signals aren't blocked by smoke the way light is
2. **Vision-based hazard detection (YOLO)** — identifies smoke/fire when visibility permits
3. **A risk classification layer** — combines confidence scores from both sensing modalities into a safe/danger decision
4. **A multi-agent reasoning layer** (planned) — reasons over the fused risk assessment to decide whether to continue, retreat, or alert a human operator
5. **A hardware safety interlock** (planned) — a physical, AI-independent switch that halts the system if gas levels are dangerous, regardless of any AI decision

## What We're Doing

### Architecture

The system fuses two sensing modalities into a shared decision pipeline:

```
WiFi-CSI Sensor --+--> Presence Model -------> confidence score --+
                   +--> Movement Model -------> confidence score --+
                   +--> Machine On/Off Model -> confidence score --+--> Risk Classifier --> Agentic Reasoning Layer --> Hardware Safety Interlock --> Action
Camera ----------------> YOLO (smoke/fire) ---> confidence score --+
```

The CSI side originally used a single shared model trained on three tasks at once. This was later split into **three independent models** (presence, movement, machine on/off), each **warm-started** from the original shared model's learned weights, after multi-task interference was found to cause real instability during development. Each independent model now produces its own confidence score rather than a single combined prediction.

### Dataset

**EHUNAM** (Scientific Data, 2025, https://doi.org/10.6084/m9.figshare.28541225) — a multi-environment WiFi-CSI dataset. This project uses the **MC3 (Industrial Laboratory)** subset — 678 of the dataset's 2,401 total recordings.

---

## Project Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd Agentic_Sentinel
```

### 2. Set up the Python environment

```bash
python -m venv env
env\Scripts\activate        # Windows
# source env/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Download and prepare the dataset

1. Download the EHUNAM dataset from the link above (start with `Summary.xlsx` to preview before committing to the full ~72GB download).
2. Extract it, then filter to the relevant MC3 (industrial) subset:
   ```bash
   python src/csi_pillar/filter_mc3_files.py
   ```
3. Preprocess the raw CSI signals (phase unwrapping, drift correction, smoothing):
   ```bash
   python src/csi_pillar/batch_preprocess_csi.py
   ```
4. Build the labeled, split training dataset:
   ```bash
   python src/csi_pillar/build_training_dataset_original.py --source "data/mc3_processed" --out "data/training_dataset_original.h5"
   ```

### 4. Train the base multi-task model

```bash
python src/csi_pillar/train_cnn.py --data "data/training_dataset_original.h5" --epochs 20
```

### 5. Train the final, independent single-task models

Each task is warm-started from the base model above:

```bash
python src/csi_pillar/train_final_model.py --task presence --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
python src/csi_pillar/train_final_model.py --task movement --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
python src/csi_pillar/train_final_model.py --task machine_onoff --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 20
```

**Note:** for machine on/off, use a decision threshold of **0.25** (not the default 0.5) when interpreting confidence scores — this was found to give a much better precision/recall balance during threshold tuning.

### 6. (Optional) Validate with cross-validation

To reproduce the reliability checks performed on each model:

```bash
python src/csi_pillar/train_movement_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
python src/csi_pillar/train_machine_onoff_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
python src/csi_pillar/train_presence_kfold.py --source "data/mc3_processed" --base_model "best_model_20mhz.keras" --epochs 15
```

---

## Tech Stack

- **CSI Modeling:** Python, TensorFlow/Keras, scikit-learn, h5py, NumPy, SciPy
- **Vision Modeling (in progress):** PyTorch, Ultralytics YOLO
- **Planned:** LangGraph + local LLM (agentic layer), ROS 2 (robotics integration), Gazebo/Webots (simulation)
