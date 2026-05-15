# Gym Exercise Classification via Wearable Sensor Data

A supervised multi-class classification project that identifies gym exercises from IMU and Human Body Capacitance (HBC) sensor signals. Eight machine learning models are benchmarked — from Naive Bayes through a weighted soft-voting ensemble — culminating in **87% accuracy and 0.86 Macro F1** across 11 exercise classes.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Models](#models)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)

---

## Overview

This project explores the automatic recognition of gym exercises using wearable sensor data. Each data sample represents a time-windowed recording from a participant wearing both Inertial Measurement Unit (IMU) and Human Body Capacitance (HBC) sensors. The task is formulated as an 11-class supervised classification problem, with the goal of learning a mapping from sensor feature vectors to discrete exercise labels.

The project proceeds through a structured experimental pipeline: exploratory analysis, multi-stage preprocessing, model training and evaluation, and finally a custom weighted soft-voting ensemble that substantially outperforms all individual classifiers.

---

## Dataset

**RecGym: Gym Workouts Recognition Dataset**  
Collected from 10 participants performing various gym exercises using wearable sensors.

| Source | Link |
|--------|------|
| UCI ML Repository | https://archive.uci.edu/dataset/1128 |
| Kaggle | https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features |
| GitHub (original) | https://github.com/zhaxidele/Toolkit-for-HBC-sensing |

### Sensor Features

| Column | Description |
|--------|-------------|
| `A_x`, `A_y`, `A_z` | Accelerometer readings (3 axes) |
| `G_x`, `G_y`, `G_z` | Gyroscope readings (3 axes) |
| `C_1` | Human Body Capacitance (HBC) sensor value |

### Exercise Classes (11 total)

| Class ID | Exercise |
|---------:|----------|
| 0 | Adductor |
| 1 | Arm Curl |
| 2 | Bench Press |
| 3 | Leg Curl |
| 4 | Leg Press |
| 5 | Riding |
| 6 | Rope Skipping |
| 7 | Running |
| 8 | Squat |
| 9 | Stair Climber |
| 10 | Walking |

> **Note:** The original dataset includes a `Null` (idle/non-exercise) class that was excluded from all experiments. The final task uses 11 exercise classes only.

---

## Preprocessing Pipeline

A four-stage preprocessing pipeline was applied sequentially before model training:

1. **Null Label Removal** — Samples labelled `Null` were replaced with `NaN` and dropped, as they do not correspond to any exercise activity.

2. **Class-wise Outlier Removal (Z-score)** — For each sensor feature, z-scores were computed *within each exercise class*. Samples with |z| > 3 were removed. Computing z-scores per class ensures that inter-class differences in signal magnitude are preserved; only intra-class anomalies caused by sensor jitter or transient noise are discarded.

3. **Min–Max Normalisation to [−1, 1]** — All sensor features were scaled to a common range. Applied *after* outlier removal so that the scaler's min/max reference values reflect typical sensor behaviour rather than extreme spikes.

4. **Moving Average Smoothing** — A window-based moving average was applied to each sensor channel to suppress high-frequency noise while retaining the low-frequency motion patterns characteristic of each exercise.

---

## Models

Eight classifiers were trained and evaluated under identical train/test splits and evaluation conditions:

| Category | Model |
|----------|-------|
| Baseline | Naive Bayes, Logistic Regression, Support Vector Machine |
| Tree / Distance | Decision Tree, k-Nearest Neighbours, Random Forest |
| Gradient Boosting | XGBoost, LightGBM |
| Ensemble | Weighted Soft Voting (KNN + RF + XGBoost) |

The weighted soft-voting ensemble combines predicted class probabilities from KNN, Random Forest, and XGBoost using weights tuned on validation-set Macro F1, prioritising balanced classification across all 11 classes.

---

## Results

### Full Model Comparison

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Naive Bayes | 0.3644 | 0.36 | 0.36 |
| Logistic Regression | 0.1884 | 0.09 | 0.12 |
| SVM | 0.1860 | 0.08 | 0.11 |
| Decision Tree | 0.5528 | 0.54 | 0.55 |
| KNN | 0.6636 | 0.65 | 0.67 |
| Random Forest | 0.6692 | 0.66 | 0.67 |
| XGBoost | 0.6481 | 0.64 | 0.65 |
| **Ensemble (Weighted Soft Vote)** | **0.8675** | **0.8595** | **0.8675** |

The ensemble achieves a **+20 percentage point** improvement in Macro F1 over the best individual model, demonstrating that complementary decision boundaries can be effectively combined to resolve ambiguous inter-class regions.

---

## Repository Structure

```
gym-exercise-classification/
├── notebook.ipynb          # Full analysis: preprocessing → modelling → evaluation
├── README.md               # This file
└── data/
    └── RecGym.csv          # Source dataset (download separately — see Dataset section)
```

> The dataset file `RecGym.csv` is not included in this repository due to size. Download it from any of the sources listed in the [Dataset](#dataset) section and place it in the `data/` directory before running the notebook.

---

## Requirements

```
python >= 3.8
pandas
numpy
scikit-learn
scipy
xgboost
lightgbm
matplotlib
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn scipy xgboost lightgbm matplotlib
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/gym-exercise-classification.git
   cd gym-exercise-classification
   ```

2. Download `RecGym.csv` and place it in the `data/` directory.

3. Open and run the notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

Cells are ordered sequentially from data loading through final result visualisation. Run all cells top-to-bottom for a full reproduction of the reported results.

---

## References

1. Bian, S., Rey, V. F., Yuan, S., & Lukowicz, P. (2025). Hybrid CNN-dilated self-attention model using inertial and body-area electrostatic sensing for gym workout recognition, counting, and user authentication. *arXiv*. https://doi.org/10.48550/arXiv.2503.06311

2. McKinney, W. (2018). *Python for Data Analysis* (2nd ed.). O'Reilly Media.

3. UCI ML Repository — RecGym dataset: https://archive.uci.edu/dataset/1128

4. Kaggle dataset: https://www.kaggle.com/datasets/zhaxidelebsz/10-gym-exercises-with-615-abstracted-features

5. Original data repository: https://github.com/zhaxidele/Toolkit-for-HBC-sensing

6. Ng, A. (2018). *Machine Learning Yearning*. deeplearning.ai.
