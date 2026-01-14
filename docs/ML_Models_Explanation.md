# StellaSentinal ML Models - Complete Technical Guide

## Overview

This document provides a comprehensive explanation of all machine learning models used in the StellaSentinal anomaly detection system.

---

## Table of Contents

1. [Isolation Forest (Primary Detector)](#1-isolation-forest-primary-detector)
2. [Local Outlier Factor (LOF)](#2-local-outlier-factor-lof)
3. [One-Class SVM (OCSVM)](#3-one-class-svm-ocsvm)
4. [Variational Autoencoder (VAE)](#4-variational-autoencoder-vae)
5. [DBSCAN Clustering](#5-dbscan-clustering)
6. [Ensemble Detector](#6-ensemble-detector)
7. [Bayesian Baseline](#7-bayesian-baseline)
8. [Statistical Baseline (MAD)](#8-statistical-baseline-mad)
9. [Predictive Detector](#9-predictive-detector)
10. [Drift Detection](#10-drift-detection)
11. [Model Orchestration Summary](#11-model-orchestration-summary)

---

## 1. Isolation Forest (Primary Detector)

**File:** `src/device_anomaly/models/anomaly_detector.py`

### How It Works

Isolation Forest is based on a simple principle: **anomalies are easier to isolate than normal points**.

- Normal Point: Requires many splits to isolate
- Anomaly: Requires few splits to isolate (it's "different")

### The Algorithm

1. Randomly select a feature
2. Randomly select a split value between min and max
3. Recursively partition data until each point is isolated
4. Anomaly Score = Average path length across all trees

**Shorter path length → More anomalous**

### Configuration in StellaSentinal

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `n_estimators` | 300 | Number of isolation trees |
| `contamination` | 0.05 | Expected 5% of data are anomalies |
| `random_state` | 42 | Reproducibility |

### Scoring Formula

```
score = -(average_path_length - expected_path_length) / normalization_factor

Interpretation:
  score < 0  →  Anomaly (shorter path than expected)
  score > 0  →  Normal (longer path than expected)
```

### Adaptive Thresholding

When the model finds 0 anomalies (distribution shift), it falls back to:

```
if model_anomaly_count == 0:
    threshold = percentile(scores, 5.0)  # Bottom 5%
    labels = -1 if score <= threshold else 1
```

---

## 2. Local Outlier Factor (LOF)

**Used in:** `ensemble_detector.py`, `ml_baseline_engine.py`

### How It Works

LOF measures **local density deviation**. A point is anomalous if its neighborhood is denser than itself.

### The Algorithm

1. Find k-nearest neighbors for each point
2. Calculate Local Reachability Density (LRD)
3. Compare each point's LRD to its neighbors' LRD

### Mathematical Formula

```
                Σ LRD(neighbor)
LOF(point) = ─────────────────────
              k × LRD(point)

Where:
  LRD(p) = 1 / (average reachability distance to k neighbors)
  Reachability Distance = max(k-distance(neighbor), actual_distance)
```

### Interpretation

| LOF Value | Meaning |
|-----------|---------|
| LOF ≈ 1 | Similar density to neighbors (normal) |
| LOF > 1 | Lower density than neighbors (outlier) |
| LOF >> 1 | Much lower density (strong anomaly) |

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_neighbors` | 20 | Size of local neighborhood |
| `novelty` | True | Enable prediction on new data |

---

## 3. One-Class SVM (OCSVM)

**Used in:** `ensemble_detector.py`

### How It Works

OCSVM learns a **decision boundary** that separates normal data from the origin in a high-dimensional feature space.

### The Algorithm

1. Map data to high-dimensional space using kernel (RBF)
2. Find hyperplane that maximizes distance from origin
3. Points on the "wrong side" are anomalies

### Mathematical Formulation

```
Minimize: ½||w||² + (1/νn)Σξᵢ - ρ

Subject to: w·φ(xᵢ) ≥ ρ - ξᵢ, ξᵢ ≥ 0

Where:
  w = normal vector to hyperplane
  ρ = offset from origin
  ξᵢ = slack variables (allow some violations)
  ν = upper bound on fraction of outliers
  φ = kernel mapping (RBF: φ(x)·φ(y) = exp(-γ||x-y||²))
```

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `kernel` | "rbf" | Radial Basis Function |
| `nu` | 0.05 | ~5% of points can be outliers |

---

## 4. Variational Autoencoder (VAE)

**File:** `src/device_anomaly/models/vae_detector.py`

### How It Works

VAE learns to **compress and reconstruct** data. Anomalies have **high reconstruction error** because they don't fit the learned normal patterns.

### Architecture

```
Input (n features)
        ↓
┌───────────────────┐
│     ENCODER       │
│  256 → 128 → 64   │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   LATENT SPACE    │
│    μ (mean)       │ ──┐
│    σ² (variance)  │   │ Reparameterization
└───────────────────┘   │ z = μ + σ × ε
          ↑             │ where ε ~ N(0,1)
          └─────────────┘
          ↓
┌───────────────────┐
│     DECODER       │
│  64 → 128 → 256   │
└─────────┬─────────┘
          ↓
Output (reconstruction)
```

### Loss Function

```
Total Loss = Reconstruction Loss + β × KL Divergence

Reconstruction Loss = MSE(input, output) = (1/n)Σ(xᵢ - x̂ᵢ)²

KL Divergence = -0.5 × Σ(1 + log(σ²) - μ² - σ²)

β (kl_weight) = 0.001
```

### Anomaly Scoring

```
reconstruction_error = mean((input - output)²)

threshold = percentile(training_errors, 95)  # Top 5% are anomalies

is_anomaly = reconstruction_error > threshold
```

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `latent_dim` | 32 | Bottleneck size |
| `hidden_dims` | [256, 128, 64] | Layer sizes |
| `dropout` | 0.2 | Regularization |
| `epochs` | 100 | Training iterations |
| `learning_rate` | 0.001 | Optimizer step size |
| `kl_weight` | 0.001 | β parameter |
| `batch_size` | 256 | Samples per batch |
| `early_stopping_patience` | 10 | Stop if no improvement |

---

## 5. DBSCAN Clustering

**Used in:** `ml_baseline_engine.py`

### How It Works

DBSCAN finds **dense regions** and labels points in sparse regions as **noise (anomalies)**.

### The Algorithm

For each point:
1. Count neighbors within radius ε (eps)
2. If neighbors ≥ min_samples → Core point
3. If reachable from core point → Border point
4. Otherwise → Noise (ANOMALY)

### Automatic Parameter Selection

```
# eps is determined automatically using k-distance graph
distances = k_nearest_neighbor_distances(data, k=min_samples)
eps = percentile(distances, 90)  # 90th percentile
```

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `min_samples` | 5 | Minimum points for dense region |
| `eps` | auto | Calculated from data distribution |

### Output Labels

| Label | Meaning |
|-------|---------|
| -1 | Noise (anomaly) |
| 0, 1, 2... | Cluster membership |

---

## 6. Ensemble Detector

**File:** `src/device_anomaly/models/ensemble_detector.py`

### How It Works

Combines multiple algorithms using **weighted voting** to reduce false positives.

### Ensemble Formula

```
                0.50 × IF_score + 0.30 × LOF_score + 0.20 × OCSVM_score
Ensemble Score = ─────────────────────────────────────────────────────────
                                0.50 + 0.30 + 0.20
```

### Score Normalization

Each algorithm's raw scores are normalized to [0, 1]:

```
normalized = (score - min_score) / (max_score - min_score)
```

### Agreement Metric

```
agreement = 1 - std(individual_predictions)

High agreement = High confidence
Low agreement = Uncertain, needs review
```

### Why Use an Ensemble?

| Single Model Problem | Ensemble Solution |
|---------------------|-------------------|
| IF misses local anomalies | LOF catches them |
| LOF struggles in high dimensions | IF handles it well |
| OCSVM sensitive to parameters | Voting averages out errors |

---

## 7. Bayesian Baseline

**File:** `src/device_anomaly/models/ml_baseline_engine.py`

### How It Works

Uses **Bayesian inference** to maintain probability distributions over metric values, updating beliefs as new data arrives.

### Mathematical Foundation

**Prior Distribution (before seeing data):**
```
μ ~ Normal(μ₀, σ²/κ₀)
σ² ~ Inverse-Gamma(α₀, β₀)
```

**Posterior Update (after seeing n observations):**
```
κₙ = κ₀ + n
μₙ = (κ₀μ₀ + n·x̄) / κₙ
αₙ = α₀ + n/2
βₙ = β₀ + 0.5·Σ(xᵢ - x̄)² + (κ₀·n·(x̄ - μ₀)²) / (2·κₙ)

Posterior Mean = μₙ
Posterior Variance = βₙ / (αₙ × κₙ)
```

### Anomaly Probability Calculation

```
z_score = (observed_value - posterior_mean) / posterior_std

p_extreme = 2 × (1 - Φ(|z_score|))  # Φ = standard normal CDF

anomaly_probability = 1 - p_extreme
```

### Severity Classification

| Probability | Severity |
|-------------|----------|
| ≥ 0.99 | Critical |
| ≥ 0.95 | Warning |
| ≥ 0.80 | Elevated |
| < 0.80 | Normal |

### Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `prior_weight` | 0.3 | Weight of historical belief |
| `likelihood_weight` | 0.7 | Weight of new data |
| `uncertainty_threshold` | 0.2 | Trigger recompute |
| `confidence_interval` | 0.95 | Credible interval width |

---

## 8. Statistical Baseline (MAD)

**File:** `src/device_anomaly/models/baseline.py`

### How It Works

Uses **Median Absolute Deviation** - a robust measure that isn't affected by outliers.

### Formula

```
Median = middle value of sorted data

MAD = median(|xᵢ - Median|)

Z-score = (x - Median) / MAD
```

### Why MAD over Standard Deviation?

```
Data: [10, 11, 12, 11, 10, 1000]  ← outlier

Standard Deviation: 403.5  ← Destroyed by outlier
MAD: 1.0                   ← Robust to outlier
```

### Threshold Determination

| Level | Percentile Range |
|-------|------------------|
| Warning | 5th - 95th |
| Critical | 1st - 99th |

---

## 9. Predictive Detector

**File:** `src/device_anomaly/models/predictive_detector.py`

### Battery Failure Prediction

**Algorithm:** Exponential Smoothing

```
# Smoothed drain rate
smoothed[0] = history[0]
for i in range(1, n):
    smoothed[i] = α × history[i] + (1-α) × smoothed[i-1]

avg_drain_per_hour = smoothed[-1]

# Predict future level
predicted_level = current_level - (avg_drain_per_hour × hours_ahead)

# Time until critical (10%)
hours_until_critical = (current_level - 10%) / avg_drain_per_hour
```

### Storage Exhaustion Prediction

**Algorithm:** Linear Regression

```
# Fit trend line
slope, intercept = polyfit(time_points, storage_history, degree=1)

if slope < 0:  # Storage decreasing
    days_until_zero = -current_storage / slope
    will_exhaust = days_until_zero < warning_days
```

### Network Degradation Prediction

**Algorithm:** Linear Trend Analysis

```
# Signal trend
signal_slope = polyfit(x, signal_history, 1)[0]

# Degradation indicators
signal_degrading = signal_slope < -0.5 dBm/period

predicted_signal_7d = current_signal + (signal_slope × 7)
```

---

## 10. Drift Detection

**File:** `src/device_anomaly/models/ml_baseline_engine.py`

Detects when data distribution changes, making the model stale.

### Method 1: Population Stability Index (PSI)

```
PSI = Σ (current% - reference%) × ln(current% / reference%)

Interpretation:
  PSI < 0.10  →  No significant change
  PSI < 0.15  →  Minor change
  PSI ≥ 0.15  →  Significant drift (ALERT)
```

### Method 2: Kolmogorov-Smirnov Test

```
KS Statistic = max|F_current(x) - F_reference(x)|

If p-value < 0.05 → Distributions are different (DRIFT)
```

### Method 3: Jensen-Shannon Divergence

```
JS(P||Q) = 0.5 × KL(P||M) + 0.5 × KL(Q||M)
where M = 0.5 × (P + Q)

If JS > 0.1 → Drift detected
```

### Method 4: Mean Shift

```
z = (mean_current - mean_reference) / std_reference

If |z| > 2.0 → Significant mean shift
```

### Method 5: Variance Ratio

```
ratio = var_current / var_reference

If ratio > 2.0 or ratio < 0.5 → Variance change detected
```

### Combined Decision

```
drift_detected = (number of methods detecting drift) >= 2
```

---

## 11. Model Orchestration Summary

### Data Flow Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INCOMING DATA                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                             │
│  • Cohort normalization (z-scores per device type)              │
│  • Temporal features (hour, day, trends)                         │
│  • Missing value imputation (median)                             │
│  • StandardScaler normalization                                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
   ┌────────────┐      ┌────────────┐      ┌────────────┐
   │ Isolation  │      │    VAE     │      │ Statistical│
   │   Forest   │      │ Autoencoder│      │  Baseline  │
   │  (35-50%)  │      │   (25%)    │      │   (MAD)    │
   └─────┬──────┘      └─────┬──────┘      └─────┬──────┘
         │                   │                   │
         │    ┌────────────┐ │                   │
         │    │    LOF     │ │                   │
         │    │   (25%)    │ │                   │
         │    └─────┬──────┘ │                   │
         │          │        │                   │
         │    ┌────────────┐ │                   │
         │    │  DBSCAN    │ │                   │
         │    │   (15%)    │ │                   │
         │    └─────┬──────┘ │                   │
         │          │        │                   │
         ▼          ▼        ▼                   ▼
   ┌─────────────────────────────────────────────────┐
   │              ENSEMBLE SCORING                    │
   │  weighted_score = Σ(weight × normalized_score)  │
   └─────────────────────────┬───────────────────────┘
                             │
                             ▼
   ┌─────────────────────────────────────────────────┐
   │           SEVERITY CLASSIFICATION                │
   │  Critical ← High ← Medium ← Low ← Normal        │
   └─────────────────────────┬───────────────────────┘
                             │
                             ▼
   ┌─────────────────────────────────────────────────┐
   │              LLM EXPLANATION                     │
   │  Natural language explanation of anomaly        │
   │  cause and recommended actions                  │
   └─────────────────────────────────────────────────┘
```

---

## Quick Reference Table

| Model | Type | Strength | Weakness | Weight |
|-------|------|----------|----------|--------|
| **Isolation Forest** | Tree ensemble | Fast, scalable, global outliers | Misses local patterns | 35-50% |
| **LOF** | Density-based | Local anomalies, clusters | Slow on large data | 25-30% |
| **VAE** | Deep learning | Complex non-linear patterns | Needs lots of data | 25% |
| **DBSCAN** | Clustering | Arbitrary shapes, no assumptions | Sensitive to eps | 15% |
| **OCSVM** | Boundary | Complex decision boundaries | Slow, parameter-sensitive | 20% |
| **Bayesian** | Statistical | Uncertainty, online updates | Assumes normality | N/A |
| **MAD Baseline** | Statistical | Robust, interpretable | Simple patterns only | N/A |
| **Predictive** | Time-series | Proactive alerts | Needs history | N/A |

---

## Key Configuration Parameters

### Primary Model (Isolation Forest)

| Parameter | Value | Impact |
|-----------|-------|--------|
| `contamination` | 0.05 | Expected anomaly rate - CRITICAL |
| `n_estimators` | 300 | More = better but slower |
| `adaptive_percentile` | 5.0 | Fallback threshold |

### VAE

| Parameter | Value | Impact |
|-----------|-------|--------|
| `kl_weight` | 0.001 | Balance reconstruction vs regularization |
| `latent_dim` | 32 | Higher = more capacity, overfit risk |
| `epochs` | 100 | With patience=10 for early stopping |

### Drift Detection Thresholds

| Method | Threshold |
|--------|-----------|
| PSI | ≥ 0.15 |
| KS Test | p < 0.05 |
| JS Divergence | > 0.1 |
| Mean Shift | \|z\| > 2.0 |
| Variance Ratio | > 2.0 or < 0.5 |

---

## Severity Classification

| Score Range | Severity | Action |
|-------------|----------|--------|
| score ≤ -0.7 | Critical | Immediate alert |
| -0.7 < score ≤ -0.5 | High | Priority alert |
| -0.5 < score ≤ -0.3 | Medium | Standard alert |
| -0.3 < score ≤ 0.0 | Low | Logged |
| score > 0.0 | Normal | No action |

---

*Document generated for StellaSentinal Anomaly Detection System*
*This multi-model approach provides defense in depth - different algorithms catch different types of anomalies, reducing both false positives and false negatives.*
