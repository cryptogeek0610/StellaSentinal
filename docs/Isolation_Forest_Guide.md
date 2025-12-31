# Isolation Forest: A Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concept](#core-concept)
3. [How It Works](#how-it-works)
4. [Key Parameters](#key-parameters)
5. [Advantages & Limitations](#advantages--limitations)
6. [Isolation Forest in This Project](#isolation-forest-in-this-project)
7. [Best Practices](#best-practices)
8. [References](#references)

---

## Introduction

**Isolation Forest** is an unsupervised machine learning algorithm designed specifically for anomaly detection. Unlike traditional methods that build a profile of "normal" data and flag deviations, Isolation Forest isolates anomalies by exploiting their inherent characteristic: **anomalies are few and different, so they are easier to isolate from normal instances**.

### Key Characteristics

- **Unsupervised**: Doesn't require labeled data (no need to know what's "normal" vs "anomalous" beforehand)
- **Tree-based**: Uses an ensemble of decision trees
- **Efficient**: Has a linear time complexity O(n), making it suitable for large datasets
- **Effective**: Works well on high-dimensional data with mixed feature types

---

## Core Concept

The fundamental idea behind Isolation Forest is simple yet powerful:

> **Anomalies are easier to isolate than normal points because they are rare and different.**

### The Isolation Principle

1. **Normal instances** cluster together in dense regions of feature space
2. **Anomalous instances** are scattered in sparse regions
3. It takes fewer random splits to isolate an anomaly than a normal point
4. The average path length in a random tree is shorter for anomalies

### Example Visualization

Imagine you're looking at a 2D plot of device metrics:
- Most devices cluster in a central region (normal behavior)
- A few devices are outliers (anomalies)

A random tree splitting randomly would quickly isolate those outliers because they're far from the dense cluster. Normal points, being clustered together, require many more splits to isolate.

---

## How It Works

### Algorithm Overview

1. **Build an Ensemble of Isolation Trees**
   - Each tree is built using random splits
   - Trees are kept small (subsample of data) to maintain randomness
   - Multiple trees vote on anomaly scores

2. **Calculate Anomaly Score**
   - For each data point, compute the average path length across all trees
   - Shorter path length → more anomalous
   - Score is normalized between -1 (most anomalous) and +1 (most normal)

3. **Classification**
   - Points with score < 0 are typically classified as anomalies
   - The exact threshold depends on the `contamination` parameter

### Mathematical Intuition

The anomaly score is based on the **average path length**:

```
score(x) = 2^(-E(h(x)) / c(n))
```

Where:
- `h(x)` = path length of instance x in a tree
- `E(h(x))` = expected (average) path length across all trees
- `c(n)` = normalization constant based on dataset size
- Score ranges: -1 (anomaly) to +1 (normal)

### Path Length Intuition

- **Short path** (few splits needed): Instance is isolated easily → likely anomaly
- **Long path** (many splits needed): Instance is deep in a cluster → likely normal

---

## Key Parameters

### 1. `n_estimators` (Number of Trees)

**Default in project: 300**

- More trees → more stable, accurate scores
- Trade-off: More computation time
- Typical range: 100-500
- Diminishing returns beyond ~300-500 trees

**Effect on this project:**
```python
self.model = IsolationForest(
    n_estimators=300,  # 300 decision trees in the ensemble
    ...
)
```

### 2. `contamination` (Expected Anomaly Proportion)

**Default in project: 0.03 (3%)**

- Proportion of data expected to be anomalous
- Range: 0.0 to 0.5 (50% max)
- Lower values → stricter detection (fewer false positives)
- Higher values → more lenient (more false positives)

**How it works:**
- If `contamination=0.03`, the algorithm expects ~3% of data to be anomalous
- It sets the decision threshold so that ~3% of training data scores below it

**In this project:**
```python
AnomalyDetectorConfig(
    contamination=0.03,  # Expect 3% of device observations to be anomalous
)
```

### 3. `random_state`

**Default in project: 42**

- Controls randomness in tree construction
- Same seed → reproducible results
- Important for debugging and consistency

### 4. `n_jobs` (Parallelization)

**Default in project: -1 (use all CPUs)**

- Number of CPU cores to use for training
- `-1` = use all available cores
- Speeds up training on multi-core machines

### 5. `max_samples` (Subsampling)

**Default: Auto (min(256, n_samples))**

- Number of samples used to build each tree
- Smaller samples → more randomness, faster training
- Helps detect local anomalies better

---

## Advantages & Limitations

### Advantages ✅

1. **No Labels Required**
   - Fully unsupervised - perfect when you don't know what anomalies look like
   - Ideal for IoT device telemetry where anomalies are rare and varied

2. **Handles High Dimensions Well**
   - Unlike distance-based methods, doesn't suffer from curse of dimensionality
   - Works well with many features (your project uses dozens of device metrics)

3. **Fast Training & Inference**
   - Linear time complexity O(n)
   - Efficient for large datasets (millions of device observations)

4. **Handles Mixed Feature Types**
   - Works with numeric features (after appropriate preprocessing)
   - Your project uses StandardScaler for normalization

5. **Robust to Irrelevant Features**
   - Random splits mean irrelevant features don't dominate
   - Feature weighting in your project helps balance importance

### Limitations ⚠️

1. **Assumes Anomalies are Rare**
   - Works best when anomalies are < 50% of data
   - If anomalies are common, performance degrades

2. **Requires Feature Engineering**
   - Still benefits from good feature selection and scaling
   - Your project does feature selection, normalization, and weighting

3. **Difficult to Interpret**
   - Hard to understand WHY a point is anomalous
   - Your project addresses this with LLM-based explanations

4. **Can Miss Clustered Anomalies**
   - If multiple anomalies form their own cluster, they might be treated as normal
   - Your hybrid model combines with temporal detection to mitigate this

5. **Sensitive to Feature Scaling**
   - Works best with normalized features
   - Your project uses `StandardScaler` for this

---

## Isolation Forest in This Project

### Implementation Overview

Your project uses Isolation Forest as the primary anomaly detection algorithm for IoT device telemetry. Here's how it's structured:

#### 1. **Wrapper Class: `AnomalyDetectorIsolationForest`**

Location: `src/device_anomaly/models/anomaly_detector.py`

```python
class AnomalyDetectorIsolationForest:
    """Thin wrapper around IsolationForest to keep things tidy."""
```

**Key Features:**
- Wraps sklearn's `IsolationForest` with project-specific preprocessing
- Handles feature selection, scaling, and weighting
- Manages missing value imputation

#### 2. **Configuration**

```python
@dataclass
class AnomalyDetectorConfig:
    contamination: float = 0.03      # 3% expected anomalies
    n_estimators: int = 300          # 300 trees
    random_state: int = 42           # Reproducibility
    min_variance: float = 1e-6       # Drop constant features
    scale_features: bool = True      # Use StandardScaler
    feature_domain_weights: dict[str, float] | None = None
```

#### 3. **Feature Selection Strategy**

Your implementation intelligently selects features:

1. **Prefer cohort-normalized features** (`*_cohort_z`) if available
2. **Fallback to baseline-normalized** (`*_z_*`) features
3. **Otherwise use raw numeric telemetry** columns
4. **Exclude** IDs, labels, and flags
5. **Drop near-constant features** (variance < 1e-6)

```python
def _select_feature_columns(self, df: pd.DataFrame) -> list[str]:
    # Prefer cohort-normalized features if they exist
    cohort_cols = [c for c in candidates if c.endswith("_cohort_z")]
    baseline_cols = [c for c in candidates if "_z_" in c]
    feature_cols = baseline_cols or cohort_cols or candidates
```

#### 4. **Feature Preprocessing Pipeline**

1. **Handle Missing Values**: Impute with median
2. **Apply Domain Weights**: Balance feature importance (e.g., battery vs. CPU)
3. **Drop Low-Variance Features**: Remove near-constant columns
4. **Standard Scaling**: Normalize to mean=0, std=1

```python
def _prepare_training_matrix(self, df: pd.DataFrame) -> np.ndarray:
    # Impute missing values
    feature_df = feature_df.fillna(self.impute_values)
    
    # Apply domain-based feature weighting
    feature_df = self._apply_feature_weights(feature_df)
    
    # Drop near-constant features
    variances = feature_df.var(ddof=0)
    keep_mask = variances > self.config.min_variance
    
    # Scale features
    if self.config.scale_features:
        self.scaler = StandardScaler()
        matrix = self.scaler.fit_transform(feature_df.values)
```

#### 5. **Usage in Hybrid Model**

Your `HybridAnomalyDetector` combines Isolation Forest with temporal pattern detection:

```python
class HybridAnomalyDetector:
    def __init__(self):
        self.global_detector = AnomalyDetectorIsolationForest(...)
        self.temporal_detector = TemporalResidualDetector(...)
        # Optionally trains per-cohort IsolationForest models
        self.cohort_models: Dict[str, AnomalyDetectorIsolationForest] = {}
```

This provides:
- **Spatial anomalies**: Detected by Isolation Forest (unusual feature combinations)
- **Temporal anomalies**: Detected by temporal detector (unusual patterns over time)

#### 6. **Score Interpretation**

From your codebase:

```python
scores = self.score(df_scored)       # Returns decision_function scores
labels = self.predict(df_scored)     # Returns -1 (anomaly) or +1 (normal)

# Scores interpretation:
# - Negative scores → more anomalous (closer to -1 = very anomalous)
# - Positive scores → more normal (closer to +1 = very normal)
# - Score of 0 is approximately the decision boundary
```

**Severity Mapping** (from `base.py`):
```python
severity_thresholds = {
    "critical": -0.5,  # Most extreme anomalies
    "high": -0.3,
    "medium": -0.1,
    "low": 0.0,        # Threshold for any anomaly
}
```

---

## Best Practices

### 1. **Data Quality**

- **Clean missing values**: Your project uses median imputation
- **Remove outliers** that are known errors (e.g., sensor malfunctions)
- **Normalize features**: Your project uses StandardScaler

### 2. **Feature Engineering**

- **Use domain knowledge**: Your project applies feature weights by domain
- **Prefer normalized features**: Cohort/baseline normalization helps
- **Remove irrelevant features**: Your project drops IDs and labels

### 3. **Hyperparameter Tuning**

**Contamination** (`0.03` in your project):
- Start with domain knowledge estimate
- Tune based on validation results
- Consider false positive rate tolerance

**Number of Trees** (`300` in your project):
- More trees = more stable but slower
- 300 is a good balance for most cases
- Monitor if increasing helps

### 4. **Evaluation**

- **Monitor false positive rate**: Too many false alarms?
- **Check detection rate**: Missing real anomalies?
- **Analyze score distribution**: Are scores reasonable?
- **Feature importance**: Which features drive anomalies?

### 5. **Handling Edge Cases**

Your project handles several edge cases well:

- **Near-constant features**: Dropped (variance check)
- **Missing features**: Imputed with median
- **Feature weighting**: Prevents single domain from dominating
- **Multiple cohorts**: Separate models for different device cohorts

### 6. **Production Considerations**

- **Model retraining**: When to retrain? (drift detection in your project)
- **Feature drift**: Monitor feature distributions over time
- **Performance**: Your `n_jobs=-1` uses parallelization
- **Scaling**: Isolation Forest handles large datasets efficiently

---

## Practical Examples

### Example 1: Basic Usage

```python
from device_anomaly.models.anomaly_detector import (
    AnomalyDetectorIsolationForest,
    AnomalyDetectorConfig
)

# Configure
config = AnomalyDetectorConfig(
    contamination=0.03,
    n_estimators=300
)

# Create detector
detector = AnomalyDetectorIsolationForest(config=config)

# Train on normal device data
detector.fit(training_df)

# Score new data
scored_df = detector.score_dataframe(new_df)

# Filter anomalies
anomalies = scored_df[scored_df['anomaly_label'] == -1]
```

### Example 2: Interpreting Scores

```python
# After scoring
scored_df = detector.score_dataframe(df)

# Analyze score distribution
print(scored_df['anomaly_score'].describe())
# Typical output:
# min: -0.8 (very anomalous)
# max: 0.5 (very normal)
# mean: ~0.0 (decision boundary)

# Severity breakdown
critical = scored_df[scored_df['anomaly_score'] < -0.5]
high = scored_df[(scored_df['anomaly_score'] < -0.3) & 
                 (scored_df['anomaly_score'] >= -0.5)]
```

### Example 3: Feature Importance

While Isolation Forest doesn't provide direct feature importance, your project uses feature weighting:

```python
# In your codebase, features are weighted by domain:
FeatureConfig.domain_weights = {
    "battery": 0.8,      # Battery features weighted down
    "cpu": 1.2,          # CPU features weighted up
    "memory": 1.0,       # Default weight
}
```

This prevents any single feature domain from dominating the detection.

---

## Common Issues & Solutions

### Issue 1: Too Many False Positives

**Symptoms**: Many normal devices flagged as anomalous

**Solutions**:
- Decrease `contamination` (e.g., 0.01 instead of 0.03)
- Check feature scaling (should be normalized)
- Review feature selection (remove noisy features)
- Consider feature weighting adjustments

### Issue 2: Missing Real Anomalies

**Symptoms**: Known anomalies not detected

**Solutions**:
- Increase `contamination` (e.g., 0.05)
- Add more relevant features
- Check if features capture anomaly patterns
- Consider hybrid approach (temporal + spatial)

### Issue 3: Inconsistent Results

**Symptoms**: Scores change between runs

**Solutions**:
- Set `random_state` (you already do: `42`)
- Ensure same features used for training and inference
- Check for data drift

### Issue 4: Slow Performance

**Symptoms**: Training/inference takes too long

**Solutions**:
- Reduce `n_estimators` (try 100-200)
- Use `n_jobs=-1` for parallelization (you already do)
- Reduce feature count
- Sample training data if very large

---

## Comparison with Other Methods

### vs. Z-Score / Statistical Methods

| Aspect | Isolation Forest | Z-Score |
|--------|-----------------|---------|
| Assumptions | Minimal | Normal distribution |
| Multivariate | Yes | Limited |
| High dimensions | Excellent | Poor |
| Non-linear patterns | Captures | Misses |
| Speed | Fast | Very fast |

**Your project**: Uses both! Hybrid model combines Isolation Forest with statistical methods.

### vs. One-Class SVM

| Aspect | Isolation Forest | One-Class SVM |
|--------|-----------------|---------------|
| Speed | Fast | Slower |
| Scalability | Excellent | Limited |
| Hyperparameters | Few | Many |
| Non-linear | Yes (via trees) | Via kernel |

**Your project**: Isolation Forest chosen for scalability and efficiency.

### vs. DBSCAN / Clustering

| Aspect | Isolation Forest | DBSCAN |
|--------|-----------------|--------|
| Outlier detection | Explicit | Implicit |
| Distance metric | Not needed | Required |
| High dimensions | Good | Poor |
| Clustered anomalies | Can miss | Can detect |

**Your project**: Hybrid approach helps detect both isolated and clustered anomalies.

---

## Advanced Topics

### 1. **Path Length Distribution**

The core of Isolation Forest is analyzing path length distributions:

```
Normal point:     Long path (many splits to isolate)
                  ████████████████████████████
                  
Anomaly:          Short path (few splits to isolate)
                  ███
```

### 2. **Score Calibration**

Your project includes `IsoScoreCalibrator` for score calibration:

- Maps raw Isolation Forest scores to calibrated probabilities
- Helps with interpretability and threshold selection

### 3. **Cohort-Specific Models**

Your `HybridAnomalyDetector` trains separate models per cohort:

```python
# Train per-cohort models for better accuracy
for cohort_id, cohort_df in cohorts.items():
    if len(cohort_df) >= min_samples:
        model = AnomalyDetectorIsolationForest(...)
        model.fit(cohort_df)
        self.cohort_models[cohort_id] = model
```

**Benefits**:
- Different device types have different normal behaviors
- More accurate detection within each cohort
- Reduces false positives from cross-cohort differences

### 4. **Feature Weighting Strategy**

Your project applies domain-based feature weights:

```python
def _apply_feature_weights(self, df: pd.DataFrame) -> pd.DataFrame:
    # Prevent single domain from dominating
    # E.g., battery metrics might vary wildly but shouldn't dominate
    weight_series = pd.Series(self.feature_weights)
    return df.mul(aligned, axis=1)
```

**Rationale**: Some feature domains (like battery) have high variance but might not be most important for anomaly detection.

---

## References

### Papers

1. **Original Paper**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." ICDM 2008.
   - Introduced the algorithm
   - Explains the theoretical foundation

2. **Extended Paper**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2012). "Isolation-Based Anomaly Detection." ACM Transactions on Knowledge Discovery from Data.
   - Extended analysis and applications

### Documentation

- **scikit-learn**: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
- **Your project's implementation**: `src/device_anomaly/models/anomaly_detector.py`

### Code References in Your Project

- **Main implementation**: `src/device_anomaly/models/anomaly_detector.py`
- **Base interface**: `src/device_anomaly/models/base.py`
- **Hybrid usage**: `src/device_anomaly/models/hybrid.py`
- **Configuration**: `src/device_anomaly/config/model_config.py`

---

## Summary

**Isolation Forest** is a powerful, efficient algorithm for anomaly detection that:

1. ✅ Works without labeled data (unsupervised)
2. ✅ Handles high-dimensional data well
3. ✅ Is fast and scalable
4. ✅ Detects anomalies by isolating them (few splits = anomaly)

**In your project**, Isolation Forest is:
- Configured with 300 trees and 3% contamination
- Enhanced with feature selection, scaling, and weighting
- Combined with temporal detection in a hybrid model
- Used with cohort-specific models for better accuracy

**Key takeaway**: Isolation Forest excels at finding "different" points in high-dimensional space, making it ideal for detecting unusual device behavior from telemetry data.

---

*Last updated: Based on codebase analysis of AnomalyDetection project*
