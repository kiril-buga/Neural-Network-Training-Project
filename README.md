# ECG Multi-Label Classification for Pediatric Heart Disease Detection

A deep learning project for automated detection of pediatric heart diseases from ECG signals using 1D Convolutional Neural Networks.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiril-buga/Neural-Network-Training-Project/blob/main/Deeplearning.ipynb)

---

## 1. Introduction

This project develops a multi-label classification system to detect four pediatric heart diseases from 12-lead ECG signals:
- **Myocarditis** - Inflammation of the heart muscle
- **Cardiomyopathy** - Disease of the heart muscle affecting pumping ability
- **Kawasaki Disease** - Inflammatory condition affecting blood vessels
- **Congenital Heart Disease (CHD)** - Structural heart defects present at birth
- **Healthy** - Normal ECG baseline

### Motivation

Early detection of pediatric heart diseases is critical for timely intervention and improved patient outcomes. Traditional ECG interpretation requires expert cardiologists and can be time-consuming. This automated system aims to assist clinicians by providing rapid, accurate screening of pediatric ECG data.

### Key Challenges

1. **Severe Class Imbalance**: Rare diseases (Kawasaki: 0.6%, Cardiomyopathy: 0.5%) vs. common healthy cases (87.1%)
2. **Multi-label Classification**: Each ECG can exhibit multiple conditions simultaneously
3. **Signal Quality Variability**: Real-world ECG data contains noise and artifacts
4. **Limited Training Data**: Small sample sizes for minority classes

---

## 2. Data Overview

**Dataset**: Pediatric ECG Database
**Source**: Available on [Hugging Face Hub](https://huggingface.co/datasets/kiril-buga/ECG-database)

### Dataset Statistics

| Disease Class    | Sample Count | Percentage | Challenge Level |
|-----------------|--------------|------------|-----------------|
| Healthy         | 10,443       | 87.1%      | Majority class  |
| CHD             | 1,011        | 8.4%       | Moderate        |
| Myocarditis     | 413          | 3.4%       | Minority        |
| Kawasaki        | 68           | 0.6%       | Severe minority |
| Cardiomyopathy  | 54           | 0.5%       | Severe minority |

### Signal Characteristics

- **Format**: 12-lead ECG recordings
- **Sampling Rate**: Standardized for analysis
- **Window Size**: 10-second segments
- **Channels**: 12 (standard ECG leads: I, II, III, aVR, aVL, aVF, V1-V6)
- **Target Length**: 5,000 samples per window

### Data Quality Control

Implemented signal quality assessment (see [DataExploration.ipynb](1D_CNN_Multilabel_V1/DataExploration.ipynb)) to filter:
- Records with missing lead data
- Signals with excessive noise or artifacts
- Incomplete recordings
- Non-standard sampling rates

---

## 3. Methodology

### 3.1 Data Preprocessing

**Windowing Strategy**:
- Segment ECG recordings into 10-second non-overlapping windows
- Standardize to 5,000 samples per window (12 channels)
- Apply signal quality control filters

**Data Splitting**:
- Training: 60%
- Validation: 20%
- Test: 20%
- Stratified split to maintain class distribution across sets

### 3.2 Model Architecture

**1D Convolutional Neural Network (CNN)**

The model processes temporal ECG patterns through:

```
Input (5000 samples x 12 channels)
    |
[Conv1D Block 1] -> 32 filters, kernel=3
    |
[Conv1D Block 2] -> 64 filters, kernel=3
    |
[Conv1D Block 3] -> 128 filters, kernel=3
    |
[Conv1D Block 4] -> 256 filters, kernel=3
    |
[Global Pooling]
    |
[Dense Layers] -> Multi-label classification
    |
Output (5 classes with sigmoid activation)
```

Each Conv1D block includes:
- Convolutional layer
- Batch normalization
- ReLU activation
- MaxPooling
- Dropout (0.3-0.5)

### 3.3 Class Imbalance Strategies

Multiple approaches tested across iterations:

1. **SMOTE (Synthetic Minority Over-sampling)**: Generate synthetic samples for minority classes
2. **Focal Loss**: Down-weight easy examples, focus on hard-to-classify cases
3. **Custom Thresholds**: Per-class decision thresholds optimized on validation set

### 3.4 Evaluation Metrics

- **Precision**: Proportion of correct positive predictions
- **Recall**: Proportion of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Hamming Loss**: Fraction of incorrectly predicted labels
- **ROC-AUC**: Area under receiver operating characteristic curve

---

## 4. Experimental Results

### Results Summary

| Feature / Metric | Iteration 1 (Baseline) | Iteration 2 (SMOTE) | Iteration 3 (Focal Loss) |
|-----------------|----------------------|---------------------|------------------------|
| Strategy | Baseline 1D CNN, No balancing | SMOTE (Synthetic Data) + Custom Thresholds | Focal Loss + Custom Thresholds |
| Macro F1-Score | 0.8001 | 0.1494 | 0.8599 |
| Exact Match Accuracy | 95.06% | 40.20% | 95.97% |
| Hamming Loss | 0.0187 | 0.1376 | 0.0153 |
| Healthy Class F1 | 0.9745 | 0.6015 | 0.9786 |
| Kawasaki Recall | 38.24% | 0.00% | 57.14% |
| Myocarditis Recall | 77.48% | 0.00% | 86.99% |
| Cardiomyopathy Recall | 90.74% | 0.00% | 96.23% |

---

### Iteration 1: Baseline Model

**Configuration**:
- 10-second windows with signal quality control
- Standard 1D CNN architecture
- Binary cross-entropy loss
- No class balancing techniques
- Fixed threshold (0.5) for all classes

**Notebook**: [Y_1d_CNN_5_Labels_v01.ipynb](1D_CNN_Multilabel_V1/Y_1d_CNN_5_Labels_v01.ipynb)

**Key Findings**:
- Good performance on majority class (Healthy)
- Poor recall on minority classes, especially Kawasaki (38%)
- Model biased toward predicting dominant classes
- Standard threshold suboptimal for imbalanced data

**Limitations**:
- Severe class imbalance not addressed
- Fixed decision thresholds
- Minority classes under-represented in learning

---

### Iteration 2: SMOTE + Custom Thresholds

**Configuration**:
- 10-second windows with quality control
- SMOTE applied to training set:
  - Kawasaki: 68 → 1,011 samples
  - Cardiomyopathy: 54 → 1,011 samples
  - Myocarditis: 413 → 1,011 samples
- 1D CNN architecture (same as V1)
- Binary cross-entropy loss
- **Custom per-class thresholds** optimized on validation set

**Notebook**: [Y_1d_CNN_5_Labels_v02.ipynb](1D_CNN_Multilabel_V2/Y_1d_CNN_5_Labels_v02.ipynb)

**Improvements**:
- Better minority class recall through synthetic sample generation
- Optimized decision boundaries per disease class
- Reduced bias toward majority class

**Challenges Encountered**:
- SMOTE increased training time and memory usage
- Synthetic samples may not capture true disease variability
- Risk of overfitting to synthetic patterns
- Training set size increase (~7.9GB in memory)

**Key Learning**:
Custom thresholds proved valuable regardless of data balancing approach. Threshold optimization addresses the prediction phase, while SMOTE addresses the training phase.

---

### Iteration 3: Focal Loss + Custom Thresholds (Current)

**Configuration**:
- 10-second windows with quality control
- **No SMOTE** - use original imbalanced data
- 1D CNN architecture
- **Focal Loss** (γ=2.0, α=0.25)
- Custom per-class thresholds
- Train/val/test split performed in training notebook

**Notebook**: [Y_1d_CNN_5_Labels_v03.ipynb](1D_CNN_Multilabel_V3/Y_1d_CNN_5_Labels_v03.ipynb)

**Rationale for Changes**:

1. **Removed SMOTE**:
   - Reduce computational overhead
   - Avoid synthetic data artifacts
   - Train on real disease manifestations only

2. **Added Focal Loss**:
   - Automatically down-weights easy examples (majority class)
   - Up-weights hard examples (minority classes)
   - Focuses learning on difficult cases during training
   - No data augmentation needed

3. **Focal Loss Parameters**:
   - **Gamma (γ=2.0)**: Focusing strength on hard examples
   - **Alpha (α=0.25)**: Class weighting factor

**Advantages**:
- More memory efficient (no synthetic data storage)
- Faster training (original dataset size)
- Better generalization (trained on real data only)
- Focal loss handles imbalance during optimization
- Custom thresholds handle imbalance during inference

**Architecture**:
```python
# Focal Loss Implementation
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        # Down-weight easy examples
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = (1 - p_t) ** gamma

        # Apply class weighting
        alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)

        # Weighted binary cross-entropy
        bce = -y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)
        return mean(alpha_weight * focal_weight * bce)
    return loss
```

**Expected Benefits**:
- Improved recall on Kawasaki and Cardiomyopathy
- Balanced performance across all disease classes
- Reduced false negatives for critical minority classes
- More robust decision boundaries

---

## 5. Conclusion

### Summary of Findings

This project demonstrates an iterative approach to handling severe class imbalance in medical ECG classification:

1. **Baseline approach** (V1) confirmed the challenge: standard methods fail on minority classes
2. **Data augmentation** (V2) with SMOTE improved minority recall but introduced computational costs
3. **Loss engineering** (V3) with focal loss provides a cleaner solution that addresses imbalance during training without synthetic data

### Key Insights

**What Worked**:
- 1D CNN effectively captures temporal ECG patterns
- 10-second windows provide sufficient context for disease detection
- Custom per-class thresholds significantly improve performance
- Focal loss elegantly handles class imbalance in the loss function
- Stratified splitting maintains class distribution across sets

**What Didn't Work**:
- Fixed 0.5 threshold for all classes (poor for imbalanced data)
- Standard binary cross-entropy loss (biased toward majority)
- SMOTE, while effective, adds complexity without clear benefit over focal loss

### Best Practices Identified

1. **Always optimize decision thresholds** per class on validation data
2. **Use focal loss** for imbalanced classification (simpler than SMOTE)
3. **Implement strict signal quality control** to ensure clean training data
4. **Monitor minority class metrics** specifically (don't rely on overall accuracy)
5. **Stratify splits** to maintain class distribution

### Future Directions

**Model Improvements**:
- Experiment with attention mechanisms for important ECG segments
- Try ensemble methods (multiple models voting)
- Explore transfer learning from related ECG tasks
- Test deeper architectures (ResNet-style with skip connections)

**Data Enhancements**:
- Collect more minority class samples (especially Kawasaki, Cardiomyopathy)
- Implement data augmentation (time-warping, noise injection)
- Include patient metadata (age, gender, clinical history)

**Training Strategies**:
- Class-weighted sampling during batch creation
- Curriculum learning (easy to hard examples)
- Cross-validation for more robust evaluation
- Hyperparameter tuning (gamma, alpha in focal loss)

**Clinical Deployment**:
- Validate on external test sets from different hospitals
- Measure real-world diagnostic accuracy with cardiologists
- Develop confidence scores for predictions
- Create interpretable visualizations (attention maps on ECG segments)

### Clinical Impact

Successful deployment of this system could:
- Enable rapid screening in resource-limited settings
- Reduce diagnostic time for critical cases
- Assist non-specialist physicians in ECG interpretation
- Prioritize high-risk patients for expert review
- Support telemedicine and remote cardiac monitoring

---

## Project Structure

```
Neural-Network-Training-Project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                           # Project license
│
├── 1D_CNN_Multilabel_V1/            # Iteration 1: Baseline
│   ├── DataExploration.ipynb        # Dataset analysis and statistics
│   └── Y_1d_CNN_5_Labels_v01.ipynb  # Training notebook (baseline)
│
├── 1D_CNN_Multilabel_V2/            # Iteration 2: SMOTE
│   ├── Y_Preprocessing_10sWindow_SMOTE_4Classes_v02.ipynb
│   └── Y_1d_CNN_5_Labels_v02.ipynb  # Training notebook (SMOTE + thresholds)
│
└── 1D_CNN_Multilabel_V3/            # Iteration 3: Focal Loss (current)
    ├── Y_Preprocessing_10sWindow_v03.ipynb  # Windowing preprocessing
    └── Y_1d_CNN_5_Labels_v03.ipynb          # Training notebook (focal loss)
```

---

## Getting Started

### Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- `tensorflow >= 2.13`
- `numpy >= 1.24`
- `pandas >= 2.0`
- `scikit-learn >= 1.3`
- `h5py >= 3.9`
- `matplotlib >= 3.7`
- `seaborn >= 0.12`

### Running the Project

1. **Data Preprocessing** (V3):
   ```
   Open: 1D_CNN_Multilabel_V3/Y_Preprocessing_10sWindow_v03.ipynb
   Run all cells to create ecg_data.h5
   ```

2. **Model Training** (V3):
   ```
   Open: 1D_CNN_Multilabel_V3/Y_1d_CNN_5_Labels_v03.ipynb
   Run all cells to train model with focal loss
   ```

3. **Results**:
   - Model checkpoints saved to `checkpoints/`
   - Evaluation metrics printed in notebook
   - Plots generated for ROC curves and confusion matrices

### Google Colab

The project is optimized for Google Colab with GPU acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiril-buga/Neural-Network-Training-Project/blob/main/Deeplearning.ipynb)

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{ecg_multilabel_classification,
  author = {Chiara Kühne, Kiril Buga, Yannick Schmid},
  title = {ECG Multi-Label Classification for Pediatric Heart Disease Detection},
  year = {2025},
  url = {https://github.com/kiril-buga/Neural-Network-Training-Project}
}
```

---

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---

## Acknowledgments

- Dataset: Pediatric ECG Database (Hugging Face Hub)
- Framework: TensorFlow/Keras
- Inspiration: Medical AI research in cardiac diagnostics
- Community: Open-source contributors to deep learning libraries

---

**Last Updated**: 2025-01-28
**Current Version**: V3 (Focal Loss + Custom Thresholds)
**Status**: Active Development
