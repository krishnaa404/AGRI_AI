# Agri AI — Tomato Leaf Disease Classification

A deep learning-based tomato leaf disease classification model using EfficientNetB0, transfer learning, and fine-tuning.

The final model classifies tomato leaf images into four disease categories and achieves **95.63% test accuracy** on a test set of 1,168 images.

## Overview

Agri AI is a computer vision project for classifying tomato leaf diseases from images.

The model uses **EfficientNetB0** with a two-stage training strategy:

1. Transfer learning with the pretrained backbone frozen.
2. Fine-tuning of the upper EfficientNetB0 layers.

### Supported Classes

- Early Blight
- Late Blight
- Spider Mites
- Target Spot

## Model

| Property | Value |
|---|---|
| Architecture | EfficientNetB0 |
| Training Method | Transfer Learning + Fine-Tuning |
| Input Size | 224 × 224 × 3 |
| Number of Classes | 4 |
| Framework | TensorFlow / Keras |

## Performance

The final model was evaluated on 1,168 previously unseen test images.

| Metric | Score |
|---|---:|
| Test Accuracy | **95.63%** |
| Test Loss | 0.1043 |
| Macro F1-Score | **95.65%** |
| Weighted F1-Score | **95.67%** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Early Blight | 98.91% | 92.81% | **95.76%** |
| Late Blight | 97.32% | 97.65% | **97.49%** |
| Spider Mites | 98.89% | 93.06% | **95.89%** |
| Target Spot | 88.58% | 98.97% | **93.49%** |

## Confusion Matrix

The confusion matrix below shows the model's predictions on the 1,168-image test set.

| Actual \ Predicted | Early Blight | Late Blight | Spider Mites | Target Spot |
|---|---:|---:|---:|---:|
| **Early Blight** | **271** | 7 | 0 | 14 |
| **Late Blight** | 3 | **291** | 1 | 3 |
| **Spider Mites** | 0 | 0 | **268** | 20 |
| **Target Spot** | 0 | 1 | 2 | **287** |

The model correctly classified **1,084 out of 1,168** test images, resulting in a test accuracy of **95.63%**.

The largest sources of misclassification were:

- 20 Spider Mites images predicted as Target Spot
- 14 Early Blight images predicted as Target Spot
- 7 Early Blight images predicted as Late Blight

## Dataset

The dataset contains **7,756 unique images** across four classes.

| Split | Images |
|---|---:|
| Training | 5,427 |
| Validation | 1,161 |
| Testing | 1,168 |
| **Total** | **7,756** |

The dataset is not included in this repository.

## Training

### Stage 1 — Transfer Learning

The pretrained EfficientNetB0 backbone was frozen while the classification head was trained for the four target classes.

### Stage 2 — Fine-Tuning

The upper layers of EfficientNetB0 were unfrozen and fine-tuned using a low learning rate to adapt the pretrained features to tomato leaf disease classification.

The best checkpoint from Stage 2 was selected based on validation performance.

## Project Structure

```text
Agri-AI/
│
├── README.md
│
├── model/
│   └── tomato_exp5_stage2_best.keras
│
├── src/
│   └── predict.py
│
├── results/
│   ├── confusion_matrix.png
│   └── classification_report.txt
│
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Agri-AI.git
cd Agri-AI
```

Create a virtual environment:

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Prediction

After installing the dependencies, a new tomato leaf image can be passed to the prediction script:

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```text
Prediction: Early Blight
Confidence: 96.42%
```

The exact command depends on the implementation of `src/predict.py`.

## Trained Model

The final model is:

```text
tomato_exp5_stage2_best.keras
```

Model details:

```text
Architecture: EfficientNetB0
Input: 224 × 224 × 3
Output: 4 classes
Training: Transfer Learning + Fine-Tuning
Framework: TensorFlow / Keras
Test Accuracy: 95.63%
```

If the model exceeds GitHub's normal file-size limits, Git LFS or a model-hosting platform should be used.

## Limitations

The model's performance may vary depending on image quality, lighting, camera characteristics, leaf orientation, background, disease severity, and field conditions.

The model is trained only for the four classes included in this project. It should not be assumed to recognize diseases outside these classes.

The model is intended for research and educational purposes and should not be treated as a replacement for professional agricultural diagnosis.

## Future Improvements

- Expand the number of supported tomato diseases
- Increase dataset diversity
- Evaluate performance on real-world field images
- Improve Target Spot precision
- Add model explainability using Grad-CAM
- Optimize the model for mobile and edge devices
- Develop a web or mobile application

## Technologies

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Author

**Krishna Patidar**

AI/ML Project — Agri AI