# CVPR-HAM10000-CNN-22-46247-1

This project implements a custom Convolutional Neural Network (CNN) using PyTorch for multi-class skin lesion classification on the HAM10000 dataset.

## Features
- Custom CNN architecture (with BatchNorm & Dropout)
- Data preprocessing using PyTorch Dataset & DataLoader
- Training with Adam optimizer and learning rate scheduler
- Evaluation on test set with classification report and confusion matrix
- Per-class performance analysis (best & worst classes)
- Model saving (`best_model.pth`)

## Results
The model achieves moderate overall accuracy but shows strong bias toward a dominant class due to dataset imbalance. Several classes receive no predictions, leading to undefined precision values. This highlights limitations in handling imbalanced data.

## Files
- `CNN_22-46247-1.ipynb` — main notebook
- `best_model.pth` — best trained model
- `baseline_model.pth` — baseline model (for comparison)

## Technologies
PyTorch, Python, NumPy, Matplotlib, Seaborn, Scikit-learn, Google Colab
