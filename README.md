# Landmark Detection System
A deep learning-based landmark recognition system developed using TensorFlow and MobileNetV2 for multi-class image classification of famous world landmarks.

## Overview
This project uses transfer learning with MobileNetV2 to identify and classify landmark images with high accuracy. The model is trained on a custom dataset collected using automated image crawling and data augmentation techniques.

## Features
- Multi-class landmark image classification
- Transfer learning using MobileNetV2
- Data augmentation for improved generalization
- Fine-tuning for enhanced model performance
- Top-3 prediction probability display
- Custom image prediction support
- Automated dataset collection pipeline

## Landmarks Included
- Taj Mahal
- Eiffel Tower
- Qutub Minar
- Great Wall of China
- Hawa Mahal
- Golden Temple
- Statue of Liberty

## Technologies Used
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- icrawler

## Project Structure
```plaintext
Landmark-Detection-System/
│
├── train_model.py
├── predict.py
├── download_dataset.py
├── check_dataset.py
├── README.md
└── .gitignore
```

## Model Performance
- Validation Accuracy: ~95%
- Architecture: MobileNetV2
- Technique: Transfer Learning + Fine-Tuning

## Installation
Install the required dependencies:
```bash
pip install tensorflow numpy matplotlib icrawler pillow
```

## Usage
### Train the Model
```bash
python train_model.py
```

### Predict Landmark from Image
```bash
python predict.py
```

## Dataset
The dataset was created using:
- Automated image crawling
- Public landmark images
- Data augmentation techniques

The dataset and trained model files are excluded from this repository due to file size limitations.

## Future Improvements
- Streamlit-based web application
- Real-time webcam landmark detection
- Expanded landmark dataset
- Cloud deployment support

## Author
Asapu Roja Devi Sree
