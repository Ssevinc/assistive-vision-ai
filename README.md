# Assistive Vision for the Visually Impaired

A real-time object and text detection system built with YOLOv8 to help visually impaired individuals detect crosswalks, stairs, and survival-critical signs in their environment.

## Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage) (tba)
- [Model Training](#model-training) (tba)
- [Demo](#demo) (tba)
- [Roadmap](#roadmap) (tba)
- [Contributing](#contributing) (tba)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## About the Project
Assistive Vision AI is a computer vision application built to assist visually impaired individuals by detecting and recognizing critical environmental cues such as:
- Street signs
- Crosswalks
- Traffic lights
- Stairs
- Warning/danger signs

The system provides **real-time audio feedback** to alert users of important surroundings, enabling safer and more independent navigation in urban environments.

---

## Features
- **Real-time object detection** for essential navigation cues
- **Text detection (OCR)** for reading street and facility signs
- **Audio feedback system** for immediate alerts
- Built with **deep learning** and **computer vision**
- Modular code structure for easy updates and expansions

---

## Tech Stack
- **Programming Language:** Python

- **Frameworks & Libraries:**  
  - PyTorch / YOLO  
  - OpenCV  
  - EasyOCR / Tesseract OCR  

- **Data Handling:** The dataset was created and annotated using [Roboflow](https://roboflow.com), which provided tools for:
- Collecting and organizing images
- Annotating bounding boxes for target classes
- Automatically splitting into train/validation/test sets
- Exporting in YOLO-compatible format

Download link: [dataset-z6sm4](https://universe.roboflow.com/seyyide/dataset-z6sm4)

- **Audio Output:** pyttsx   
- **Model Management:** Git LFS (for large files)

- **Deployment:** (Planned) Streamlit / Mobile App

---

## 🚀 Installation
```bash
# Clone the repo
git clone https://github.com/Ssevinc/assistive-vision-ai.git
cd assistive-vision-ai

# Install dependencies
pip install -r requirements.txt