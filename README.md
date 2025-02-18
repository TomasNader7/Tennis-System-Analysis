# Tennis-System-Analysis

## Overview
This project analyzes tennis matches using computer vision and deep learning techniques. It leverages YOLO for object detection, Convolutional Neural Networks (CNNs) for classification, and Python-based data processing to extract insights from match footage.

## Features
- **Player Tracking**: Detects and tracks players' movements using YOLO.
- **Ball Detection**: Identifies the tennis ball in real-time.
- **Stroke Classification**: Uses CNNs to classify different types of strokes.
- **Match Statistics**: Extracts key performance metrics such as shot accuracy, player positioning, and rally duration.
- **Visualization**: Generates annotated video outputs with detected objects and tracking paths.

## Technologies Used
- **Python**: Core programming language for implementation.
- **YOLO (You Only Look Once)**: Object detection for player and ball tracking.
- **Convolutional Neural Networks (CNNs)**: Stroke classification.
- **OpenCV**: Video processing and computer vision operations.
- **TensorFlow/Keras**: Deep learning model training and inference.
- **Matplotlib & Seaborn**: Data visualization.

## Models Used
- YOLO v8 for player detection
- Fine Tuned YOLO for tennis ball detection
- Court Key point extraction
- Trained YOLOV5 model: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
- Trained tennis court key point model: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing

## Training
- Tennis ball detector with YOLO: training/tennis_ball_detector_training.ipynb
- Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Tennis-System-Analysis.git
   cd Tennis-System-Analysis
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Ensure you have the necessary YOLO model weights:
   - Download YOLO weights and place them in the models/ directory
4. Run the analysis script:
     ```bash
     python analyze.py --input video.mp4 --output output.mp4

## Requirements
- python3.8
- ultralytics
- pytroch
- pandas
- numpy
- opencv

## Usage
- To analyze a match, provide an input video file.
- The script processes the video frame-by-frame, detecting and tracking players and the ball.
- The output video contains bounding boxes and tracking lines.
- Extracted data is saved in a CSV file for further analysis.
  
## Output Videos
Here is a screenshot from one of the output videos:

![image](https://github.com/user-attachments/assets/e877099f-aaba-414d-ae17-b756a910ad09)


