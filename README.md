# Tongue AI - Medical Image Analysis

A web application for analyzing tongue images using deep learning segmentation and computer vision techniques to provide health insights.

## Technologies Used

### Deep Learning & AI
- **U-Net** - Convolutional neural network for tongue image segmentation
- **PyTorch** - Deep learning framework for model training and inference
- **OpenAI Vision API** (optional) - AI-powered image analysis

### Computer Vision & Image Processing
- **OpenCV** - Image processing, color space conversion, edge detection, and texture analysis
- **NumPy** - Numerical operations for color and texture analysis
- **Pillow** - Image handling and format conversion

### Web Framework
- **Flask** - Python web framework for the application backend

### Key Features
- **Image Segmentation** - U-Net model isolates tongue region from background
- **Color Analysis** - RGB/HSV analysis for coating, redness, and color variations
- **Texture Analysis** - Variance-based detection of dryness and surface irregularities
- **Edge Detection** - Canny edge detection for crack and fissure identification
- **Health Insights** - Automated analysis of tongue characteristics with recommendations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the segmentation model (optional if model already exists):
```bash
python training/train.py
```

3. Run the application:
```bash
python app.py
```

## Usage

Upload a tongue image through the web interface. The system will:
1. Segment the tongue using the U-Net model
2. Analyze color, texture, and visual features
3. Generate a health report with findings and recommendations

**Note:** This tool provides visual analysis only and is not a medical diagnosis. Consult healthcare professionals for medical concerns.

