# Yellow Sticky Trap Insect Detection App

This Streamlit application provides a user-friendly interface for detecting insects on yellow sticky traps.

## Features

- Upload images of yellow sticky traps
- Adjust detection parameters (confidence threshold, tile size, etc.)
- Visualize detection results with bounding boxes
- Download annotated images and XML annotations for further use

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the trained model in the `model` directory

## Usage

Run the Streamlit app:

```bash
python run.py
```

Or directly with Streamlit:

```bash
streamlit run app.py
```

## Interface Guide

1. **Upload**: Click on the upload button to select an image
2. **Settings**: Adjust the detection parameters in the sidebar
3. **Run Detection**: Click the button to process the image
4. **Results**: View the detections and statistics
5. **Download**: Save the annotated image and XML annotation file

## Annotation Format

The application generates annotations in Pascal VOC XML format, which can be used for training object detection models.
