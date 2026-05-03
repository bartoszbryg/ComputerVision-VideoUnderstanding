# ComputerVision-VideoUnderstanding

## Overview
This project is a computer vision system that detects, tracks, and describes objects in video.

It combines:
- YOLOv8 for object detection
- DeepSORT for multi-object tracking
- BLIP for image captioning
- Motion analysis using object history

## Features
- Detects objects (person, car, bicycle, etc.)
- Tracks objects across frames with consistent IDs
- Generates natural language descriptions of scenes
- Detects movement direction (left, right, up, down)

## Pipeline

**Video → YOLO Detection → DeepSORT Tracking → Motion Analysis → NLP → Output**

The system processes video frame-by-frame using the following pipeline:

This pipeline ensures that object identity and motion are preserved across frames, enabling more meaningful scene understanding.

The system processes video frame-by-frame using the following pipeline:

1. **Input (main.py)**
   - Reads video frames using OpenCV
   - Controls the overall pipeline execution

2. **Object Detection (detector.py)**
   - Uses YOLOv8 to detect objects in each frame
   - Outputs bounding boxes, labels, and confidence scores

3. **Object Tracking (tracker.py)**
   - Uses DeepSORT to assign consistent IDs to detected objects
   - Maintains object identity across frames
   - Stores position history for motion analysis

4. **Motion Analysis (tracker.py + nlp.py)**
   - Uses stored object history to determine movement direction
   - Example: moving left, right, up, or down

5. **Natural Language Processing (nlp.py)**
   - Uses BLIP to generate a caption for selected frames
   - Combines caption with detected objects and motion data
   - Produces human-readable descriptions

6. **Visualization (drawer.py)**
   - Draws bounding boxes and object IDs on frames
   - Displays captions on the video

7. **Output (utils.py + main.py)**
   - Saves annotated video
   - Saves structured JSON data
   - Saves text descriptions

## How to Run

```bash
python main.py --video input.mp4
