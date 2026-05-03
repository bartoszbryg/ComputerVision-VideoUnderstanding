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
Video -> YOLO Detection -> DeepSORT Tracking -> Motion Analysis -> NLP -> Output

## How to Run

```bash
python main.py --video input.mp4
