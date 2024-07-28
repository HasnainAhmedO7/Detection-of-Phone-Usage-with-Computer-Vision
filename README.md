### Phone Usage Detection with Computer Vision

This project utilizes computer vision techniques to detect phone usage activity by combining face mesh analysis with object detection. It leverages MediaPipe for facial landmarks and YOLOv8 for object detection to identify when a person is using a phone.

### Project Overview

The main objective of this project is to monitor and detect phone usage activity using a live video feed. The system analyzes the orientation of a person's head and detects objects in the video stream to infer if a person is using a cell phone. The following technologies are used:

#### MediaPipe: For facial landmark detection and head pose estimation.

#### YOLOv8: For real-time object detection to identify phones.

### Setup

#### Clone the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
Create and Activate a Virtual Environment
```

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

#### Install Required Packages

Make sure to have requirements.txt in your repository with the following content:

opencv-python
mediapipe
numpy
ultralytics
Install the packages using pip:

```bash
pip install -r requirements.txt
```

### Usage

#### Run the Script

```bash
python your_script.py
```

This script will open your webcam and start processing the video feed. It will:

#### Detect facial landmarks and head pose using MediaPipe.

Use YOLOv8 to identify objects (e.g., phones) in the video.
Display information on the screen about head orientation and detected objects.
Interpreting the Output

The system will display a live video feed with annotations indicating:
Head orientation (Looking Left, Looking Right, Looking Down, Looking Up, or Forward).
Detected objects with bounding boxes and labels.
If the head orientation indicates looking down and a cell phone is detected, a message will be shown indicating that the person is using a phone.
Stopping the Script

To stop the script, press the ESC key while the video window is active.

### Troubleshooting

Camera Not Detected:
Ensure your camera is properly connected and accessible. Check permissions if running on a system with restricted camera access.

#### Dependencies Issues:

Verify that all dependencies are correctly installed and compatible with your Python version.

#### Model File Issues:

Ensure that yolov8n.pt is correctly downloaded and located in the same directory as the script or update the path accordingly.

For any questions or contributions, please contact hasnainahmed1947@gmail.com
