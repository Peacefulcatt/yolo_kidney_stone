## YOLOv8 Video Inference with Grid Visualization

This code performs object detection using the YOLOv8 model on a video file and overlays a grid onto the frames to assist in analysis. It provides interactive controls to adjust the grid density and the size of detected objects for better visualization.

### Prerequisites

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- Ultralytics (`ultralytics`)

### Installation

1. Install required Python packages using pip:

   ```bash
   pip install opencv-python numpy ultralytics
   ```

2. Make sure you have the YOLOv8 model weights file (`best.pt`) in the specified path:
   ```
   /home/peacefulcat/PycharmProjects/yolo_kidney_stone/weights/best.pt
   ```

3. Replace the `results.py` file in the Ultralytics library:
   Replace the existing `results.py` file located at your computer `/'something'/'bla'/'bla'/venv/lib/python3.10/site-packages/ultralytics/engine/results.py` with the provided `results.py` file.

### Usage

1. Replace `"/home/peacefulcat/sample1.mp4"` with the path to your video file or use the webcam by changing the VideoCapture argument to `0`.

2. Run the script:

   ```bash
   python main.py
   ```

3. The video frames will be displayed with YOLOv8 object detection results and an overlay grid.

### Controls

- Press `q` to exit the video playback.
- Press `1` to increase the grid density (spacing).
- Press `2` to decrease the grid density, with a minimum spacing of 10 pixels.
- Press `4` to increase the size of detected objects.
- Press `5` to decrease the size of detected objects, with a minimum diameter of 0.1 mm.
- Press `p` to pause/unpause the video playback.

### Code Explanation

The provided code reads frames from a video source, applies YOLOv8 object detection using the Ultralytics library, overlays a grid onto the frames, and provides interactive controls to adjust the grid density and object size. Here's a high-level overview of the code:

1. Libraries and Functions: Import necessary libraries, including OpenCV (`cv2`), NumPy (`numpy`), and Ultralytics (`ultralytics`). Define the `draw_grids` function to overlay a grid onto an image.

2. Load YOLOv8 Model: Load the pre-trained YOLOv8 model using the Ultralytics library.

3. Open Video Source: Open the video file or webcam stream using OpenCV's `VideoCapture`.

4. Loop through Frames: Process each frame in the video using a loop.

5. YOLOv8 Inference: Apply YOLOv8 object detection to the current frame using the loaded model.

6. Draw Grids: Overlay a grid on the frame using the `draw_grids` function.

7. Visualize Results: Plot the YOLOv8 detection results on the frame with the grid overlay.

8. Combine Frames: Combine the frame with the detection results and the grid overlay.

9. Display Frames: Display the combined frame with grid and detection results.

10. Interactive Controls: Check for key presses and respond accordingly to adjust grid density and object size. Also, implement pause functionality.

11. Cleanup: Release the video capture object and close the display window when the video processing is complete.

### Customization

You can customize the code by adjusting parameters such as the paths to the YOLOv8 model weights and the video source, grid settings (spacing, color, and alpha), and the initial object diameter. Additionally, you can extend this code to save the processed video frames or perform further analysis on the detected objects.
