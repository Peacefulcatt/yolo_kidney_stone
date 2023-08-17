import cv2
import numpy as np
from ultralytics import YOLO

def draw_grids(image, spacing, color=(255, 255, 255), alpha=0.3):
    overlay = image.copy()
    height, width = overlay.shape[:2]

    for x in range(0, width, spacing):
        cv2.line(overlay, (x, 0), (x, height), color, 1)

    for y in range(0, height, spacing):
        cv2.line(overlay, (0, y), (width, y), color, 1)

    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

# Load the YOLOv8 model
model = YOLO('/home/peacefulcat/PycharmProjects/yolokidneystone/runs/detect/train2/weights/best.pt')

# Open the video file
video_path = "/home/peacefulcat/sample1.mp4"
cap = cv2.VideoCapture(video_path) # To use webcam cv2.VideoCapture(0)

# Initialize the pause flag
paused = False

# Grid settings
diameter= 2.3
grid_spacing = 50  # Adjust this value to control the spacing between grid lines
grid_color = (255, 255, 255)  # White color for the grid lines
grid_alpha = 0.2  # Transparency of the grid overlay

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Apply YOLOv8 inference on the frame
        results = model(frame)

        # Draw grids on the frame
        frame_with_grids = draw_grids(frame, grid_spacing, grid_color, grid_alpha)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(frame=frame_with_grids,diameter=diameter,grid_spacing=grid_spacing,grid_alpha=0.3,grid_color=(255,255,255))


        # Combine the frame with grids and the annotated frame
        combined_frame = cv2.addWeighted(annotated_frame, 0.8, frame_with_grids, 0.8, 0)

        cv2.putText(combined_frame, f"Grid Space: {grid_spacing} pixels", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # Display the combined frame
        cv2.putText(combined_frame, f"Diameter: {diameter:.1f} mm", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("YOLOv8 Inference with Grids", combined_frame)

        # Check for key press events
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        elif key == 49:  # Press '1' key to increase grid density
            grid_spacing += 10
        elif key == 50:  # Press '2' key to decrease grid density
            if grid_spacing > 10: # Prevents grid_spacing to be 0
                grid_spacing -= 10
        elif key == ord('4'):  # Press '4' key
            diameter += 0.3
        elif key == ord('5'): # Press '5' key
            if diameter > 0.1:
                diameter -= 0.3
        elif key == ord("p"):
            paused = not paused

        # Pause if the pause flag is True
        while paused:
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
