import cv2
import numpy as np

grid_space = 70  # Adjust this value according to your specific case
diameter= 0.2


def detect_blue_object_rectangle(frame, lower_blue, upper_blue, max_bounding_width=200, max_bounding_height=200):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the biggest contour is the blue object
        biggest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest_contour)

        # Limit the bounding box width and height to the specified values
        if w > max_bounding_width:
            w = max_bounding_width
        if h > max_bounding_height:
            h = max_bounding_height

        return x, y, w, h

    return None

def calculate_touching_arm_heights(contour, x, y, w, h, num_pixels=10):
    touching_arm_heights = []

    # Calculate heights along both left and right touching arms
    for direction in [1, -1]:  # 1 for right, -1 for left
        arm_x = x if direction == -1 else x + w
        arm_points = [point[0][1] for point in contour if arm_x - num_pixels <= point[0][0] <= arm_x + num_pixels]

        if arm_points:
            arm_height = max(arm_points) - min(arm_points)
            touching_arm_heights.append(arm_height)

    return touching_arm_heights


def detect_blue_object(frame, lower_blue, upper_blue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assuming the biggest contour is the blue object
        biggest_contour = max(contours, key=cv2.contourArea)
        return biggest_contour

    return None


def draw_transparent_grids(frame, blue_object_width, alpha=0.5):
    overlay = frame.copy()
    height, width, _ = frame.shape
    rows = int(height / blue_object_width)
    cols = int(width / blue_object_width)

    for i in range(1, rows):
        y = i * blue_object_width
        cv2.line(overlay, (0, y), (width, y), (255, 255, 255), 1)

    for i in range(1, cols):
        x = i * blue_object_width
        cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 1)

    # Add the overlay with transparency to the original frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def main():
    paused = False
    video_path = "/home/peacefulcat/sample4.mp4"
    cap = cv2.VideoCapture(video_path)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    lower_blue_rectangle = np.array([100, 50, 50])
    upper_blue_rectangle = np.array([130, 255, 255])
    while True:
        global grid_space,diameter
        key = cv2.waitKey(30)
        if key == ord('p'):  # Press 'p' key to pause the video
            paused = not paused
        if not paused:
            ret,frame =cap.read()
            if not ret:
                break

            blue_object_contour = detect_blue_object(frame, lower_blue, upper_blue)
            blue_object_info = detect_blue_object_rectangle(frame, lower_blue_rectangle, upper_blue_rectangle)
            if blue_object_contour is not None:
                x, y, w, h = blue_object_info

                # Draw the rectangular bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

                # Draw the segmentation boundaries using the detected contour, limited within the bounding box
                contour_to_draw = blue_object_contour.copy()

                # Loop through the contour points and only consider points within the bounding box
                for point in contour_to_draw:
                    px, py = point[0]
                    if x <= px < x + w and y <= py < y + h:
                        point[0][0] = px
                        point[0][1] = py
                    else:
                        point[0][0] = x + w - 1
                        point[0][1] = y + h - 1

                # Draw the green segmentation boundaries on the frame
                cv2.drawContours(frame, [contour_to_draw], 0, (0, 255, 0), 1)  # Green segmentation boundaries

                # Calculate the height along the left touching arm
                touching_arm_heights = calculate_touching_arm_heights(blue_object_contour, x, y, w, h, num_pixels=15)

                # Display the height along the left touching arm
                # Display the heights along the touching arms
                if len(touching_arm_heights) >= 2:
                    cv2.putText(frame, f"Height (Left Touching Arm): {touching_arm_heights[1]} pixels", (0, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(frame, f"Height (Right Touching Arm): {touching_arm_heights[0]} pixels", (0, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv2.putText(frame, f"Height: {min(touching_arm_heights)} pixels", (0, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, "Height: N/A", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                draw_transparent_grids(frame, grid_space)
                cv2.putText(frame, f"Grid Space: {grid_space} pixels", (0,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 255, 255), 1)
                cv2.putText(frame, f"Diameter: {diameter:.1f}mm", (0, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            if key == 27:    # Press 'esc' key to exit
                break
            elif key == 49:  # Press '1' key to increase grid density
                grid_space += 10
            elif key == 50:  # Press '2' key to decrease grid density
                grid_space -= 10
            elif key == ord('4'):  # Press '4' key
                diameter += 0.1
            elif key == ord('5'):  # Press '5' key
                diameter -= 0.1

            cv2.imshow("Blue Object Detection", frame)


        if key == 27:  # Press 'esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
