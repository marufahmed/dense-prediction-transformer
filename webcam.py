import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Canvas

# Camera intrinsic parameters (example values, replace with actual)
FOCAL_LENGTH = 500  # in mm
SENSOR_WIDTH = 36  # in mm (depends on the camera)
IMAGE_WIDTH = 1920  # in pixels
PIXEL_SIZE = SENSOR_WIDTH / IMAGE_WIDTH  # mm per pixel

# Global variables
points = []
image = None

def capture_image():
    global image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture image.")
        return
    
    image = frame
    cv2.imwrite("captured_image.jpg", frame)
    open_image("captured_image.jpg")

def open_image(filepath):
    global image
    image = cv2.imread(filepath)
    if image is None:
        print("Error: Could not open image.")
        return
    
    cv2.imshow("Image Viewer", image)
    cv2.setMouseCallback("Image Viewer", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def click_event(event, x, y, flags, param):
    global points, image
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image Viewer", image)
        
        if len(points) == 2:
            calculate_distance()
            points.clear()

def calculate_distance():
    (x1, y1), (x2, y2) = points
    pixel_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    real_distance_cm = (pixel_distance * PIXEL_SIZE) / 10  # Convert mm to cm
    print(f"Measured Distance: {real_distance_cm:.2f} cm")

def main():
    root = tk.Tk()
    root.title("Webcam Measurement App")
    
    btn_capture = tk.Button(root, text="Capture Image", command=capture_image)
    btn_capture.pack()
    
    btn_open = tk.Button(root, text="Open Image", command=lambda: open_image(filedialog.askopenfilename()))
    btn_open.pack()
    
    root.mainloop()

if __name__ == "__main__":
    main()
