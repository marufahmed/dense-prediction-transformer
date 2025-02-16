import torch
import numpy as np
import cv2
from transformers import DPTImageProcessor, DPTForDepthEstimation
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class DepthMeasurementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Measurement Tool")
        
        # Initialize model
        self.feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        
        # Camera parameters (in millimeters)
        self.focal_length = tk.StringVar(value="2.35")  # mm
        self.camera_height = tk.StringVar(value="900")  # mm (90cm)
        self.sensor_width = tk.StringVar(value="6.4")   # mm (typical for mobile phones)
        
        # Setup UI
        self.setup_ui()
        
        # State variables
        self.image_path = None
        self.image = None
        self.photo = None
        self.depth_map = None
        self.points = []
        self.scale_x = 1.0
        self.scale_y = 1.0
        
    def setup_ui(self):
        # Parameters Frame
        param_frame = tk.Frame(self.root)
        param_frame.pack(pady=5, padx=5, fill="x")
        
        tk.Label(param_frame, text="Focal Length (mm):").pack(side="left", padx=5)
        tk.Entry(param_frame, textvariable=self.focal_length, width=10).pack(side="left", padx=5)
        
        tk.Label(param_frame, text="Camera Height (mm):").pack(side="left", padx=5)
        tk.Entry(param_frame, textvariable=self.camera_height, width=10).pack(side="left", padx=5)
        
        tk.Label(param_frame, text="Sensor Width (mm):").pack(side="left", padx=5)
        tk.Entry(param_frame, textvariable=self.sensor_width, width=10).pack(side="left", padx=5)
        
        # Buttons Frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5, fill="x")
        
        tk.Button(btn_frame, text="Load Image", command=self.load_image).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Clear Points", command=self.clear_points).pack(side="left", padx=5)
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(expand=True, fill="both", padx=5, pady=5)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Status bar
        self.status_var = tk.StringVar(value="Click 'Load Image' to begin...")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        
        if self.image_path:
            self.clear_points()
            self.image = Image.open(self.image_path)
            self.resize_and_display_image()
            self.depth_map = self.get_depth_map()
            self.status_var.set("Image loaded. Click points to measure distances.")
    
    def get_depth_map(self):
        """Generate depth map from the loaded image."""
        # Resize image to 800px width while maintaining aspect ratio
        image = self.image.resize((800, int(800 * self.image.size[1] / self.image.size[0])),
                                Image.Resampling.LANCZOS)
        
        # Get depth prediction
        encoding = self.feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**encoding)
            predicted_depth = outputs.predicted_depth
        
        # Normalize depth values to be in meters
        depth_min = torch.min(predicted_depth)
        depth_max = torch.max(predicted_depth)
        predicted_depth = (predicted_depth - depth_min) / (depth_max - depth_min)
        
        # Scale to match camera height
        camera_height_m = float(self.camera_height.get()) / 1000  # convert mm to meters
        predicted_depth = predicted_depth * camera_height_m
        
        # Resize to match input image dimensions
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        return prediction.cpu().numpy()
    
    def compute_distance(self, point1, point2):
        """Compute real-world distance between two points in millimeters."""
        # Get original image dimensions
        original_size = self.image.size
        depth_map_width = 800
        depth_map_height = int(800 * original_size[1] / original_size[0])
        
        # Scale points to match depth map size
        scale_x = depth_map_width / original_size[0]
        scale_y = depth_map_height / original_size[1]
        
        point1_scaled = (int(point1[0] * scale_x), int(point1[1] * scale_y))
        point2_scaled = (int(point2[0] * scale_x), int(point2[1] * scale_y))
        
        # Get depth values (in meters)
        d1 = self.depth_map[point1_scaled[1], point1_scaled[0]]
        d2 = self.depth_map[point2_scaled[1], point2_scaled[0]]
        
        # Convert to millimeters
        d1 *= 1000
        d2 *= 1000
        
        # Calculate pixel size in mm at the object distance
        focal_length_mm = float(self.focal_length.get())
        sensor_width_mm = float(self.sensor_width.get())
        pixel_size_mm = sensor_width_mm / original_size[0]
        
        # Convert pixel coordinates to real-world coordinates (in mm)
        principal_point = (original_size[0]/2, original_size[1]/2)
        
        # Calculate real-world X and Y coordinates
        x1 = (point1[0] - principal_point[0]) * pixel_size_mm * d1 / focal_length_mm
        y1 = (point1[1] - principal_point[1]) * pixel_size_mm * d1 / focal_length_mm
        z1 = d1
        
        x2 = (point2[0] - principal_point[0]) * pixel_size_mm * d2 / focal_length_mm
        y2 = (point2[1] - principal_point[1]) * pixel_size_mm * d2 / focal_length_mm
        z2 = d2
        
        # Compute Euclidean distance
        distance_mm = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        # Print debug information
        print(f"Debug Info:")
        print(f"Point 1: ({x1:.1f}, {y1:.1f}, {z1:.1f}) mm")
        print(f"Point 2: ({x2:.1f}, {y2:.1f}, {z2:.1f}) mm")
        print(f"Depth values: {d1:.1f}, {d2:.1f} mm")
        print(f"Pixel size at object: {pixel_size_mm:.3f} mm")
        
        return distance_mm
    
    def on_canvas_click(self, event):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        # Convert canvas coordinates to original image coordinates
        x = int(event.x / self.scale_x)
        y = int(event.y / self.scale_y)
        
        self.points.append((x, y))
        
        # Draw point on canvas
        self.canvas.create_oval(event.x-3, event.y-3, event.x+3, event.y+3, 
                              fill="red", outline="white")
        
        if len(self.points) == 2:
            distance_mm = self.compute_distance(self.points[0], self.points[1])
            distance_cm = distance_mm / 10
            self.status_var.set(f"Distance between points: {distance_cm:.1f} cm")
            
            # Draw line between points
            p1_canvas = (self.points[0][0] * self.scale_x, self.points[0][1] * self.scale_y)
            p2_canvas = (self.points[1][0] * self.scale_x, self.points[1][1] * self.scale_y)
            self.canvas.create_line(p1_canvas[0], p1_canvas[1], 
                                  p2_canvas[0], p2_canvas[1], 
                                  fill="yellow", width=2)
            
            self.points = []  # Reset points for next measurement
    

    def resize_and_display_image(self):
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate scaling to fit image in canvas
        img_width, img_height = self.image.size
        self.scale_x = canvas_width / img_width
        self.scale_y = canvas_height / img_height
        scale = min(self.scale_x, self.scale_y)
        
        # Resize image
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        self.scale_x = scale
        self.scale_y = scale
        
        resized_image = self.image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized_image)
        
        # Update canvas
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
    
    def clear_points(self):
        """Clear all points and lines from the canvas."""
        if self.photo:  # Only clear if there's an image
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.points = []
            self.status_var.set("Points cleared. Click to measure new distances.")

def main():
    root = tk.Tk()
    app = DepthMeasurementApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
