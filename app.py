import torch
import numpy as np
import cv2
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image

# Load the model
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# Camera intrinsics (assuming a pinhole model)
FOCAL_LENGTH = 6.81  # Adjust based on camera specs
PRINCIPAL_POINT = (650, 850)  # Assuming image center

def get_depth_map(image_path):
    """Generate depth map from an image."""
    image_raw = Image.open(image_path)
    image = image_raw.resize(
        (800, int(800 * image_raw.size[1] / image_raw.size[0])),
        Image.Resampling.LANCZOS,
    )

    encoding = feature_extractor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)
        predicted_depth = outputs.predicted_depth

    # Resize to original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

def pixel_to_3d(x, y, depth):
    """Convert pixel coordinates (x, y) and depth value to 3D world coordinates."""
    X = (x - PRINCIPAL_POINT[0]) * depth / FOCAL_LENGTH
    Y = (y - PRINCIPAL_POINT[1]) * depth / FOCAL_LENGTH
    Z = depth  # The depth map gives us Z directly
    return np.array([X, Y, Z])

def compute_distance(image_path, point1, point2):
    """Compute real-world distance between two points given an image and depth map."""
    image_raw = Image.open(image_path)
    original_size = image_raw.size  # (width, height)
    resized_width = 800
    resized_height = int(800 * original_size[1] / original_size[0])

    # Compute scale factors
    scale_x = resized_width / original_size[0]
    scale_y = resized_height / original_size[1]

    # Scale points to match resized image
    point1_scaled = (int(point1[0] * scale_x), int(point1[1] * scale_y))
    point2_scaled = (int(point2[0] * scale_x), int(point2[1] * scale_y))

    depth_map = get_depth_map(image_path)

    # Extract depth at rescaled points
    d1 = depth_map[point1_scaled[1], point1_scaled[0]]
    d2 = depth_map[point2_scaled[1], point2_scaled[0]]

    # Convert pixels to real-world coordinates
    p1_3d = pixel_to_3d(point1_scaled[0], point1_scaled[1], d1)
    p2_3d = pixel_to_3d(point2_scaled[0], point2_scaled[1], d2)

    # Compute Euclidean distance
    distance = np.linalg.norm(p2_3d - p1_3d)
    return distance


# Example usage
image_path = "input/6.jpg"
point1 = (275, 1035)  # Example point in pixels (x1, y1)
point2 = (914, 1035)  # Example point in pixels (x2, y2)

distance = compute_distance(image_path, point1, point2)
print(f"Estimated real-world distance between points: {distance:.2f} units")
