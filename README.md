# Depth Measurement Tool

A Python application that uses the Dense Prediction Transformer (DPT) model to measure real-world distances between points in a monocular image. This tool is particularly useful for measuring objects in top-down view images when you know the camera height and parameters.

## Features

- Load and analyze images using DPT for depth estimation
- Interactive point-and-click measurement interface
- Real-world distance calculations in centimeters
- Image navigation with zoom and pan capabilities
- Support for custom camera parameters
- Debug information for measurement verification

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers library
- OpenCV
- PIL (Pillow)
- NumPy
- Tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd dense-prediction-transformer
```

2. Install required packages:
```bash
poetry install
```

## Usage

1. Run the application:
```bash
python depth_measurement_app.py
```

2. Configure camera parameters:
   - Focal Length (mm): Your camera's focal length (check image metadata)
   - Camera Height (mm): Height of camera from the measured surface
   - Sensor Width (mm): Your camera's sensor width (typically 6.4mm for mobile phones)

3. Navigate the image:
   - Zoom: Use mouse wheel or zoom slider
   - Pan: Click and drag to move around
   - Reset: Click "Reset View" to return to original size

4. Measure distances:
   - Click "Load Image" to open your image
   - Click two points on the image to measure the distance between them
   - The distance will be displayed in centimeters
   - Use "Clear Points" to remove existing measurements

## Tips for Accurate Measurements

1. Use top-down view images for best results
2. Ensure camera height is measured accurately
3. Use correct camera parameters (focal length, sensor width)
4. For better precision, zoom in when placing measurement points
5. Check debug information in console for measurement verification

## Known Limitations

- Requires accurate camera parameters for precise measurements
- Works best with top-down view images
- Depth estimation may vary based on image quality and lighting
- Some DPT model weights are initialized randomly (see console warnings)

## Troubleshooting

If measurements seem incorrect:
1. Verify camera parameters are entered correctly in millimeters
2. Check camera height measurement
3. Ensure image is taken from a proper top-down angle
4. Look at debug information printed in the console
5. Try measuring a known distance to calibrate the system

## Contributing

Feel free to open issues or submit pull requests with improvements.
