
# FaceSwap Demo

## Description
FaceSwap Demo is a Python-based tool for swapping faces in images. It utilizes advanced computer vision and deep learning techniques to detect faces in images and seamlessly replace them with other faces.

## Features
- Face detection and alignment
- Face swapping between two images
- Support for multiple face swaps in a single image
- Easy-to-use command-line interface

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package installer)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/omelahealth/faceswap-demo.git
   cd faceswap-demo
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To use the FaceSwap Demo, run the following command:

```
python faceswap.py -source ./test/profile.jpg -target ./test/raw_image_0.png
```

Replace `<source_image_path>` and `<target_image_path>` with the paths to your source and target images.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

