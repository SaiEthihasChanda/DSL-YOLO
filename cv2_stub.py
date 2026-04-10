"""
Minimal cv2 stub using PIL and numpy - works without OpenGL/libGL
This allows YOLO and other packages to import cv2 without failing on headless systems
"""

import numpy as np
from PIL import Image
import io

# OpenCV constants for color transformations
COLOR_RGB2BGR = 4
COLOR_BGR2RGB = 5
COLOR_GRAY2BGR = 6
IMREAD_COLOR = 1
IMREAD_GRAYSCALE = 0

__version__ = "4.10.0-headless-stub"

def cvtColor(src, code):
    """Convert between color spaces using PIL/numpy"""
    if not isinstance(src, np.ndarray):
        src = np.array(src)
    
    if code == COLOR_RGB2BGR or code == COLOR_BGR2RGB:
        # RGB <-> BGR: just swap R and B channels
        if len(src.shape) == 3 and src.shape[2] == 3:
            return src[..., ::-1].copy()
    elif code == COLOR_GRAY2BGR:
        # Grayscale to BGR: repeat grayscale across 3 channels
        if len(src.shape) == 2:
            return np.stack([src, src, src], axis=-1)
    
    return src

def imread(filename, flags=IMREAD_COLOR):
    """Read image file using PIL"""
    try:
        img = Image.open(filename)
        if flags == IMREAD_GRAYSCALE:
            img = img.convert('L')
            return np.array(img)
        else:  # IMREAD_COLOR
            img = img.convert('RGB')
            # OpenCV uses BGR, PIL uses RGB - return as-is and convert in code
            return np.array(img)
    except Exception as e:
        return None

def imwrite(filename, img):
    """Write image file using PIL"""
    try:
        if isinstance(img, np.ndarray):
            # Assume BGR format (OpenCV standard)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img[..., ::-1]  # Convert BGR to RGB
                img_pil = Image.fromarray(img, 'RGB')
            else:
                img_pil = Image.fromarray(img)
        else:
            img_pil = Image.fromarray(img)
        img_pil.save(filename)
        return True
    except Exception as e:
        return False

class Mat:
    """Minimal Mat class for API compatibility"""
    def __init__(self, data):
        self.data = data

# Stub any other OpenCV functions that might be called
def resize(src, dsize, interpolation=None):
    """Resize image"""
    if isinstance(src, np.ndarray):
        img = Image.fromarray(src)
        img_resized = img.resize(dsize[::-1], Image.Resampling.LANCZOS)
        return np.array(img_resized)
    return src

def putText(img, text, org, fontFace, fontScale, color, thickness=1):
    """Add text to image (stub - returns img unchanged)"""
    return img

def rectangle(img, pt1, pt2, color, thickness=1):
    """Draw rectangle (stub - returns img unchanged)"""
    return img

def circle(img, center, radius, color, thickness=1):
    """Draw circle (stub - returns img unchanged)"""
    return img

def line(img, pt1, pt2, color, thickness=1):
    """Draw line (stub - returns img unchanged)"""
    return img

# Default empty values for compatibility
CASCADE_CLASSIFIER_POOL = None
