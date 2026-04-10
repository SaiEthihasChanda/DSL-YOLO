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

def warpAffine(src, M, dsize):
    """Warp image affine (stub - returns src unchanged)"""
    return src

def getRotationMatrix2D(center, angle, scale):
    """Get rotation matrix (stub)"""
    return np.eye(2, 3, dtype=np.float32)

def getMat(size, dtype):
    """Create matrix/array"""
    return np.zeros(size, dtype=dtype)

def inRange(src, lowerb, upperb):
    """Color range filtering (stub - returns binary same shape)"""
    if isinstance(src, np.ndarray):
        return np.ones(src.shape[:2], dtype=np.uint8) * 255
    return src

def findContours(src, mode, method):
    """Find contours (stub - returns empty list)"""
    return []

def drawContours(img, contours, contourIdx, color, thickness):
    """Draw contours (stub - returns img unchanged)"""
    return img

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

def polylines(img, pts, isClosed, color, thickness=1):
    """Draw polylines (stub - returns img unchanged)"""
    return img

def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1):
    """Draw ellipse (stub - returns img unchanged)"""
    return img

def arrowedLine(img, pt1, pt2, color, thickness=1, tipLength=0.1):
    """Draw arrow (stub - returns img unchanged)"""
    return img

def bilateralFilter(src, d, sigmaColor, sigmaSpace):
    """Bilateral filter (stub - returns src unchanged)"""
    return src

def GaussianBlur(src, ksize, sigmaX, sigmaY=None):
    """Gaussian blur (stub - returns src unchanged)"""
    return src

def medianBlur(src, ksize):
    """Median blur (stub - returns src unchanged)"""
    return src

def morphologyEx(src, op, kernel, iterations=1):
    """Morphology operation (stub - returns src unchanged)"""
    return src

def Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False):
    """Canny edge detection (stub - returns edges)"""
    if isinstance(image, np.ndarray):
        return np.zeros_like(image, dtype=np.uint8)
    return image

def split(m):
    """Split multi-channel image"""
    if isinstance(m, np.ndarray):
        if len(m.shape) == 3:
            return [m[:, :, i] for i in range(m.shape[2])]
        else:
            return [m]
    return [m]

def merge(mv):
    """Merge channels"""
    if isinstance(mv, (list, tuple)) and len(mv) > 0:
        return np.stack(mv, axis=-1)
    return mv[0] if mv else None

def floodFill(image, seedPoint, newVal):
    """Flood fill (stub - returns image unchanged)"""
    return image

def moments(contour):
    """Calculate moments (stub - returns empty dict)"""
    return {}

def matchTemplate(image, templ, method):
    """Template matching (stub - returns zeros array)"""
    if isinstance(image, np.ndarray) and isinstance(templ, np.ndarray):
        h, w = image.shape[:2]
        th, tw = templ.shape[:2]
        if w >= tw and h >= th:
            return np.zeros((h - th + 1, w - tw + 1), dtype=np.float32)
    return np.array([], dtype=np.float32).reshape(0, 0)

# OpenCV configuration functions (stubs)
def setNumThreads(nthreads):
    """Stub: Set number of threads for OpenCV operations"""
    pass

def getNumThreads():
    """Stub: Get number of threads"""
    return 1

def setUseOptimized(onoff):
    """Stub: Enable/disable optimizations"""
    pass

def useOptimized():
    """Stub: Check if optimizations enabled"""
    return True

# Default empty values for compatibility
CASCADE_CLASSIFIER_POOL = None
