"""Find the court in a broadcast frame, so models see players at a usable
size instead of a full 1280x720 frame shrunk to 224px.

Detection is by colour (BWF mats are green) rather than ShuttleSet's
per-match homography, because uploaded clips have no homography: training
and inference must crop the same way, or the model sees different inputs
in production than it trained on. The homography corners are used only to
check this detector (scripts in vision/ report the IoU).
"""
import cv2
import numpy as np

# HSV range for the green court mat (OpenCV hue is 0-179)
GREEN_LO = np.array([35, 40, 40])
GREEN_HI = np.array([90, 255, 255])
# Players jump and the shuttle goes high above the mat: extend the box up
# by this fraction of the court height, and a little sideways.
EXTEND_UP = 0.45
EXTEND_SIDE = 0.06


def detect_court(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0, y0, x1, y1) of the court plus headroom, or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    biggest = max(contours, key=cv2.contourArea)
    h_img, w_img = frame.shape[:2]
    if cv2.contourArea(biggest) < 0.05 * h_img * w_img:
        return None  # close-up or crowd shot: no court in view
    x, y, w, h = cv2.boundingRect(biggest)
    x0 = max(0, int(x - EXTEND_SIDE * w))
    x1 = min(w_img, int(x + w + EXTEND_SIDE * w))
    y0 = max(0, int(y - EXTEND_UP * h))
    return x0, y0, x1, y + h


def stable_court_box(frames: list[np.ndarray]) -> tuple[int, int, int, int] | None:
    """Median box over several frames -- the broadcast camera is static
    during play, and the median shrugs off frames where it isn't."""
    boxes = [b for b in (detect_court(f) for f in frames) if b is not None]
    if not boxes:
        return None
    return tuple(int(v) for v in np.median(np.array(boxes), axis=0))


def crop(frame: np.ndarray, box: tuple[int, int, int, int], size: int = 224) -> np.ndarray:
    x0, y0, x1, y1 = box
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
