from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .object_utils import DetectedObject


def draw_boxes(image_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(0, 165, 255), thickness=2) -> np.ndarray:
	img = image_bgr.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
	return img


def save_thumbnail(path: str, image_bgr: np.ndarray, max_size: int = 640) -> None:
	h, w = image_bgr.shape[:2]
	scale = min(1.0, float(max_size) / float(max(h, w)))
	if scale < 1.0:
		image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
	cv2.imwrite(path, image_bgr)