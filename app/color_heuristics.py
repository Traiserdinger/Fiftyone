from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class ColorThresholds:
	# HSV ranges (OpenCV: H in [0,179])
	orange_lower1: Tuple[int, int, int] = (5, 120, 80)
	orange_upper1: Tuple[int, int, int] = (20, 255, 255)

	red_lower1: Tuple[int, int, int] = (0, 120, 80)
	red_upper1: Tuple[int, int, int] = (10, 255, 255)
	red_lower2: Tuple[int, int, int] = (170, 120, 80)
	red_upper2: Tuple[int, int, int] = (179, 255, 255)

	min_orange_ratio: float = 0.02
	min_red_ratio: float = 0.02


def ratio_in_mask(mask: np.ndarray) -> float:
	total = mask.size
	if total <= 0:
		return 0.0
	return float(np.count_nonzero(mask)) / float(total)


def detect_orange_and_red(image_bgr: np.ndarray, thresholds: ColorThresholds, roi_xyxy: Tuple[int, int, int, int] | None = None) -> Tuple[float, float]:
	if roi_xyxy is not None:
		x1, y1, x2, y2 = roi_xyxy
		crop = image_bgr[y1:y2, x1:x2]
	else:
		crop = image_bgr
	if crop.size == 0:
		return 0.0, 0.0
	img_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

	# Orange
	mask_orange = cv2.inRange(img_hsv, np.array(thresholds.orange_lower1), np.array(thresholds.orange_upper1))
	orange_ratio = ratio_in_mask(mask_orange)

	# Red (two ranges)
	mask_red1 = cv2.inRange(img_hsv, np.array(thresholds.red_lower1), np.array(thresholds.red_upper1))
	mask_red2 = cv2.inRange(img_hsv, np.array(thresholds.red_lower2), np.array(thresholds.red_upper2))
	mask_red = cv2.bitwise_or(mask_red1, mask_red2)
	red_ratio = ratio_in_mask(mask_red)

	return orange_ratio, red_ratio