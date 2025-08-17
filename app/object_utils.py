from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import ujson as json


@dataclass
class DetectedObject:
	label: str
	score: float
	bbox_xyxy_norm: Tuple[float, float, float, float]  # x1, y1, x2, y2 in [0,1]

	def to_abs_xyxy(self, width: int, height: int) -> Tuple[int, int, int, int]:
		x1 = max(0, min(width - 1, int(self.bbox_xyxy_norm[0] * width)))
		y1 = max(0, min(height - 1, int(self.bbox_xyxy_norm[1] * height)))
		x2 = max(0, min(width - 1, int(self.bbox_xyxy_norm[2] * width)))
		y2 = max(0, min(height - 1, int(self.bbox_xyxy_norm[3] * height)))
		if x2 <= x1:
			x2 = min(width - 1, x1 + 1)
		if y2 <= y1:
			y2 = min(height - 1, y1 + 1)
		return x1, y1, x2, y2


VEHICLE_LABELS = {
	"Car",
	"Vehicle",
	"Taxi",
	"Bus",
	"Truck",
	"Van",
	"Ambulance",
	"Limousine",
	"Sports car",
}


def _is_vehicle_label(name: str) -> bool:
	name_norm = name.strip().lower()
	for lab in VEHICLE_LABELS:
		if name_norm == lab.lower():
			return True
	return False


def _parse_tfhub_like(json_data: Dict[str, Any]) -> List[DetectedObject]:
	labels = json_data.get("detection_class_entities") or json_data.get("labels") or []
	scores = json_data.get("detection_scores") or json_data.get("scores") or []
	boxes = json_data.get("detection_boxes") or json_data.get("boxes") or []
	objects: List[DetectedObject] = []
	for i in range(min(len(labels), len(scores), len(boxes))):
		lab = labels[i]
		if isinstance(lab, bytes):
			lab = lab.decode("utf-8", errors="ignore")
		sc = float(scores[i])
		box = boxes[i]
		# TF Hub boxes are [ymin, xmin, ymax, xmax]
		if isinstance(box, dict):
			ymin = float(box.get("ymin", 0.0))
			xmin = float(box.get("xmin", 0.0))
			ymax = float(box.get("ymax", 1.0))
			xmax = float(box.get("xmax", 1.0))
		else:
			ymin, xmin, ymax, xmax = [float(v) for v in box]
		objects.append(DetectedObject(label=str(lab), score=sc, bbox_xyxy_norm=(xmin, ymin, xmax, ymax)))
	return objects


def _parse_generic(json_data: Dict[str, Any]) -> List[DetectedObject]:
	objs = json_data.get("objects") or json_data.get("detections") or []
	results: List[DetectedObject] = []
	for o in objs:
		lab = str(o.get("label") or o.get("class") or o.get("name") or "")
		score = float(o.get("score") or o.get("confidence") or 0.0)
		bbox = o.get("bbox") or o
		xmin = float(bbox.get("xmin") if bbox.get("xmin") is not None else bbox.get("x_min", 0.0))
		ymin = float(bbox.get("ymin") if bbox.get("ymin") is not None else bbox.get("y_min", 0.0))
		xmax = float(bbox.get("xmax") if bbox.get("xmax") is not None else bbox.get("x_max", 1.0))
		ymax = float(bbox.get("ymax") if bbox.get("ymax") is not None else bbox.get("y_max", 1.0))
		# May already be normalized; assume inputs are in [0,1]
		results.append(DetectedObject(label=lab, score=score, bbox_xyxy_norm=(xmin, ymin, xmax, ymax)))
	return results


def load_objects(json_path: str) -> List[DetectedObject]:
	if not os.path.exists(json_path):
		return []
	with open(json_path, "r") as f:
		data = json.load(f)
	if isinstance(data, dict):
		if "detection_boxes" in data or "detection_class_entities" in data:
			return _parse_tfhub_like(data)
		return _parse_generic(data)
	elif isinstance(data, list):
		# list of objects
		return _parse_generic({"objects": data})
	return []


def derive_object_json_path(objects_root: str, rel_keyframe_path: str) -> str:
	base, _ = os.path.splitext(rel_keyframe_path)
	return os.path.join(objects_root, f"{base}.json")


def pick_vehicle_detections(objects: List[DetectedObject], min_score: float = 0.3) -> List[DetectedObject]:
	return [o for o in objects if _is_vehicle_label(o.label) and o.score >= min_score]