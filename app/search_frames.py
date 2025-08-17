from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .config import AppConfig
from .utils.frame_indexer import iter_keyframes
from .object_utils import derive_object_json_path, load_objects, pick_vehicle_detections
from .color_heuristics import ColorThresholds, detect_orange_and_red
from .metadata_utils import load_metadata_for_video
from .visualize import draw_boxes, save_thumbnail
from .map_utils import load_map_for_video


@dataclass
class MatchResult:
	rel_path: str
	abs_path: str
	video_id: str
	frame_name: str
	frame_index: Optional[int]
	time_sec: Optional[float]
	fps: Optional[float]
	orange_ratio: float
	red_ratio: float
	vehicle_boxes: List[Tuple[int, int, int, int]]


def parse_args() -> dict:
	p = argparse.ArgumentParser(description="Search frames that contain orange car and red carpet")
	p.add_argument("--keyframes_dir", required=True)
	p.add_argument("--objects_dir", required=True)
	p.add_argument("--metadata_dir", required=False)
	p.add_argument("--map_keyframes_csv", required=False)
	p.add_argument("--clip_features_npy", required=False)
	p.add_argument("--output_dir", required=False, default="/workspace/output")
	p.add_argument("--num_results", type=int, default=50)
	p.add_argument("--vehicle_min_score", type=float, default=0.3)
	p.add_argument("--min_orange_ratio", type=float, default=0.02)
	p.add_argument("--min_red_ratio", type=float, default=0.02)
	p.add_argument("--max_workers", type=int, default=max(1, os.cpu_count() or 1))
	p.add_argument("--enable_clip_text_filter", action="store_true")
	return vars(p.parse_args())


def load_image(path: str) -> Optional[np.ndarray]:
	img = cv2.imread(path, cv2.IMREAD_COLOR)
	return img


def main() -> None:
	args = parse_args()
	cfg = AppConfig.from_args(args)
	os.makedirs(cfg.output_dir, exist_ok=True)

	thresholds = ColorThresholds(
		min_orange_ratio=float(args.get("min_orange_ratio", 0.02)),
		min_red_ratio=float(args.get("min_red_ratio", 0.02)),
	)

	results: List[MatchResult] = []
	# cache per video
	video_to_map: Dict[str, dict] = {}
	video_to_meta: Dict[str, dict] = {}
	for rel, abs_path in tqdm(iter_keyframes(cfg.keyframes_dir), desc="Scanning keyframes", unit="frame"):
		video_id = rel.split(os.sep)[0]
		frame_name = os.path.basename(rel)
		obj_json_path = derive_object_json_path(cfg.objects_dir, rel)
		objects = load_objects(obj_json_path)
		vehicles = pick_vehicle_detections(objects, min_score=float(args["vehicle_min_score"]))
		if not vehicles:
			continue

		img = load_image(abs_path)
		if img is None:
			continue
		h, w = img.shape[:2]
		vehicle_boxes_abs: List[Tuple[int, int, int, int]] = []
		orange_ok = False
		for v in vehicles:
			x1, y1, x2, y2 = v.to_abs_xyxy(w, h)
			vehicle_boxes_abs.append((x1, y1, x2, y2))
			orange_ratio, _ = detect_orange_and_red(img, thresholds, roi_xyxy=(x1, y1, x2, y2))
			if orange_ratio >= thresholds.min_orange_ratio:
				orange_ok = True
				break
		if not orange_ok:
			continue

		# check red carpet in the whole image (or lower region heuristic)
		_, red_ratio_full = detect_orange_and_red(img, thresholds, roi_xyxy=None)
		red_ok = red_ratio_full >= thresholds.min_red_ratio
		if not red_ok:
			# optional heuristic: check lower half as "carpet area"
			lower_y = int(h * 0.5)
			_, red_ratio_lower = detect_orange_and_red(img, thresholds, roi_xyxy=(0, lower_y, w, h))
			red_ok = red_ratio_lower >= thresholds.min_red_ratio
			red_ratio = red_ratio_lower
		else:
			red_ratio = red_ratio_full
		if not red_ok:
			continue

		# try to derive frame index/time/fps
		frame_index = None
		time_sec = None
		fps = None
		try:
			stem, _ = os.path.splitext(frame_name)
			frame_num = int(stem)
		except Exception:
			frame_num = None
		if frame_num is not None and args.get("map_keyframes_csv"):
			if video_id not in video_to_map:
				video_to_map[video_id] = load_map_for_video(args["map_keyframes_csv"], video_id)
			entry = video_to_map.get(video_id, {}).get(frame_num)
			if entry is not None:
				frame_index = int(entry.frame_index)
				time_sec = float(entry.second)
				fps = float(entry.fps)
		else:
			frame_index = frame_num

		results.append(MatchResult(
			rel_path=rel,
			abs_path=abs_path,
			video_id=video_id,
			frame_name=frame_name,
			frame_index=frame_index,
			time_sec=time_sec,
			fps=fps,
			orange_ratio=thresholds.min_orange_ratio,
			red_ratio=red_ratio,
			vehicle_boxes=vehicle_boxes_abs,
		))

	# sort by combination (red_ratio desc)
	results.sort(key=lambda r: (r.red_ratio), reverse=True)

	# output
	print("Top matches:")
	for i, r in enumerate(results[: int(args["num_results"])], start=1):
		time_txt = f" time={r.time_sec:.2f}s" if r.time_sec is not None else ""
		fps_txt = f" fps={r.fps:.1f}" if r.fps is not None else ""
		print(f"{i:03d}. video={r.video_id} frame={r.frame_name} frame_index={r.frame_index}{time_txt}{fps_txt} red_ratio={r.red_ratio:.3f} path={r.rel_path}")

		# save thumbnail
		img = load_image(r.abs_path)
		if img is None:
			continue
		img_boxed = draw_boxes(img, r.vehicle_boxes)
		thumb_path = os.path.join(cfg.output_dir, f"{r.video_id}_{r.frame_name}")
		save_thumbnail(thumb_path, img_boxed)

	print(f"Saved {min(len(results), int(args['num_results']))} thumbnails to {cfg.output_dir}")


if __name__ == "__main__":
	main()