from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class FrameMapEntry:
	frame: int
	second: float
	fps: float
	frame_index: int


def _infer_cols(df: pd.DataFrame):
	frame_col = next((c for c in df.columns if c.lower() in {"frame", "frame_number", "index"}), None)
	sec_col = next((c for c in df.columns if c.lower() in {"second", "time", "sec", "timestamp"}), None)
	fps_col = next((c for c in df.columns if c.lower() == "fps"), None)
	idx_col = next((c for c in df.columns if c.lower() in {"frame_index", "index_in_video", "global_index"}), None)
	vid_col = next((c for c in df.columns if c.lower() in {"video", "video_id", "vid", "source"}), None)
	return frame_col, sec_col, fps_col, idx_col, vid_col


def load_map_csv(csv_path: str) -> Dict[int, FrameMapEntry]:
	df = pd.read_csv(csv_path)
	frame_col, sec_col, fps_col, idx_col, _ = _infer_cols(df)
	if frame_col is None:
		raise ValueError("CSV must contain a frame column")
	res: Dict[int, FrameMapEntry] = {}
	for _, row in df.iterrows():
		frame_number = int(row[frame_col])
		second_val = float(row[sec_col]) if sec_col is not None and pd.notna(row[sec_col]) else 0.0
		fps_val = float(row[fps_col]) if fps_col is not None and pd.notna(row[fps_col]) else 0.0
		frame_index_val = int(row[idx_col]) if idx_col is not None and pd.notna(row[idx_col]) else frame_number
		res[frame_number] = FrameMapEntry(frame=frame_number, second=second_val, fps=fps_val, frame_index=frame_index_val)
	return res


def load_map_for_video(map_path: str, video_id: str) -> Dict[int, FrameMapEntry]:
	if not map_path:
		return {}
	if os.path.isdir(map_path):
		candidate = os.path.join(map_path, f"{video_id}.csv")
		if os.path.exists(candidate):
			return load_map_csv(candidate)
		return {}
	# map_path is file
	df = pd.read_csv(map_path)
	frame_col, sec_col, fps_col, idx_col, vid_col = _infer_cols(df)
	if vid_col is not None:
		df = df[df[vid_col].astype(str) == video_id]
	if frame_col is None:
		return {}
	res: Dict[int, FrameMapEntry] = {}
	for _, row in df.iterrows():
		frame_number = int(row[frame_col])
		second_val = float(row[sec_col]) if sec_col is not None and pd.notna(row[sec_col]) else 0.0
		fps_val = float(row[fps_col]) if fps_col is not None and pd.notna(row[fps_col]) else 0.0
		frame_index_val = int(row[idx_col]) if idx_col is not None and pd.notna(row[idx_col]) else frame_number
		res[frame_number] = FrameMapEntry(frame=frame_number, second=second_val, fps=fps_val, frame_index=frame_index_val)
	return res


def lookup_frame(map_data: Dict[int, FrameMapEntry], frame_number: int) -> Optional[FrameMapEntry]:
	return map_data.get(frame_number)