import os
import re
from typing import Iterator, List, Tuple


def _natural_key(s: str):
	return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def iter_keyframes(keyframes_root: str) -> Iterator[Tuple[str, str]]:
	"""Yield (relative_path, absolute_path) in a stable, natural-sorted order.

	Assumes structure: <keyframes_root>/<video_id>/<frame_file> (e.g., 0000.jpg)
	"""
	for video_id in sorted(os.listdir(keyframes_root), key=_natural_key):
		video_dir = os.path.join(keyframes_root, video_id)
		if not os.path.isdir(video_dir):
			continue
		frame_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]
		frame_files.sort(key=_natural_key)
		for fname in frame_files:
			rel = os.path.join(video_id, fname)
			abs_path = os.path.join(video_dir, fname)
			yield rel, abs_path


def build_index(keyframes_root: str) -> List[str]:
	"""Return list of relative paths ordered consistently with feature index."""
	return [rel for rel, _ in iter_keyframes(keyframes_root)]