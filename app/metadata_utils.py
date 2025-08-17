from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def load_metadata_for_video(metadata_root: Optional[str], video_id: str) -> Dict[str, Any]:
	if not metadata_root:
		return {}
	path = os.path.join(metadata_root, f"{video_id}.json")
	if not os.path.exists(path):
		return {}
	try:
		with open(path, "r") as f:
			return json.load(f)
	except Exception:
		return {}