import os
from dataclasses import dataclass


@dataclass
class AppConfig:
	keyframes_dir: str
	objects_dir: str
	metadata_dir: str | None = None
	map_keyframes_csv: str | None = None
	clip_features_npy: str | None = None
	output_dir: str = "/workspace/output"
	max_workers: int = max(1, os.cpu_count() or 1)
	enable_clip_text_filter: bool = False

	@staticmethod
	def from_args(args: dict) -> "AppConfig":
		return AppConfig(
			keyframes_dir=args["keyframes_dir"],
			objects_dir=args["objects_dir"],
			metadata_dir=args.get("metadata_dir"),
			map_keyframes_csv=args.get("map_keyframes_csv"),
			clip_features_npy=args.get("clip_features_npy"),
			output_dir=args.get("output_dir", "/workspace/output"),
			max_workers=int(args.get("max_workers", max(1, os.cpu_count() or 1))),
			enable_clip_text_filter=bool(args.get("enable_clip_text_filter", False)),
		)