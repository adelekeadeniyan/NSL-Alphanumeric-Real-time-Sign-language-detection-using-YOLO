from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract image frames from videos inside NSL_Videos with unique label-based naming."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("NSL_Videos"),
        help="Folder containing source videos (default: NSL_Videos).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("NSL_Videos") / "frames",
        help="Folder where extracted frames will be saved.",
    )
    parser.add_argument(
        "--every-n-frames",
        type=int,
        default=1,
        help="Save one frame every N frames (default: 1).",
    )
    parser.add_argument(
        "--max-frames-per-video",
        type=int,
        default=0,
        help="Optional cap on saved frames per video. 0 means no cap.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality from 0 to 100 (default: 95).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Optional output width. Use with --height to resize.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Optional output height. Use with --width to resize.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Optional custom filename prefix. If not specified, uses label name (e.g. 'a').",
    )
    parser.add_argument(
        "--no-unique-id",
        action="store_true",
        help="Disable unique timestamp+UUID suffix in filenames.",
    )
    return parser.parse_args()


def find_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []

    return sorted(
        file_path
        for file_path in input_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS
    )


def maybe_resize(frame, width: int, height: int):
    if width > 0 and height > 0:
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def get_label_for_video(video_path: Path, input_root: Path) -> str:
    """Extract class/label name for a video (e.g. 'a', '1')."""
    try:
        relative = video_path.relative_to(input_root)
        if len(relative.parts) > 1:
            parent_name = relative.parts[0]
            # If parent folder is generic container (like OpenCamera), stem is the label
            if parent_name.lower() in {"opencamera", "videos", "nsl_videos", "input"}:
                return video_path.stem
            return parent_name
    except ValueError:
        pass
    return video_path.stem


def generate_unique_id() -> str:
    """Generate a unique timestamp + short UUID hex string."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{short_uuid}"


def extract_frames_from_video(
    video_path: Path,
    input_root: Path,
    output_root: Path,
    every_n_frames: int,
    max_frames_per_video: int,
    jpg_quality: int,
    width: int,
    height: int,
    custom_prefix: str = "",
    use_unique_id: bool = True,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    relative_parent = video_path.parent.relative_to(input_root)
    video_output_dir = output_root / relative_parent / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine label (e.g. 'a', '1') and unique run tag
    label = custom_prefix if custom_prefix else get_label_for_video(video_path, input_root)
    unique_tag = f"_{generate_unique_id()}" if use_unique_id else ""

    frame_index = 0
    saved_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % every_n_frames == 0:
            frame_to_save = maybe_resize(frame, width=width, height=height)
            # Format: <label>_<timestamp>_<uuid>_<count>.jpg (e.g., a_20260724_143940_a7f9b2_000000.jpg)
            frame_name = f"{label}{unique_tag}_{saved_count:06d}.jpg"
            out_path = video_output_dir / frame_name
            cv2.imwrite(
                str(out_path),
                frame_to_save,
                [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
            )
            saved_count += 1

            if max_frames_per_video > 0 and saved_count >= max_frames_per_video:
                break

        frame_index += 1

    capture.release()
    return saved_count


def main() -> None:
    args = parse_args()

    if args.every_n_frames < 1:
        raise ValueError("--every-n-frames must be >= 1")

    if args.max_frames_per_video < 0:
        raise ValueError("--max-frames-per-video must be >= 0")

    if not (0 <= args.jpg_quality <= 100):
        raise ValueError("--jpg-quality must be between 0 and 100")

    if (args.width > 0 and args.height == 0) or (args.height > 0 and args.width == 0):
        raise ValueError("Provide both --width and --height to resize frames")

    videos = find_videos(args.input_dir)
    if not videos:
        print(f"[INFO] No videos found in: {args.input_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for video_path in videos:
        saved = extract_frames_from_video(
            video_path=video_path,
            input_root=args.input_dir,
            output_root=args.output_dir,
            every_n_frames=args.every_n_frames,
            max_frames_per_video=args.max_frames_per_video,
            jpg_quality=args.jpg_quality,
            width=args.width,
            height=args.height,
            custom_prefix=args.prefix,
            use_unique_id=not args.no_unique_id,
        )
        total_saved += saved
        print(f"[OK] {video_path.name}: saved {saved} frames")

    print(f"[DONE] Processed {len(videos)} videos, saved {total_saved} frames to {args.output_dir}")


if __name__ == "__main__":
    main()

