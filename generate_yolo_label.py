from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing required library: {e}")
    print("Please install dependencies: pip install opencv-python mediapipe tqdm")
    sys.exit(1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = Path("hand_landmarker.task")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate YOLOv8 bounding box label files from hand gesture images using MediaPipe Hands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir",
        "-i",
        type=Path,
        default=Path("dataset"),
        help="Path to input dataset folder containing class subfolders (e.g. dataset/1, dataset/a).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("NSL"),
        help="Path to root output directory where parallel 'images/' and 'labels/' folders will be created.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Pixel margin padding around the bounding box.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold for MediaPipe Hands (0.0 to 1.0).",
    )
    parser.add_argument(
        "--max-hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect per image.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("skipped_images.log"),
        help="Path to log file for skipped images where no hand was detected.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MediaPipe hand_landmarker.task model file.",
    )
    return parser.parse_args()


def sort_key(label_name: str) -> tuple[int, int | str]:
    """Natural sorting key: numbers first numerically, then strings alphabetically."""
    if label_name.isdigit():
        return (0, int(label_name))
    return (1, label_name.lower())


def discover_classes(dataset_dir: Path) -> list[str]:
    """Scan dataset_dir for subdirectories representing class labels."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")

    classes = [
        d.name
        for d in dataset_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    classes.sort(key=sort_key)
    return classes


def get_unique_filename(label: str, img_path: Path, used_stems: set[str]) -> str:
    """Generate a unique base filename starting with the class label."""
    stem = img_path.stem
    if not stem.startswith(f"{label}_"):
        base_stem = f"{label}_{stem}"
    else:
        base_stem = stem

    unique_stem = base_stem
    counter = 1
    while unique_stem in used_stems:
        unique_stem = f"{base_stem}_{counter}"
        counter += 1

    used_stems.add(unique_stem)
    return unique_stem


def ensure_model_file(model_path: Path) -> Path:
    """Download hand_landmarker.task if not present locally."""
    if not model_path.exists():
        print(f"[INFO] Downloading MediaPipe Hand Landmarker model from {MODEL_URL}...")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print(f"[INFO] Model downloaded successfully to '{model_path}'.")
    return model_path


class HandDetectorWrapper:
    """
    Robust Hand Detector supporting both legacy MediaPipe Solutions
    and modern MediaPipe Tasks API across all Python/MediaPipe versions.
    """

    def __init__(self, max_hands: int = 2, min_confidence: float = 0.5, model_path: Path = DEFAULT_MODEL_PATH):
        self.max_hands = max_hands
        self.min_confidence = min_confidence
        self.use_legacy = False

        # Attempt legacy mp.solutions first if present
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            try:
                self.detector = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=max_hands,
                    min_detection_confidence=min_confidence,
                )
                self.use_legacy = True
                print("[INFO] Initialized MediaPipe Solutions Hands detector.")
                return
            except Exception:
                pass

        # Fallback to modern MediaPipe Tasks API
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        ensure_model_file(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("[INFO] Initialized MediaPipe Tasks HandLandmarker detector.")

    def detect(self, img_path: Path, cv2_bgr_img) -> list[list[tuple[float, float]]]:
        """Detect hands in an image, returning normalized landmark points [(x, y), ...] per hand."""
        hands_landmarks_list: list[list[tuple[float, float]]] = []

        if self.use_legacy:
            img_rgb = cv2.cvtColor(cv2_bgr_img, cv2.COLOR_BGR2RGB)
            results = self.detector.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    pts = [(lm.x, lm.y) for lm in hand_lms.landmark]
                    hands_landmarks_list.append(pts)
        else:
            mp_img = mp.Image.create_from_file(str(img_path))
            results = self.detector.detect(mp_img)
            if results.hand_landmarks:
                for hand_lms in results.hand_landmarks:
                    pts = [(lm.x, lm.y) for lm in hand_lms]
                    hands_landmarks_list.append(pts)

        return hands_landmarks_list

    def close(self) -> None:
        if hasattr(self.detector, "close"):
            self.detector.close()


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    padding: int = 20,
    min_confidence: float = 0.5,
    max_hands: int = 2,
    log_file: Path = Path("skipped_images.log"),
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    classes = discover_classes(dataset_dir)
    if not classes:
        print(f"[ERROR] No class subfolders found in '{dataset_dir}'. Exiting.")
        return

    label_to_id = {label: idx for idx, label in enumerate(classes)}

    images_out_dir = output_dir / "images"
    labels_out_dir = output_dir / "labels"
    images_out_dir.mkdir(parents=True, exist_ok=True)
    labels_out_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths from class subfolders
    all_image_tasks: list[tuple[Path, str, int]] = []
    for label in classes:
        class_dir = dataset_dir / label
        for img_file in class_dir.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                all_image_tasks.append((img_file, label, label_to_id[label]))

    if not all_image_tasks:
        print(f"[WARN] No valid images found in subfolders under '{dataset_dir}'. Exiting.")
        return

    print("=" * 65)
    print(" YOLOv8 MediaPipe Hand Gesture Bounding Box Generator ")
    print("=" * 65)
    print(f" Input Dataset Folder    : {dataset_dir.resolve()}")
    print(f" Output Root Folder      : {output_dir.resolve()}")
    print(f" Discovered Classes ({len(classes):02d}) : {', '.join(classes)}")
    print(f" Total Images Found      : {len(all_image_tasks)}")
    print(f" Bounding Box Padding    : {padding} px")
    print(f" Min Detection Confidence: {min_confidence}")
    print("-" * 65)

    # Initialize skipped log file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_file, "a", encoding="utf-8")
    log_handle.write(f"\n--- Processing Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    detector = HandDetectorWrapper(
        max_hands=max_hands,
        min_confidence=min_confidence,
        model_path=model_path,
    )

    success_count = 0
    skipped_count = 0
    used_stems: set[str] = set()

    for img_path, label_name, class_id in tqdm(all_image_tasks, desc="Processing Images"):
        img = cv2.imread(str(img_path))
        if img is None:
            skipped_count += 1
            log_handle.write(f"SKIPPED | {img_path} | Reason: Failed to load image file\n")
            continue

        h, w, _ = img.shape
        detected_hands = detector.detect(img_path, img)

        if not detected_hands:
            skipped_count += 1
            log_handle.write(f"SKIPPED | {img_path} | Reason: No hand detected\n")
            continue

        yolo_boxes: list[str] = []
        for hand_landmarks in detected_hands:
            x_coords = [pt[0] * w for pt in hand_landmarks]
            y_coords = [pt[1] * h for pt in hand_landmarks]

            # Determine pixel bounding box with padding clamped to image bounds
            x_min = max(0.0, min(x_coords) - padding)
            x_max = min(float(w), max(x_coords) + padding)
            y_min = max(0.0, min(y_coords) - padding)
            y_max = min(float(h), max(y_coords) + padding)

            box_w = x_max - x_min
            box_h = y_max - y_min

            if box_w <= 0 or box_h <= 0:
                continue

            # Convert to normalized YOLO format: <class_id> <x_center> <y_center> <width> <height>
            x_center = (x_min + x_max) / 2.0 / w
            y_center = (y_min + y_max) / 2.0 / h
            norm_w = box_w / w
            norm_h = box_h / h

            # Clamp normalized floats to [0.0, 1.0]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_boxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if not yolo_boxes:
            skipped_count += 1
            log_handle.write(f"SKIPPED | {img_path} | Reason: Bounding box coordinates invalid\n")
            continue

        # Generate unique filename prefixed with class label
        unique_stem = get_unique_filename(label_name, img_path, used_stems)
        dest_img_path = images_out_dir / f"{unique_stem}{img_path.suffix.lower()}"
        dest_txt_path = labels_out_dir / f"{unique_stem}.txt"

        # Copy image to NSL/images/
        shutil.copy2(img_path, dest_img_path)

        # Write corresponding YOLO label text file to NSL/labels/
        with open(dest_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_boxes) + "\n")

        success_count += 1

    detector.close()
    log_handle.close()

    # Generate dataset.yaml configuration file for YOLOv8 training
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# YOLOv8 Dataset Configuration\n")
        f.write(f"path: {output_dir.name}\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write("names:\n")
        for label_name, idx in label_to_id.items():
            f.write(f"  {idx}: \"{label_name}\"\n")

    print("\n" + "=" * 65)
    print(" Processing Finished! ")
    print("=" * 65)
    print(f" Successfully Processed & Saved : {success_count}")
    print(f" Skipped Images (Logged)         : {skipped_count}")
    print(f" Output Images Directory         : {images_out_dir.resolve()}")
    print(f" Output Labels Directory         : {labels_out_dir.resolve()}")
    print(f" YOLOv8 Dataset Config YAML      : {yaml_path.resolve()}")
    print(f" Skipped Images Log File         : {log_file.resolve()}")
    print("=" * 65)


def main() -> None:
    args = parse_args()
    process_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        padding=args.padding,
        min_confidence=args.min_confidence,
        max_hands=args.max_hands,
        log_file=args.log_file,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
