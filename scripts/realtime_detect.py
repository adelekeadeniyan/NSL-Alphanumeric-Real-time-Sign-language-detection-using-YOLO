from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' package is not installed.")
    print("Please install required dependencies: pip install ultralytics opencv-python")
    sys.exit(1)


# Default search locations for best.pt weights
DEFAULT_WEIGHT_PATHS = [
    Path("runs") / "detect" / "train_yolo11n" / "weights" / "best.pt",
    Path("runs") / "detect" / "train" / "weights" / "best.pt",
    Path("best.pt"),
]

DEFAULT_DATA_YAML = Path("data.yaml")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def find_best_weights(specified_path: str | None = None) -> Path:
    """Find the best.pt weights file from command line argument or standard run directories."""
    if specified_path:
        p = Path(specified_path)
        if p.exists():
            return p.resolve()
        else:
            print(f"[WARNING] Specified weights file not found: {specified_path}")

    # Check hardcoded defaults
    for p in DEFAULT_WEIGHT_PATHS:
        if p.exists():
            return p.resolve()

    # Search recursively in runs/detect/ for any best.pt, sorting by latest modified time
    runs_dir = Path("runs") / "detect"
    if runs_dir.exists():
        found_weights = list(runs_dir.glob("**/weights/best.pt"))
        if found_weights:
            latest_weight = max(found_weights, key=lambda f: f.stat().st_mtime)
            return latest_weight.resolve()

    # Fallback to yolo11n.pt baseline if no trained best.pt found
    fallback = Path("yolo11n.pt")
    if fallback.exists():
        print("[WARNING] Could not locate trained 'best.pt' in runs/detect. Falling back to pre-trained 'yolo11n.pt'.")
        return fallback.resolve()

    raise FileNotFoundError(
        "Could not find any 'best.pt' model weights! "
        "Please specify `--weights path/to/best.pt` or place your trained weights in `runs/detect/train_yolo11n/weights/best.pt`."
    )


def generate_class_colors(num_classes: int = 100) -> list[tuple[int, int, int]]:
    """Generate bright distinct BGR colors for each class."""
    np.random.seed(42)
    colors = []
    for _ in range(num_classes):
        color = tuple(map(int, np.random.randint(50, 255, size=3)))
        colors.append(color)
    return colors


def safe_imshow(winname: str, mat: np.ndarray, display_enabled: bool = True) -> tuple[bool, bool]:
    """
    Safely call cv2.imshow catching potential headless environment cv2.error.
    Returns: (display_success, is_headless_error)
    """
    if not display_enabled:
        return False, False
    try:
        cv2.imshow(winname, mat)
        return True, False
    except (cv2.error, Exception):
        return False, True


def safe_destroy_windows() -> None:
    """Safely destroy OpenCV GUI windows if supported."""
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def draw_detections(
    frame: np.ndarray,
    results,
    class_names: dict[int, str],
    colors: list[tuple[int, int, int]],
    conf_thresh: float,
    show_labels: bool = True,
) -> tuple[np.ndarray, int, dict[str, int]]:
    """Draw bounding boxes and class labels on a frame."""
    annotated = frame.copy()
    detection_count = 0
    detected_classes_summary: dict[str, int] = {}

    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        detection_count = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = class_names.get(cls_id, f"Class {cls_id}")

            detected_classes_summary[cls_name] = detected_classes_summary.get(cls_name, 0) + 1

            color = colors[cls_id % len(colors)]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            if show_labels:
                label_str = f"{cls_name} {conf:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_y = max(y1 - 10, text_h + 10)

                cv2.rectangle(
                    annotated,
                    (x1, label_y - text_h - 4),
                    (x1 + text_w + 6, label_y + baseline),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated,
                    label_str,
                    (x1 + 3, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    return annotated, detection_count, detected_classes_summary


def draw_hud(
    frame: np.ndarray,
    fps: float,
    latency_ms: float,
    conf_thresh: float,
    detection_count: int,
) -> np.ndarray:
    """Draw top HUD info bar and bottom controls bar."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    # Top HUD background
    hud_height = 45
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Top HUD text
    hud_text = f"YOLO11 | FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | Conf: {conf_thresh:.2f} | Detections: {detection_count}"
    cv2.putText(
        annotated,
        hud_text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom Legend bar
    legend_y = h - 12
    legend_text = "[Q]: Exit | [SPACE]: Pause | [S]: Snapshot | [+/-]: Conf | [C]: Labels"
    cv2.putText(
        annotated,
        legend_text,
        (12, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time Object Detection with trained YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=str,
        default=None,
        help="Path to trained model weights (e.g., runs/detect/train_yolo11n/weights/best.pt).",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="Path to dataset.yaml file for class names.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Video/Image source: '0' or webcam index for live camera, or path to a video/image file.",
    )
    parser.add_argument(
        "--conf",
        "-c",
        type=float,
        default=0.25,
        help="Initial confidence threshold (0.01 - 1.00). Adjust live with + / - keys.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for overlapping bounding boxes.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (e.g. 640, 416, 320).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Target hardware device ('0' for CUDA GPU, 'cpu', etc.). Leave empty for auto.",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default="",
        help="Optional path to save output detection video (e.g., output_realtime.mp4).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable GUI OpenCV display window (useful for headless environments or automated runs).",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="YOLO Real-Time Object Detection",
        help="Title of OpenCV window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Resolve weights path
    try:
        weights_path = find_best_weights(args.weights)
    except FileNotFoundError as err:
        print(f"[ERROR] {err}")
        sys.exit(1)

    print("=" * 80)
    print(" 🚀 YOLO REAL-TIME OBJECT DETECTION ")
    print("=" * 80)
    print(f" Loaded Weights File : {weights_path}")
    print(f" Dataset Config YAML : {args.data.resolve() if args.data.exists() else 'Using model names'}")
    print(f" Input Source        : {args.source}")
    print(f" Conf Threshold      : {args.conf:.2f} (Adjust live with '+' and '-')")
    print(f" IoU Threshold       : {args.iou:.2f}")
    print(f" Image Size          : {args.imgsz}x{args.imgsz}")
    print("=" * 80)

    # 2. Load YOLO Model
    print(f"[INFO] Initializing YOLO model from '{weights_path.name}'...")
    model = YOLO(str(weights_path))

    class_names = model.names
    num_classes = len(class_names)
    colors = generate_class_colors(max(num_classes, 100))
    print(f"[INFO] Loaded {num_classes} classes from model architecture.")

    snapshot_dir = Path("snapshots")
    snapshot_dir.mkdir(exist_ok=True)

    # Track GUI display availability
    display_enabled = not args.no_display

    # Check if input source is a static image file
    source_path = Path(args.source)
    if source_path.is_file() and source_path.suffix.lower() in IMAGE_EXTENSIONS:
        print(f"[INFO] Processing single image input: {source_path}")
        image = cv2.imread(str(source_path))
        if image is None:
            print(f"[ERROR] Unable to read image file: {source_path}")
            sys.exit(1)

        t0 = time.time()
        results = model.predict(
            source=image,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device if args.device else None,
            verbose=False,
        )
        infer_time = (time.time() - t0) * 1000.0

        annotated, count, classes_summary = draw_detections(
            image, results, class_names, colors, args.conf, show_labels=True
        )
        annotated = draw_hud(annotated, fps=0.0, latency_ms=infer_time, conf_thresh=args.conf, detection_count=count)

        out_img_path = snapshot_dir / f"detected_{source_path.name}"
        cv2.imwrite(str(out_img_path), annotated)
        print(f"[SUCCESS] Detections found: {count} objects.")
        for name, cnt in classes_summary.items():
            print(f"   - {name}: {cnt}")
        print(f"[SUCCESS] Saved output image to: {out_img_path.resolve()}")

        # Display window safely
        shown, is_headless = safe_imshow(args.window_name, annotated, display_enabled)
        if shown:
            print("[INFO] Press any key on the OpenCV window to close.")
            cv2.waitKey(0)
        elif is_headless:
            print("[NOTICE] GUI display unavailable in current shell. Output image saved to snapshots folder.")

        safe_destroy_windows()
        return

    # 3. Determine input source (webcam index or video file)
    source_val: str | int = args.source
    if source_val.isdigit():
        source_val = int(source_val)

    print(f"[INFO] Opening video stream source: {source_val}...")
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source_val}")
        print("Suggestions:")
        print("  1. Check if your webcam is connected and not in use by another application.")
        print("  2. Try another camera index like `--source 1`.")
        print("  3. Pass a video file path like `--source path/to/video.mp4`.")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input == 0 or np.isnan(fps_input):
        fps_input = 30.0

    print(f"[SUCCESS] Video stream opened! Resolution: {frame_width}x{frame_height} @ {fps_input:.1f} FPS")

    video_writer = None
    if args.save_video:
        out_path = Path(args.save_video)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps_input, (frame_width, frame_height))
        print(f"[INFO] Saving output video to: {out_path.resolve()}")

    conf_thresh = args.conf
    show_labels = True
    is_paused = False

    print("\n" + "-" * 80)
    print(" ⌨️  KEYBOARD CONTROLS:")
    print("   [Q] or [ESC] : Quit & exit detector")
    print("   [P] or [SPACE] : Pause / Resume video stream")
    print("   [S]          : Take snapshot frame & save to snapshots/")
    print("   [+] / [-]    : Increase / Decrease confidence threshold (step 0.05)")
    print("   [C]          : Toggle annotations / bounding box labels")
    print("-" * 80 + "\n")

    prev_frame_time = time.time()
    fps_history = []
    warned_headless = False

    try:
        while True:
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] End of video stream reached or stream stopped.")
                    break
            else:
                time.sleep(0.03)

            start_infer = time.time()

            results = model.predict(
                source=frame,
                conf=conf_thresh,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device if args.device else None,
                verbose=False,
            )

            infer_time_ms = (time.time() - start_infer) * 1000.0

            annotated_frame, count, _ = draw_detections(
                frame, results, class_names, colors, conf_thresh, show_labels=show_labels
            )

            curr_frame_time = time.time()
            frame_delta = curr_frame_time - prev_frame_time
            prev_frame_time = curr_frame_time
            fps = 1.0 / frame_delta if frame_delta > 0 else 30.0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            hud_frame = draw_hud(annotated_frame, avg_fps, infer_time_ms, conf_thresh, count)

            if video_writer is not None:
                video_writer.write(hud_frame)

            shown, is_headless = safe_imshow(args.window_name, hud_frame, display_enabled)
            if is_headless and not warned_headless:
                display_enabled = False
                warned_headless = True
                print("[NOTICE] GUI display window not supported in current environment. Detection stream running headlessly.")

            if display_enabled:
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # 'q' or ESC
                    print("\n[INFO] Exiting detection window...")
                    break
                elif key in (ord("p"), 32):  # 'p' or SPACE
                    is_paused = not is_paused
                    state_str = "PAUSED" if is_paused else "RESUMED"
                    print(f"[INFO] Video stream {state_str}")
                elif key in (ord("s"), ord("S")):  # Save snapshot
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    snap_path = snapshot_dir / f"snapshot_{timestamp}.png"
                    cv2.imwrite(str(snap_path), hud_frame)
                    print(f"[SUCCESS] Saved snapshot to: {snap_path.resolve()}")
                elif key in (ord("+"), ord("=")):  # Increase confidence
                    conf_thresh = min(1.0, conf_thresh + 0.05)
                    print(f"[INFO] Increased Conf Threshold to: {conf_thresh:.2f}")
                elif key in (ord("-"), ord("_")):  # Decrease confidence
                    conf_thresh = max(0.01, conf_thresh - 0.05)
                    print(f"[INFO] Decreased Conf Threshold to: {conf_thresh:.2f}")
                elif key in (ord("c"), ord("C")):  # Toggle class labels
                    show_labels = not show_labels
                    print(f"[INFO] Labels visibility set to: {show_labels}")

    except KeyboardInterrupt:
        print("\n[INFO] Detection interrupted by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        safe_destroy_windows()
        print("[INFO] Resources released and windows closed cleanly.")


if __name__ == "__main__":
    main()
