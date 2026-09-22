#!/usr/bin/env python3
"""
Preprocess images with YOLOv11 segmentation + optional OpenCV realtime display.

- Mirror source tree into destination
- Copy CSV files unchanged
- For JPEGs: detect instances, pick largest mask, black out background, crop to mask bbox, save
- If --show: display (original+mask overlay | cropped result) in real time with OpenCV

Controls (when --show):
- q : quit immediately
- space : pause/resume
"""

from pathlib import Path
import argparse
import shutil
import sys
import warnings

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError as e:
    print("Ultralytics not installed. Install with: pip install ultralytics", file=sys.stderr)
    raise

JPEG_EXTS = {".jpg", ".jpeg"}

def load_model(model_path: str, device: str = "") -> YOLO:
    model = YOLO(model_path)
    return model.to(device) if hasattr(model, "to") and device else model

def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def to_bgr(np_rgb: np.ndarray) -> np.ndarray:
    # RGB -> BGR for OpenCV
    return np_rgb[..., ::-1]

def overlay_mask_rgb(image_rgb: np.ndarray, mask_bool: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a solid color (green) on mask area for visualization (in RGB)."""
    overlay = image_rgb.copy()
    color = np.array([0, 255, 0], dtype=np.uint8)  # green in RGB (only for visualization)
    overlay[mask_bool] = (alpha * color + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)
    return overlay

def make_panel(left_rgb: np.ndarray, right_rgb: np.ndarray, max_height: int = 900) -> np.ndarray:
    """Stack left|right RGB panels; resize to fit max_height if needed. Return BGR for OpenCV display."""
    # Match heights
    lh, lw = left_rgb.shape[:2]
    rh, rw = right_rgb.shape[:2]
    if lh != rh:
        # scale right to left height
        scale = lh / rh
        right_rgb = cv2.resize(right_rgb[..., ::-1][..., ::-1], (int(rw * scale), lh), interpolation=cv2.INTER_AREA) if cv2 else right_rgb
        # Above trick avoids importing cv2 check when not installed; but if not installed, we won't call this anyway.
        # Safer approach below without cv2:
        if not cv2:
            from PIL import Image as _PILImage
            right_rgb = np.array(_PILImage.fromarray(right_rgb).resize((int(rw * scale), lh), _PILImage.BILINEAR))
    panel = np.concatenate([left_rgb, right_rgb], axis=1)

    # Resize for screen friendliness
    h, w = panel.shape[:2]
    if h > max_height:
        scale = max_height / h
        if cv2:
            panel = cv2.resize(panel, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)
        else:
            from PIL import Image as _PILImage
            panel = np.array(_PILImage.fromarray(panel).resize((int(w * scale), max_height), _PILImage.BILINEAR))
    return to_bgr(panel)

def process_image(model: YOLO,
                  src_img_path: Path,
                  dst_img_path: Path,
                  conf: float = 0.25,
                  imgsz: int = 1024,
                  show: bool = False,
                  delay_ms: int = 1,
                  window_name: str = "Preprocess",
                  max_height: int = 900):
    # Load with EXIF-aware orientation and convert to RGB
    with Image.open(src_img_path) as im_raw:
        im = ImageOps.exif_transpose(im_raw).convert("RGB")
    w, h = im.size

    results = model.predict(
        im,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
        task="segment"
    )

    if not results:
        warnings.warn(f"No results for image: {src_img_path}")
        if show and cv2 is not None:
            _show_text_only(str(src_img_path), "No results", delay_ms, window_name)
        return False

    res = results[0]
    if res.masks is None or res.masks.data is None or len(res.masks.data) == 0:
        warnings.warn(f"No instances detected: {src_img_path}")
        if show and cv2 is not None:
            left = np.array(im)
            right = np.zeros_like(left)
            left_annot = left.copy()
            # Put text
            _put_text_in_rgb(left_annot, "No instance", (20, 40))
            panel_bgr = make_panel(left_annot, right, max_height=max_height)
            _imshow_and_wait(panel_bgr, delay_ms, window_name)
        return False

    masks = res.masks.data
    masks_np = masks.float().cpu().numpy()  # [N,H,W], floats
    masks_bin = masks_np >= 0.5
    areas = masks_bin.reshape(masks_bin.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(areas))
    best_mask = masks_bin[best_idx]

    if areas[best_idx] == 0:
        warnings.warn(f"Zero-area mask (thresholding removed everything): {src_img_path}")
        if show and cv2 is not None:
            _show_text_only(str(src_img_path), "Zero-area mask", delay_ms, window_name)
        return False

    ys, xs = np.where(best_mask)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    im_np = np.array(im)  # RGB
    # Black background, keep foreground
    out_np = np.zeros_like(im_np, dtype=np.uint8)
    out_np[best_mask] = im_np[best_mask]

    # Crop to bbox
    left_i = int(max(0, xmin))
    top_i = int(max(0, ymin))
    right_i = int(min(w, xmax + 1))
    bottom_i = int(min(h, ymax + 1))
    cropped_np = out_np[top_i:bottom_i, left_i:right_i]

    if cropped_np.size == 0:
        warnings.warn(f"Empty crop region for: {src_img_path}")
        if show and cv2 is not None:
            _show_text_only(str(src_img_path), "Empty crop", delay_ms, window_name)
        return False

    # Save
    ensure_parent(dst_img_path)
    Image.fromarray(cropped_np).save(dst_img_path, quality=95)

    # Show (optional)
    if show and cv2 is not None:
        left_overlay = overlay_mask_rgb(im_np, best_mask, alpha=0.45)
        right_view = cropped_np
        panel_bgr = make_panel(left_overlay, right_view, max_height=max_height)

        # Add a small header with filename
        _put_header(panel_bgr, f"{src_img_path}  ->  {dst_img_path}")

        # show and handle key events
        if not _imshow_and_wait(panel_bgr, delay_ms, window_name):
            # user pressed q to quit early
            return "QUIT"

    return True

def _put_text_in_rgb(img_rgb: np.ndarray, text: str, org=(10, 30)):
    if cv2 is None:
        return
    img_bgr = to_bgr(img_rgb)
    cv2.putText(img_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr[..., ::-1]

def _put_header(img_bgr: np.ndarray, text: str):
    if cv2 is None:
        return
    h = 30
    pad = 8
    header = np.zeros((h, img_bgr.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, text, (pad, int(h*0.7)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    img_bgr[:h] = header

def _show_text_only(title: str, msg: str, delay_ms: int, window_name: str):
    if cv2 is None:
        return
    canvas = np.zeros((200, 900, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, msg, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    _imshow_and_wait(canvas, delay_ms, window_name)

def _imshow_and_wait(img_bgr: np.ndarray, delay_ms: int, window_name: str) -> bool:
    """
    Returns True to continue, False if user requested quit.
    Space toggles pause.
    """
    if cv2 is None:
        return True
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_bgr)
    paused = False
    while True:
        key = cv2.waitKey(0 if paused else max(1, delay_ms)) & 0xFF
        if key == ord('q') or key == 27:  # q or Esc
            cv2.destroyWindow(window_name)
            return False
        if key == ord(' '):  # space -> pause/resume
            paused = not paused
            if not paused:
                # draw current frame again so it "advances" once
                cv2.imshow(window_name, img_bgr)
            continue
        # any other key (or timeout) -> continue
        break
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess images with YOLOv11 segmentation (with optional OpenCV display).")
    parser.add_argument("--src", required=True, type=Path, help="Source directory")
    parser.add_argument("--dst", required=True, type=Path, help="Destination directory (will be created if missing)")
    parser.add_argument("--model", required=True, type=str, help="Path to YOLOv11 segmentation model (e.g., yolo11n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size (default: 1024)")
    parser.add_argument("--device", type=str, default="", help='Device (e.g., "cpu", "cuda:0"; default: auto)')
    parser.add_argument("--skip_nonjpeg", action="store_true", help="Skip non-JPEG, non-CSV files")
    parser.add_argument("--show", action="store_true", help="Show results in real time with OpenCV")
    parser.add_argument("--delay", type=int, default=1, help="Delay (ms) between frames when showing (default: 1)")
    parser.add_argument("--max_height", type=int, default=900, help="Max panel height when showing (default: 900)")
    args = parser.parse_args()

    if args.show and cv2 is None:
        print("OpenCV not installed. Install with: pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    src: Path = args.src
    dst: Path = args.dst

    if not src.exists() or not src.is_dir():
        print(f"Source directory does not exist or is not a directory: {src}", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model, device=args.device)

    all_files = [p for p in src.rglob("*") if p.is_file()]
    if not all_files:
        print("No files found in source directory.", file=sys.stderr)
        sys.exit(1)

    processed = 0
    copied_csv = 0
    skipped_no_instance = 0
    skipped_other = 0

    for p in tqdm(all_files, desc="Processing"):
        rel = p.relative_to(src)
        out_path = dst / rel

        ext = p.suffix.lower()
        if ext == ".csv":
            ensure_parent(out_path)
            shutil.copy2(p, out_path)
            copied_csv += 1
            continue

        if ext in JPEG_EXTS:
            status = process_image(
                model=model,
                src_img_path=p,
                dst_img_path=out_path,
                conf=args.conf,
                imgsz=args.imgsz,
                show=args.show,
                delay_ms=args.delay,
                window_name="YOLOv11 Preprocess",
                max_height=args.max_height,
            )
            if status == "QUIT":
                break
            if status:
                processed += 1
            else:
                skipped_no_instance += 1
            continue

        if args.skip_nonjpeg:
            skipped_other += 1
        else:
            ensure_parent(out_path)
            shutil.copy2(p, out_path)

    if args.show and cv2 is not None:
        cv2.destroyAllWindows()

    print(
        f"\nDone.\n"
        f"  JPEGs processed: {processed}\n"
        f"  CSVs copied:     {copied_csv}\n"
        f"  Skipped (no inst or empty): {skipped_no_instance}\n"
        f"  Skipped other types: {skipped_other}\n"
    )

if __name__ == "__main__":
    main()


"""
python preprocess.py --src /home/quentin/Desktop/earthworm_instance_seg/ecoscience_2025 --dst /home/quentin/Desktop/earthworm_instance_seg/ecoscience_2025-results --model runs/segment/train_2025-09-22/weights/best.pt
"""