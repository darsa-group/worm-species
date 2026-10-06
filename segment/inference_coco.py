from __future__ import annotations

import shutil
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageOps

PAD_PX = 150         # padding before stage-1 inference
SQ_MARGIN = 300      # extra px added to the square's side
SMOOTH_MASK = True   # simple morphological + blur smoothing of stage-2 mask

# ---------- Helpers ----------
def min_area_rect_from_poly(poly):
    cnt = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
    return cv2.minAreaRect(cnt)  # ((cx,cy),(w,h),angle)

def square_from_rotated_rect(rect, extra_margin=0):
    (cx, cy), (w, h), angle = rect
    side = max(w, h) + 2 * extra_margin
    theta = np.deg2rad(angle).astype(np.float32)
    ux = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    uy = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    half = side / 2.0
    c = np.array([cx, cy], dtype=np.float32)
    tl = c - ux * half - uy * half
    tr = c + ux * half - uy * half
    br = c + ux * half + uy * half
    bl = c - ux * half + uy * half
    return np.stack([tl, tr, br, bl]).astype(np.float32), int(round(side))

def warp_quad_to_square(img_rgb, quad_pts, side):
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad_pts, dst)  # quad -> square
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    warped_bgr = cv2.warpPerspective(
        img_bgr, M, (side, side),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    return cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB), M

def overlay_translucent_bgr(base_bgr, mask_bool, color=(0, 0, 255), alpha=0.35):
    out = base_bgr.copy()
    overlay = base_bgr.copy()
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    out[mask_bool] = overlay[mask_bool]
    return out

def show_blocking(title, bgr, max_wh=1600):
    h, w = bgr.shape[:2]
    scale = min(1.0, max_wh / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(title, bgr)
    cv2.waitKey(1)

# ---------- New: COCO helpers ----------
def mask_to_coco_polygons(mask_u8):
    """
    Convert a single-component binary mask (0/255) to COCO polygons.
    Returns a list of polygons, each polygon is a flat list [x1,y1,x2,y2,...]
    """
    # Ensure binary
    mask = (mask_u8 > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        # Optional simplification to keep polygons reasonably small
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.0005 * peri, True)
        if len(approx) >= 3:
            poly = approx.reshape(-1, 2).astype(float).ravel().tolist()
            # COCO wants polygons with 6+ numbers (>=3 points)
            if len(poly) >= 6:
                polys.append(poly)
    return polys

def mask_to_bbox_and_area(mask_u8):
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0:
        return [0, 0, 0, 0], 0
    x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    area = int((mask_u8 > 0).sum())
    return [x, y, w, h], area

# ---------- Core with components filtering + visualization ----------
def process_file_two_stage_rotated_square_mask_vis_largest_cc(
    src_path: str, dst_path: str, model: YOLO,
    pad_px: int = PAD_PX, sq_margin: int = SQ_MARGIN, visualize: bool = False
):
    """
    Returns:
        ok (bool),
        info (dict | None) with keys:
            'basename', 'width', 'height', 'mask_on_orig' (uint8), 'category_id' (int)
        (mask_on_orig is 0/255 in ORIGINAL image coordinates)
    """
    # Load original
    try:
        with Image.open(src_path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Open failed {src_path}: {e}")
        return False, None
    orig_rgb = np.array(img)
    Ho, Wo = orig_rgb.shape[:2]

    # Stage 1: pad + detect
    padded = ImageOps.expand(img, border=pad_px, fill=(0, 0, 0))
    padded_rgb = np.array(padded)
    H1, W1 = padded_rgb.shape[:2]

    res1 = model(padded)[0]
    if res1.boxes is None or res1.boxes.xyxy is None or len(res1.boxes) == 0:
        print(f"[WARNING] Stage-1: no detections in {src_path}")
        return False, None

    xyxy = res1.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    idx = int(np.argmax(areas))  # largest bbox

    # Orientation from stage-1 mask polygon if available
    poly_for_rect = None
    if res1.masks is not None and len(res1.masks) > idx:
        polys_abs = getattr(res1.masks, "xy", None)
        polys_norm = getattr(res1.masks, "xyn", None)
        if polys_abs is not None and polys_abs[idx] is not None:
            p = polys_abs[idx]
            poly_for_rect = p if isinstance(p, np.ndarray) else p[0]
        elif polys_norm is not None and polys_norm[idx] is not None:
            p = polys_norm[idx]
            p = p if isinstance(p, np.ndarray) else p[0]
            poly_for_rect = np.column_stack([p[:, 0] * W1, p[:, 1] * H1])

    if poly_for_rect is not None and len(poly_for_rect) >= 3:
        rect = min_area_rect_from_poly(poly_for_rect.astype(np.float32))
    else:
        x1, y1, x2, y2 = xyxy[idx]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = (x2 - x1), (y2 - y1)
        rect = ((cx, cy), (w, h), 0.0)

    quad, side = square_from_rotated_rect(rect, extra_margin=sq_margin)

    # Warp rotated square -> square crop (Stage-2 input)
    crop_rgb, M = warp_quad_to_square(padded_rgb, quad, side)
    Hc = Wc = side

    # Stage 2: detect on crop; use MASK tensors
    res2 = model(Image.fromarray(crop_rgb))[0]
    if res2.masks is None or len(res2.masks) == 0:
        print(f"[WARNING] Stage-2: no masks in {src_path}")
        return False, None

    m = res2.masks.data
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m)  # (N,h,w)
    inst_areas = m.reshape(m.shape[0], -1).sum(axis=1)
    j = int(np.argmax(inst_areas))
    mask_small = m[j]  # (h,w) float 0..1

    # Resize to crop size if needed
    mh, mw = mask_small.shape
    if (mh, mw) != (Hc, Wc):
        mask_u8 = cv2.resize(mask_small, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask_u8 * 255).astype(np.uint8)
    else:
        mask_u8 = (mask_small * 255).astype(np.uint8)

    # Optional clean-up (close + light blur + threshold)
    if SMOOTH_MASK:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u8 = cv2.GaussianBlur(mask_u8, (0, 0), sigmaX=0.6)
        _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    # ---- keep only the largest connected component ----
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels > 2:
        comp_areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = 1 + int(np.argmax(comp_areas))
        mask_u8 = np.where(labels == keep_label, 255, 0).astype(np.uint8)
        removed = (num_labels - 2)
        print(f"[INFO] {src_path}: removed {removed} smaller component(s); kept label {keep_label}.")
    elif num_labels == 2:
        pass
    else:
        print(f"[WARNING] {src_path}: mask empty after CC filtering.")
        return False, None

    # Apply final mask to the crop (foreground original, background black)
    result_rgb = crop_rgb.copy()
    result_rgb[mask_u8 == 0] = 0

    # -------- Map mask back to ORIGINAL image coords (for COCO) --------
    dst_pts = np.array([[0, 0], [Wc-1, 0], [Wc-1, Hc-1], [0, Hc-1]], dtype=np.float32)
    invM = cv2.getPerspectiveTransform(dst_pts, quad.astype(np.float32))
    mask_on_padded = cv2.warpPerspective(
        mask_u8, invM, (W1, H1),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    if pad_px > 0:
        mask_on_orig = mask_on_padded[pad_px:H1-pad_px, pad_px:W1-pad_px]
    else:
        mask_on_orig = mask_on_padded
    if mask_on_orig.shape[:2] != (Ho, Wo):
        mask_on_orig = cv2.resize(mask_on_orig, (Wo, Ho), interpolation=cv2.INTER_NEAREST)

    # -------- Visualization on ORIGINAL image (final mask only) --------
    if visualize:
        orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        vis_bgr = overlay_translucent_bgr(orig_bgr, mask_on_orig.astype(bool),
                                          color=(255, 0, 0), alpha=0.35)
        show_blocking("Final (largest component) mask over ORIGINAL image", vis_bgr)

    # -------- Save result with very conservative JPEG --------
    try:
        os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
        saved_image = cv2.imwrite(
            dst_path,
            cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )
        saved_mask = cv2.imwrite(
            dst_path + ".png",
            mask_on_orig,
        )
        if not saved_image or not saved_mask:
            raise OSError(f"OpenCV could not write image or mask: {dst_path}")
        print(f"[INFO] Saved: {dst_path}")
    except Exception as e:
        print(f"[ERROR] Save failed {dst_path}: {e}")
        return False, None

    # Determine category id from stage-2 boxes if available; otherwise default to 1
    cat_id = 1
    try:
        if res2.boxes is not None and len(res2.boxes) > j and res2.boxes.cls is not None:
            cls_tensor = res2.boxes.cls
            cls_np = cls_tensor.cpu().numpy() if hasattr(cls_tensor, "cpu") else np.array(cls_tensor)
            cat_id = int(cls_np[j]) + 1  # +1 so COCO category ids start at 1
    except Exception:
        pass

    info = {
        "basename": os.path.basename(src_path),
        "width": Wo,
        "height": Ho,
        "mask_on_orig": mask_on_orig,  # uint8 0/255, EXIF-oriented source coordinates
        "mask_on_crop": mask_u8,  # coordinates aligned with the segmented RGB image
        "category_id": cat_id
    }
    return True, info


def is_jpeg(fname: str) -> bool:
    return os.path.splitext(fname)[1].lower() in (".jpg", ".jpeg")

def build_coco_categories(model):
    """
    Build COCO categories from model.names.
    COCO category ids must be >=1. We'll map:
      model index k (0-based) -> COCO id = k+1
    """
    cats = []
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        # names: {0:"cat", 1:"dog", ...}
        for k, v in sorted(names.items(), key=lambda kv: kv[0]):
            cats.append({"id": int(k) + 1, "name": str(v)})
    elif isinstance(names, (list, tuple)):
        for k, v in enumerate(names):
            cats.append({"id": k + 1, "name": str(v)})
    else:
        # fallback to single category
        cats = [{"id": 1, "name": "object"}]
    return cats

def mirror_and_process(src_root: str, dst_root: str, model):
    """
    Recreate subdirectories from src_root into dst_root.
    For JPEG files: apply process_file(), and accumulate COCO annotations.
    For non-JPEG files: copy verbatim.
    """
    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)

    # COCO accumulation
    coco = {
        "images": [],
        "annotations": [],
        "categories": build_coco_categories(model)
    }
    next_image_id = 1
    next_ann_id = 1

    # Map from basename -> image_id to avoid duplicates if same basename repeats
    used_basenames = {}

    for dirpath, _, filenames in os.walk(src_root):
        rel_dir = os.path.relpath(dirpath, src_root)
        dst_dir = os.path.join(dst_root, rel_dir if rel_dir != "." else "")
        os.makedirs(dst_dir, exist_ok=True)

        for fname in filenames:
            src_path = os.path.join(dirpath, fname)
            dst_path = os.path.join(dst_dir, fname)

            if is_jpeg(fname):
                ok, info = process_file_two_stage_rotated_square_mask_vis_largest_cc(src_path, dst_path, model)
                if not ok or info is None:
                    print(f"[WARNING] Skipped {src_path}")
                    continue

                basename = info["basename"]
                width = info["width"]
                height = info["height"]
                mask_on_orig = info["mask_on_orig"]
                category_id = info["category_id"]

                # Create or reuse image record by basename only
                if basename in used_basenames:
                    image_id = used_basenames[basename]
                else:
                    image_id = next_image_id
                    used_basenames[basename] = image_id
                    next_image_id += 1
                    coco["images"].append({
                        "id": image_id,
                        "file_name": basename,   # <-- basename only
                        "width": width,
                        "height": height
                    })

                # Build annotation from mask
                segs = mask_to_coco_polygons(mask_on_orig)
                if not segs:
                    print(f"[WARNING] {src_path}: polygonization produced empty segmentation; skipping annotation.")
                    continue

                bbox, area = mask_to_bbox_and_area(mask_on_orig)
                ann = {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "segmentation": segs,
                    "area": float(area),
                    "bbox": [float(x) for x in bbox],
                    "iscrowd": 0
                }
                coco["annotations"].append(ann)
                print(basename, ann)
                next_ann_id += 1

            elif os.path.splitext(src_path)[1] == ".csv":
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"[ERROR] Could not copy {src_path}: {e}")
            else:

                print(f"skipping {src_path}")

    # Write COCO json at the root of DST_ROOT
    ann_path = os.path.join(dst_root, "annotations.json")
    try:
        with open(ann_path, "w") as f:
            json.dump(coco, f)
        print(f"[INFO] Wrote COCO annotations: {ann_path}")
    except Exception as e:
        print(f"[ERROR] Could not write COCO JSON: {e}")
