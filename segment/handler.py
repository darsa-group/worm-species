# handler.py
# Hugging Face Inference Endpoints - Custom Handler for Ultralytics YOLOv11-seg
# Returns: {"instances":[{"label":str,"score":float,"polygon":[[x,y],...]},...],
#           "width": int, "height": int}

import io
import base64
from typing import Any, Dict, List, Union

from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


class EndpointHandler:
    def __init__(self, path: str = "."):
        """
        Called once on container startup.
        `path` points to the repo root mounted in the endpoint.
        """
        # Resolve weights using Hub API to get the raw binary (handles LFS/private).
        self.repo_id = "dashingzombie/yolov11-segmentation_earth-worm"
        self.filename = "best.pt"  # change if you prefer last.pt

        weights_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="model"
        )
        self.model = YOLO(weights_path)

        # If class names were not baked into the checkpoint, you can force them:
        if not getattr(self.model, "names", None):
            self.model.names = {0: "body_mask"}  # single-class fallback

    def _to_image(self, payload: Dict[str, Any]) -> Image.Image:
        """
        Accepts either:
          - {"inputs": {"image": <base64-string>}}  (Serverless-style)
          - {"inputs": <base64-string>}
          - {"image_bytes": <raw-bytes>}            (Toolkit raw)
        """
        if "image_bytes" in payload:
            return Image.open(io.BytesIO(payload["image_bytes"])).convert("RGB")

        inputs = payload.get("inputs", payload.get("instances", None))
        if isinstance(inputs, dict):
            img_b64 = inputs.get("image") or inputs.get("img") or inputs.get("data")
        else:
            img_b64 = inputs

        if isinstance(img_b64, str):
            # strip possible 'data:image/jpeg;base64,' prefix
            if "," in img_b64:
                img_b64 = img_b64.split(",", 1)[1]
            data = base64.b64decode(img_b64)
            return Image.open(io.BytesIO(data)).convert("RGB")

        raise ValueError("No image provided. Expected 'image_bytes' or base64 string under 'inputs'.")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs per request. `data` is the incoming JSON/body parsed by the Toolkit.
        Returns JSON-serializable dict.
        """
        image = self._to_image(data)
        W, H = image.size

        # confidence threshold can be overridden via params
        params = data.get("parameters", {}) or data.get("options", {})
        conf = float(params.get("conf", 0.25))

        results = self.model(image, conf=conf, verbose=False)[0]
        names = results.names

        instances: List[Dict[str, Any]] = []
        if results.masks is not None:
            # polygons per instance: results.masks.xy (list of Nx2 arrays)
            for i, poly in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                score = float(results.boxes.conf[i].item())
                polygon = [[float(x), float(y)] for x, y in poly]
                instances.append({
                    "label": names[cls_id],
                    "score": score,
                    "polygon": polygon
                })
        else:
            # Fallback to boxes if masks missing (rare for -seg)
            for i, b in enumerate(results.boxes.xyxy.tolist()):
                x1, y1, x2, y2 = [float(v) for v in b]
                cls_id = int(results.boxes.cls[i].item())
                score = float(results.boxes.conf[i].item())
                instances.append({
                    "label": names[cls_id],
                    "score": score,
                    "bbox_xyxy": [x1, y1, x2, y2]
                })

        return {"instances": instances, "width": W, "height": H}
