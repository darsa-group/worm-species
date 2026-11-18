"""Recorded YOLO11 segmentation recipe; requires human training annotations."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="YOLO dataset YAML with independent annotations and the original partitions",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-weights", default="yolo11l-seg.pt")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional new run setting; historical script did not specify it",
    )
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("The annotated YOLO dataset YAML is required")
    output = args.output.resolve()
    if output.exists():
        parser.error("Preserve existing outputs and choose a fresh directory")
    if min(args.epochs, args.image_size, args.batch_size) < 1:
        parser.error("Training counts must be positive")
    # Use the original human annotations and split, rather than generated masks.
    from ultralytics import YOLO

    options = dict(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        device=args.device,
        project=str(output.parent),
        name=output.name,
        exist_ok=False,
    )
    if args.seed is not None:
        options["seed"] = args.seed
    # Ultralytics writes the fitted checkpoint and training log to this new folder.
    YOLO(args.initial_weights).train(**options)


if __name__ == "__main__":
    main()
