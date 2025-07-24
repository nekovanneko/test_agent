#!/usr/bin/env python3
"""
horse_tracker.py
=================

This script performs multi-object tracking for horses in a video using the
Ultralytics YOLO object detection and tracking API. It reads an input video,
detects objects in each frame, tracks them across frames, and writes an
annotated output video where each detected horse is surrounded by a bounding
box and labelled with a unique track ID. The IDs remain consistent across
frames, allowing you to follow each individual horse throughout the video.

The script relies on the Ultralytics YOLOv8 models for detection and the
internal tracker (ByteTrack/BoT-SORT) for track assignment. You can
optionally specify a custom model file or tracker configuration. If no model
is provided, the lightweight `yolov8n.pt` model (downloadable from the
Ultralytics repository) is used by default. Tracking is enabled via the
``persist`` flag, which ensures that track identities persist across frames.

Dependencies:
  * ultralytics (YOLOv8 library)  ``pip install ultralytics``
  * opencv‑python                ``pip install opencv-python``
  * numpy                        ``pip install numpy``

Example usage:

    python horse_tracker.py --source input.mp4 --output output.mp4 --model yolov8n.pt

The above command reads ``input.mp4``, tracks only horses, draws bounding
boxes with their IDs, and writes the annotated video to ``output.mp4``.

Note:
    This script does not download models automatically. Ensure that the
    specified model file exists locally. You can obtain pre-trained YOLOv8
    weights (e.g. ``yolov8n.pt``) from the Ultralytics repository.
"""

import argparse
import os
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics is required for this script. Install it with 'pip install ultralytics'"
    ) from exc


def get_horse_class_index(model) -> int:
    """Return the class index corresponding to 'horse' for the loaded model.

    The Ultralytics YOLO model stores class names in the ``names`` attribute.
    Depending on the version, this may be a list or a dict. This function
    searches through that structure to find the index of the 'horse' class.

    Args:
        model: An instance of ``YOLO`` from the Ultralytics library.

    Returns:
        The integer index of the 'horse' class in the model's class list.

    Raises:
        ValueError: If the 'horse' class cannot be found in the model's
            ``names`` attribute.
    """
    # ``model.names`` can be either a list or a dict mapping indices to names.
    names = getattr(model.model, 'names', None) or getattr(model, 'names', None)
    if isinstance(names, dict):
        for idx, name in names.items():
            if name == 'horse':
                return int(idx)
    else:
        # assume names is iterable (e.g. list)
        for idx, name in enumerate(names):
            if name == 'horse':
                return idx
    raise ValueError("Horse class not found in model's names. Ensure the model is trained on COCO or includes 'horse' class.")


def create_unique_color(track_id: int) -> tuple[int, int, int]:
    """Generate a deterministic color for a given track ID.

    By seeding NumPy's RNG with a value derived from the track ID, this
    function produces a consistent pseudo-random RGB color. This makes it
    easier to visually distinguish between different tracks across frames.

    Args:
        track_id: Unique identifier for the tracked object.

    Returns:
        A tuple of three integers representing an RGB color.
    """
    # Offset the seed so that ID 0 and ID 1 produce different colors
    np.random.seed(int(track_id) + 42)
    color = np.random.randint(0, 255, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def process_video(source: str, output: str, model_path: str, tracker_cfg: str | None, persist: bool) -> None:
    """Run horse tracking on the specified video and save the annotated output.

    Args:
        source: Path to the input video file.
        output: Path where the output video will be saved.
        model_path: Path to the YOLO model (.pt file).
        tracker_cfg: Optional path to a tracker configuration YAML file. If
            provided, YOLO will use the specified tracker instead of its default.
        persist: Whether to persist tracks across frames. When ``True``, track
            identities are maintained over time.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Determine the index for 'horse'
    horse_idx = get_horse_class_index(model)

    # Open the video file
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {source}")

    # Retrieve video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # fallback default if FPS cannot be determined

    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run tracking on the current frame
        results = model.track(
            frame,
            persist=persist,
            tracker=tracker_cfg if tracker_cfg else None,
        )

        # ``results`` is a list-like structure; we take the first element for single frame
        res = results[0]
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            # ``boxes.cls`` holds class IDs, ``boxes.id`` holds track IDs
            cls_ids = boxes.cls
            track_ids = boxes.id
            xyxy = boxes.xyxy
            # Convert to CPU/numpy for iteration
            if hasattr(cls_ids, 'cpu'):
                cls_ids = cls_ids.cpu().numpy()
                track_ids = track_ids.cpu().numpy()
                xyxy = xyxy.cpu().numpy()
            # Iterate through all detections
            for box, cls_id, t_id in zip(xyxy, cls_ids, track_ids):
                if int(cls_id) == horse_idx and t_id is not None:
                    x1, y1, x2, y2 = map(int, box)
                    color = create_unique_color(int(t_id))
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # Compose label text
                    label = f'ID {int(t_id)}'
                    # Compute text size for background rectangle
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    # Draw background rectangle behind text for readability
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_h - baseline - 4),
                        (x1 + text_w, y1),
                        color,
                        thickness=-1,
                    )
                    # Put the text label
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

        # Write the annotated frame to the output video
        out.write(frame)

    # Clean up resources
    cap.release()
    out.release()
    print(f'Finished processing. Annotated video saved to: {output}')

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the horse tracking script."""
    parser = argparse.ArgumentParser(
        description='Track horses in a video using YOLO and display unique IDs.'
    )
    parser.add_argument(
        '--source',
        required=True,
        help='Path to the input video file.',
    )
    parser.add_argument(
        '--output',
        default='output.mp4',
        help='Path for the output annotated video file (default: output.mp4).',
    )
    parser.add_argument(
        '--model',
        default='yolov8n.pt',
        help=(
            'Path to the YOLO model (.pt file). Defaults to yolov8n.pt. '
            'Download pre-trained models from https://github.com/ultralytics/yolov8'
        ),
    )
    parser.add_argument(
        '--tracker',
        default=None,
        help=(
            'Optional: path to a tracker configuration YAML (e.g. botsort.yaml). '
            'If omitted, the default tracker is used.'
        ),
    )
    parser.add_argument(
        '--persist',
        action='store_true',
        help='Persist tracks across frames (enabled by default).',
    )
    args = parser.parse_args()
    # If user does not explicitly disable persist, set it to True
    if not args.persist:
        args.persist = True
    return args


def main() -> None:
    args = parse_args()
    process_video(
        source=args.source,
        output=args.output,
        model_path=args.model,
        tracker_cfg=args.tracker,
        persist=args.persist,
    )


if __name__ == '__main__':
    main()
