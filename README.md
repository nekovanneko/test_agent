# Horse Tracking with YOLO

This repository contains a Python script, **`horse_tracker.py`**, which uses the
Ultralytics YOLOv8 object detection models to track horses in a video. The
script processes an input video frame‑by‑frame, applies object detection and
tracking, filters the detections to keep only the *horse* class, and draws
bounding boxes annotated with unique IDs for each tracked horse. The
annotated video is saved as output.

## Features

* **Multi‑object tracking** – assigns a persistent ID to each horse and follows
  it across frames.
* **Class filtering** – only the `horse` class is tracked; other detected
  objects are ignored.
* **Customizable model and tracker** – specify your own YOLOv8 weights and
  tracker configuration if desired.

## Requirements

To run the tracking script you will need the following Python packages:

* [`ultralytics`](https://github.com/ultralytics/ultralytics) – provides the
  YOLOv8 models and tracking API.
* [`opencv-python`](https://pypi.org/project/opencv-python/) – for video I/O
  and drawing operations.
* `numpy` – used for generating deterministic colours.

Install them via pip:

```
pip install ultralytics opencv-python numpy
```

> **Note:** the script does not download model weights automatically. You
> should download a YOLOv8 detection model (e.g. `yolov8n.pt`, `yolov8s.pt`,
> etc.) from the Ultralytics GitHub releases page or train your own model
> containing the `horse` class. Place the `.pt` file in a known location and
> reference it with the `--model` argument.

## Usage

Run `horse_tracker.py` from the command line:

```
python horse_tracker.py \
    --source path/to/input_video.mp4 \
    --output path/to/output_video.mp4 \
    --model yolov8n.pt \
    --tracker botsort.yaml \
    --persist
```

**Arguments:**

| Argument    | Description                                                                                                         |
|-------------|---------------------------------------------------------------------------------------------------------------------|
| `--source`  | **Required.** Path to the input video file.                                                                         |
| `--output`  | Path for the output annotated video file. Defaults to `output.mp4` in the current directory.                        |
| `--model`   | Path to the YOLO model `.pt` file. Defaults to `yolov8n.pt`.                                                         |
| `--tracker` | Optional tracker configuration YAML (e.g. `botsort.yaml` or `bytetrack.yaml`). If omitted, YOLO uses its default.   |
| `--persist` | Enable persistence of track IDs between frames. The script enables this by default.                                 |

For example, the following command will process `video.mp4` using the
pre‑trained `yolov8n.pt` model, assign IDs to each horse, draw bounding boxes
and ID labels, and write the result to `annotated.mp4`:

```
python horse_tracker.py --source video.mp4 --output annotated.mp4 --model yolov8n.pt --persist
```

## How it works

1. **Model loading.** The script loads a YOLOv8 detection model via
   Ultralytics. You can specify any model path supported by Ultralytics; by
   default it uses the small and fast `yolov8n.pt` model.
2. **Class identification.** It queries the model’s `names` list or mapping
   to find the index corresponding to the `horse` class. If the loaded model
   does not include horses, the script raises an error.
3. **Video processing.** The video is read frame‑by‑frame using OpenCV. For
   each frame, the script calls `model.track(frame, persist=True)` to
   perform detection and tracking. The `persist=True` flag ensures tracks
   remain consistent across frames【20873791326352†L300-L307】.
4. **Filtering and annotation.** The script filters detections to retain only
   those whose class ID matches the `horse` index. For each tracked horse, it
   draws a bounding box and writes a label of the form `ID <track_id>` on
   the frame.
5. **Output.** Annotated frames are written to the output video using
   OpenCV’s `VideoWriter`. At the end of processing, the script closes all
   resources and prints the path of the saved video.

## License

This project is provided for educational purposes and does not include
pre‑trained weights. Please ensure you have the right to use any models and
videos you process.
