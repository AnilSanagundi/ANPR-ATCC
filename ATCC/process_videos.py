# process_videos.py
import os
import cv2
import math
import time
import glob
import pandas as pd
import numpy as np
from ultralytics import YOLO

# --------- USER CONFIG ----------
VIDEO_FOLDER = "C:\\Users\\anil sanagundi\\OneDrive\\Desktop\\ATCC\\Dataset"
OUTPUT_CSV_FOLDER = "outputs/csvs"
SUMMARY_FOLDER = "outputs/summaries"
MODEL_WEIGHTS = "yolov8s.pt"    # change if you have other weights
COUNT_LINE_Y = None             # if None, it will be set to 2/3 height of each video
MAX_DISAPPEAR_FRAMES = 10       # tracker param
DIST_THRESHOLD = 60             # px, for re-identifying same object
ALLOWED_CLASSES = None          # None => all COCO classes. Or set like ['car','truck','bus','motorbike','bicycle','truck','bus','car']
# --------------------------------

os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

model = YOLO(MODEL_WEIGHTS)

# a small COCO id -> name mapping (ultralytics results give class id)
# we'll get names from model.names
class_names = model.names

class CentroidTracker:
    def __init__(self, max_disappear=MAX_DISAPPEAR_FRAMES, dist_thresh=DIST_THRESHOLD):
        self.next_object_id = 0
        self.objects = {}  # object_id -> (cx, cy)
        self.disappeared = {}  # object_id -> frames disappeared
        self.dist_thresh = dist_thresh
        self.max_disappear = max_disappear

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # No detections
        if len(input_centroids) == 0:
            # mark disappeared
            to_remove = []
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappear:
                    to_remove.append(oid)
            for oid in to_remove:
                self.deregister(oid)
            return self.objects.copy()

        # If no existing objects, register all
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects.copy()

        # Build distance matrix between existing object centroids and new input
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = np.linalg.norm(np.array(object_centroids)[:, None, :] - np.array(input_centroids)[None, :, :], axis=2)

        # For each existing object, find the closest detection, greedily
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if D[r, c] > self.dist_thresh:
                continue
            oid = object_ids[r]
            self.objects[oid] = tuple(input_centroids[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        # anything not matched -> disappeared++
        unmatched_rows = set(range(0, D.shape[0])) - used_rows
        for r in unmatched_rows:
            oid = object_ids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappear:
                self.deregister(oid)

        # anything unmatched cols -> new registrations
        unmatched_cols = set(range(0, D.shape[1])) - used_cols
        for c in unmatched_cols:
            self.register(tuple(input_centroids[c]))

        return self.objects.copy()


def process_video(video_path):
    basename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count_line_y = COUNT_LINE_Y or int(height * 2 / 3)
    ct = CentroidTracker()
    frame_id = 0

    rows = []
    per_video_counts = {}
    counted_track_ids = set()

    start_time = time.time()
    print(f"Processing {basename} (fps={fps}, size={width}x{height})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        # run YOLOv8 detection (fast single inference per frame)
        results = model.predict(source=[frame], imgsz=640, conf=0.35, verbose=False)  # returns list
        # results[0] is the result for this frame
        r = results[0]

        input_centroids = []
        det_infos = []  # keep (cx, cy, cls_name, conf, bbox)

        # r.boxes has xxyy numpy? using ultralytics result API - adapt
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for box in boxes:
                # box.xyxy, box.conf, box.cls
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "__len__") or hasattr(box.xyxy, "cpu") else box.xyxy
                # sometimes ultralytics returns tensor-like; attempt safe extraction
                try:
                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                except:
                    continue
                conf = float(box.conf[0]) if hasattr(box.conf, "__len__") else float(box.conf)
                cls_id = int(box.cls[0]) if hasattr(box.cls, "__len__") else int(box.cls)
                cls_name = class_names.get(cls_id, str(cls_id))
                if ALLOWED_CLASSES and cls_name not in ALLOWED_CLASSES:
                    continue
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                input_centroids.append((cx, cy))
                det_infos.append({"cx": cx, "cy": cy, "class": cls_name, "conf": conf, "bbox": (x1, y1, x2, y2)})

        # update tracker
        tracked_objects = ct.update(input_centroids)  # dict id -> (cx, cy)

        # map tracked IDs to the detection info (approx)
        # We'll assign each detection to the closest tracked object
        for det in det_infos:
            # find nearest track id by distance
            best_id = None
            best_dist = 1e9
            for oid, centroid in tracked_objects.items():
                d = math.hypot(centroid[0] - det["cx"], centroid[1] - det["cy"])
                if d < best_dist:
                    best_dist = d
                    best_id = oid
            if best_id is None:
                continue
            # add row
            rows.append({
                "video": basename,
                "frame": frame_id,
                "time_s": frame_id / fps,
                "track_id": best_id,
                "class": det["class"],
                "conf": det["conf"],
                "cx": det["cx"],
                "cy": det["cy"]
            })

            # counting: when the object's centroid crosses the horizontal line from above to below
            # To detect crossing, we need previous y of this track; use a small local history by scanning rows
            # Get previous row for same track if exists
            prev_rows = [r for r in rows[:-1] if r["track_id"] == best_id]
            prev_y = prev_rows[-1]["cy"] if prev_rows else None
            if prev_y is not None:
                # if crossing from above to below
                if prev_y < count_line_y <= det["cy"]:
                    # ensure we count this track only once
                    if (basename, best_id) not in counted_track_ids:
                        counted_track_ids.add((basename, best_id))
                        per_video_counts.setdefault(det["class"], 0)
                        per_video_counts[det["class"]] += 1

        # progress print occasionally
        if frame_id % 200 == 0:
            print(f"  frame {frame_id}")

    cap.release()
    elapsed = time.time() - start_time
    print(f"Finished {basename} processing in {elapsed:.1f}s - total frames: {frame_id}")

    # write per-frame CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_CSV_FOLDER, f"{os.path.splitext(basename)[0]}.csv")
    df.to_csv(csv_path, index=False)

    # write summary
    summary = {"video": basename, "total_frames": frame_id, "fps": fps, "width": width, "height": height, "count_line_y": count_line_y}
    # add per class counts
    for k, v in per_video_counts.items():
        summary[f"count_{k}"] = v
    # total vehicles
    summary["total_vehicles"] = sum(per_video_counts.values()) if per_video_counts else 0

    summary_df = pd.DataFrame([summary])
    summary_csv_path = os.path.join(SUMMARY_FOLDER, f"{os.path.splitext(basename)[0]}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    return csv_path, summary_csv_path


def main():
    video_files = sorted([p for p in glob.glob(os.path.join(VIDEO_FOLDER, "*")) if p.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))])
    summary_rows = []
    if not video_files:
        print("No videos found in", VIDEO_FOLDER)
        return

    for v in video_files:
        csv_path, summary_path = process_video(v)
        # append to master summary
        df = pd.read_csv(summary_path)
        summary_rows.append(df)

    if summary_rows:
        master = pd.concat(summary_rows, ignore_index=True).fillna(0)
        master.to_csv(os.path.join(SUMMARY_FOLDER, "master_summary.csv"), index=False)
        print("Master summary written to outputs/summaries/master_summary.csv")


if __name__ == "__main__":
    main()
