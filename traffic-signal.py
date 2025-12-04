"""
traffic-signal.py (GUI edition)
- Adds a small Tkinter GUI allowing:
    * Select Video (open file explorer)
    * Open Camera (system webcam)
    * Save Output (choose file to save annotated output)
    * Stop processing
- Uses OpenCV DNN YOLOv3 + simple centroid tracker (same logic as fixed script)
- Background processing via threading so GUI remains responsive.

Usage:
    python traffic-signal.py

Requirements:
    pip install opencv-python numpy requests imutils
"""

import os
import cv2
import numpy as np
import requests
import time
import imutils
import threading
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import OrderedDict

# -------- Configuration (same as before) --------
YOLO_CFG = "yolov3.cfg"
YOLO_WEIGHTS = "yolov3.weights"
COCO_NAMES = "coco.names"

YOLO_CFG_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
YOLO_WEIGHTS_URL = "https://pjreddie.com/media/files/yolov3.weights"
COCO_NAMES_URL = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
DETECTION_SKIP_FRAMES = 2  # run detection every N frames

VEHICLE_CLASSES = set(['car', 'bus', 'truck', 'motorbike', 'motorcycle'])

# -------- Utilities: download & yolo loading --------
def download_file(url, dest_path, chunk_size=1024*1024):
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(dest_path + '.tmp', 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                f.flush()
        os.replace(dest_path + '.tmp', dest_path)
        return True
    except Exception as e:
        try:
            if os.path.exists(dest_path + '.tmp'):
                os.remove(dest_path + '.tmp')
        except Exception:
            pass
        print("Download failed for {}: {}".format(url, e))
        return False

def ensure_yolo_files():
    missing = []
    for fname, url in ((YOLO_CFG, YOLO_CFG_URL), (YOLO_WEIGHTS, YOLO_WEIGHTS_URL), (COCO_NAMES, COCO_NAMES_URL)):
        if not os.path.exists(fname):
            missing.append((fname, url))
    if not missing:
        return True
    # Try automatic but keep UI-friendly messages
    for fname, url in missing:
        print("Attempting to download {} ...".format(fname))
        ok = download_file(url, fname)
        if not ok:
            return False
    # sanity for weights
    if os.path.exists(YOLO_WEIGHTS):
        if os.path.getsize(YOLO_WEIGHTS) < 10_000_000:
            return False
    return True

def load_yolo_net():
    if not ensure_yolo_files():
        return None, None
    try:
        net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        with open(COCO_NAMES, 'r') as f:
            classes = [c.strip() for c in f.readlines() if c.strip()]
        return net, classes
    except Exception as e:
        print("Failed to load YOLO network:", e)
        return None, None

def get_outputs_names(net):
    layer_names = net.getLayerNames()
    try:
        return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except Exception:
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -------- Simple centroid tracker (same as before) --------
class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=50):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = (centroid, bbox)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cX = int(x + w / 2.0)
            cY = int(y + h / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], rects[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [v[0] for v in self.objects.values()]
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = (input_centroids[col], rects[col])
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], rects[col])

        return self.objects

# -------- Detection & processing (adapted to accept a stop_event) --------
def detect_objects_yolo(net, classes, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = max(0, int(center_x - bw / 2))
                y = max(0, int(center_y - bh / 2))
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))
                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    rects = []
    labels = []
    if len(indices) > 0:
        for i in indices.flatten():
            lbl = classes[class_ids[i]] if class_ids[i] < len(classes) else str(class_ids[i])
            rects.append(boxes[i])
            labels.append(lbl)
    return rects, labels

def process_video_source_threaded(source, net, classes, output_path, stop_event, status_callback=None):
    """
    Runs in a background thread. Reads source (file path or camera index or url),
    annotates frames, shows display and writes output_path if provided.
    stop_event is a threading.Event used to stop processing.
    status_callback(optional) is called with simple status strings.
    """
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            if status_callback:
                status_callback(f"Cannot open source: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = CentroidTracker(max_disappeared=40, max_distance=60)
        frame_no = 0
        total_count = 0

        if status_callback:
            status_callback("Processing...")

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1

            rects = []
            if net is not None and (frame_no % DETECTION_SKIP_FRAMES == 0):
                rects_all, labels = detect_objects_yolo(net, classes, frame)
                # filter only vehicles
                rects = [r for r, lbl in zip(rects_all, labels) if lbl in VEHICLE_CLASSES]
                tracker.update(rects)
            else:
                tracker.update([])

            current_ids = list(tracker.objects.keys())
            total_count = max(total_count, len(current_ids))

            annotated = frame.copy()
            for object_id, (centroid, bbox) in tracker.objects.items():
                x, y, w, h = bbox
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, "ID {}".format(object_id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.putText(annotated, "CurrentTracked: {} TotalObserved: {}".format(len(current_ids), total_count),
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if out is not None:
                out.write(annotated)

            # Display (non-blocking small wait)
            display = imutils.resize(annotated, width=900)
            cv2.imshow("Traffic Detection", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        if status_callback:
            status_callback(f"Stopped. Total observed approx: {total_count}")
    except Exception as e:
        if status_callback:
            status_callback(f"Error: {e}")
        print("Processing thread error:", e)

# -------- Tkinter GUI --------
class TrafficGUI:
    def __init__(self, root):
        self.root = root
        root.title("Traffic Signal - Select Video or Camera")
        root.geometry("450x220")

        # state
        self.net = None
        self.classes = None
        self.processing_thread = None
        self.stop_event = threading.Event()
        self.current_source = None
        self.output_path = None

        # UI elements
        self.select_btn = tk.Button(root, text="Select Video", width=20, command=self.select_video)
        self.select_btn.pack(pady=8)

        self.cam_btn = tk.Button(root, text="Open Camera", width=20, command=self.open_camera)
        self.cam_btn.pack(pady=8)

        self.save_btn = tk.Button(root, text="Save Output...", width=20, command=self.save_output)
        self.save_btn.pack(pady=8)

        self.stop_btn = tk.Button(root, text="Stop", width=20, command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(pady=8)

        self.status_label = tk.Label(root, text="Status: idle", anchor="w")
        self.status_label.pack(fill="x", padx=10, pady=10)

        self.load_models_btn = tk.Button(root, text="Load / Check Models", command=self.load_models)
        self.load_models_btn.pack(pady=4)

        # preload models in background so UI starts faster
        self.root.after(100, self.load_models_background)

    def load_models_background(self):
        t = threading.Thread(target=self.load_models, daemon=True)
        t.start()

    def load_models(self):
        self.set_status("Checking YOLO model files...")
        net, classes = load_yolo_net()
        if net is None or classes is None:
            self.set_status("YOLO files missing or failed to load. Click 'Load / Check Models' to retry or download manually.")
            # show instructions in popup
            txt = ("YOLO files are missing or corrupted.\n\n"
                   "Please download and place these files in the same folder as this script:\n"
                   "1) yolov3.cfg -> https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg\n"
                   "2) yolov3.weights -> https://pjreddie.com/media/files/yolov3.weights  (~200 MB)\n"
                   "3) coco.names -> https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names\n\n"
                   "After placing them, click 'Load / Check Models' again.")
            messagebox.showwarning("YOLO files missing", txt)
            self.net = None
            self.classes = None
            return
        self.net = net
        self.classes = classes
        self.set_status("Models loaded. Ready.")

    def set_status(self, text):
        self.status_label.config(text="Status: " + text)

    def select_video(self):
        path = filedialog.askopenfilename(title="Select video file", filetypes=(("MP4 files","*.mp4;*.mov;*.avi;*.mkv"),("All files","*.*")))
        if not path:
            return
        self.current_source = path
        self.start_processing(source=path)

    def open_camera(self):
        # default camera index 0
        self.current_source = 0
        self.start_processing(source=0)

    def save_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=(("MP4 files","*.mp4"),("All files","*.*")))
        if not path:
            return
        self.output_path = path
        self.set_status("Output will be saved to: " + path)

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_event.set()
            self.processing_thread.join(timeout=5)
            self.set_status("Stopping...")
        self.stop_btn.config(state=tk.DISABLED)
        self.select_btn.config(state=tk.NORMAL)
        self.cam_btn.config(state=tk.NORMAL)
        self.load_models_btn.config(state=tk.NORMAL)

    def start_processing(self, source):
        # prevent double-start
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showinfo("Already running", "Processing is already running. Please stop first.")
            return
        # ensure models loaded
        if self.net is None or self.classes is None:
            # Attempt load now
            self.set_status("Loading models...")
            self.load_models()
            if self.net is None:
                return

        self.stop_event.clear()
        self.stop_btn.config(state=tk.NORMAL)
        self.select_btn.config(state=tk.DISABLED)
        self.cam_btn.config(state=tk.DISABLED)
        self.load_models_btn.config(state=tk.DISABLED)
        self.set_status("Starting processing...")

        def status_cb(msg):
            # called from processing thread
            def upd():
                self.set_status(msg)
            try:
                self.root.after(1, upd)
            except Exception:
                pass

        self.processing_thread = threading.Thread(
            target=process_video_source_threaded,
            args=(source, self.net, self.classes, self.output_path, self.stop_event, status_cb),
            daemon=True
        )
        self.processing_thread.start()

# -------- Run GUI --------
def main():
    root = tk.Tk()
    app = TrafficGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_close(root, app))
    root.mainloop()

def on_close(root, app):
    # stop any running threads cleanly
    try:
        app.stop_event.set()
        if app.processing_thread and app.processing_thread.is_alive():
            app.processing_thread.join(timeout=2)
    except Exception:
        pass
    try:
        root.destroy()
    except Exception:
        pass

if __name__ == "__main__":
    main()
