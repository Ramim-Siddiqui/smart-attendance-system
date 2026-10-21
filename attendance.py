"""
Smart Attendance System - engine.

Recognises enrolled people on a CCTV / webcam stream, labels everyone else
Unknown-1, Unknown-2, ... and logs IN / OUT times to logs/attendance_YYYY-MM-DD.csv.

Normally you use the window:      python app.py
Command line (optional):
    python attendance.py enroll           # re-learn faces from images/<NAME>/
    python attendance.py run              # run with the saved camera settings
    python attendance.py run --source 0   # webcam index, RTSP URL or a video file
    python attendance.py report [--date YYYY-MM-DD]
"""
import argparse
import base64
import csv
import glob
import json
import logging
import os
import re
import shutil
import socket
import sys
import threading
import time
import urllib.parse
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import onnxruntime as ort

# =============================================================================
# PATHS AND SETTINGS
# =============================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HOME = os.environ.get("SMART_ATTENDANCE_HOME", APP_DIR)  # where images/, data/, logs/ live
MODEL_DIR = os.path.join(APP_DIR, "models", "buffalo_l")
IMAGES_DIR = os.path.join(HOME, "images")
DATA_DIR = os.path.join(HOME, "data")
LOG_DIR = os.path.join(HOME, "logs")
CROPS_DIR = os.path.join(DATA_DIR, "enrolled_faces")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")
FACES_FILE = os.path.join(DATA_DIR, "known_faces.npz")
CACHE_FILE = os.path.join(DATA_DIR, "embedding_cache.npz")
OPEN_VISITS_FILE = os.path.join(DATA_DIR, "open_visits.json")
STRANGERS_DIR = os.path.join(DATA_DIR, "strangers")
for _d in (IMAGES_DIR, DATA_DIR, LOG_DIR):
    os.makedirs(_d, exist_ok=True)

DEFAULT_SETTINGS = {
    "camera_type": "webcam",      # webcam | cctv | url | file
    "webcam_index": 0,
    "brand": "Hikvision", "ip": "", "port": 554, "username": "admin", "password": "",
    "channel": 1, "stream": "main",  # main = full quality, sub = lighter / faster
    "custom_url": "", "video_file": "",
    "match_threshold": 0.42,      # similarity needed to recognise a known person
    "exit_after_sec": 8,          # not seen this long = left; OUT = last time seen
    "det_size": 640,              # 640 accurate, 480 / 320 faster
    "min_face_px": 40,            # ignore faces smaller than this
}
CONFIRM_FRAMES = 3     # frames before a new person is logged IN (filters false detections)
ID_VOTES = 2           # frames that must match the same known person to name them
SAME_FACE_SIM = 0.5    # two faces this similar are the same person
STALL_SEC = 10         # no new frame for this long = reconnect the camera
AUGMENT = True         # extra variations of each enrollment photo
PHOTO_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# rtsp address formats; {main} is 1 for the main stream, 2 for the sub stream; {sub0} is 0 / 1
CAMERA_BRANDS = {
    "Hikvision / HiLook": "rtsp://{auth}{ip}:{port}/Streaming/Channels/{channel}0{main}",
    "Dahua / CP Plus / Imou": "rtsp://{auth}{ip}:{port}/cam/realmonitor?channel={channel}&subtype={sub0}",
    "Ezviz": "rtsp://{auth}{ip}:{port}/H.264",
    "TP-Link Tapo / VIGI": "rtsp://{auth}{ip}:{port}/stream{main}",
    "Uniview": "rtsp://{auth}{ip}:{port}/unicast/c{channel}/s{sub0}/live",
    "XMeye / generic DVR": "rtsp://{ip}:{port}/user={user}&password={password}&channel={channel}&stream={sub0}.sdp",
}
CAMERA_BRANDS["Hikvision"] = CAMERA_BRANDS["Hikvision / HiLook"]  # older settings files

log = logging.getLogger("attendance")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fh = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_fh)
    if sys.stdout is not None:  # pythonw has no console
        _sh = logging.StreamHandler(sys.stdout)
        _sh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        log.addHandler(_sh)


def _protect(text, encrypt):
    """Encrypt the camera password for the current Windows user (DPAPI)."""
    if os.name != "nt":
        return base64.b64encode(text.encode()).decode() if encrypt else base64.b64decode(text).decode()
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    data = text.encode("utf-8") if encrypt else base64.b64decode(text)
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in, blob_out = Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), Blob()
    fn = ctypes.windll.crypt32.CryptProtectData if encrypt else ctypes.windll.crypt32.CryptUnprotectData
    if not fn(ctypes.byref(blob_in), None, None, None, None, 0, ctypes.byref(blob_out)):
        raise OSError("Windows could not encrypt/decrypt the password")
    result = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    ctypes.windll.kernel32.LocalFree(blob_out.pbData)
    return base64.b64encode(result).decode() if encrypt else result.decode("utf-8")


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                saved = json.load(f)
            s.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS and k != "password"})
            if saved.get("password_encrypted"):
                s["password"] = _protect(saved["password_encrypted"], encrypt=False)
        except Exception:
            log.exception("Could not read %s, using defaults", SETTINGS_FILE)
    return s


def save_settings(s):
    out = {k: s.get(k, v) for k, v in DEFAULT_SETTINGS.items() if k != "password"}
    if s.get("password"):
        out["password_encrypted"] = _protect(s["password"], encrypt=True)
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, SETTINGS_FILE)


def build_source(s):
    """Turn the camera settings into what OpenCV opens: an int (webcam) or a URL / file path."""
    kind = s.get("camera_type", "webcam")
    if kind == "webcam":
        return int(s.get("webcam_index", 0))
    if kind == "url":
        return s.get("custom_url", "").strip()
    if kind == "file":
        return s.get("video_file", "").strip()
    user, pw = s.get("username", ""), s.get("password", "")
    q = lambda v: urllib.parse.quote(str(v), safe="")
    auth = f"{q(user)}:{q(pw)}@" if user else ""
    template = CAMERA_BRANDS.get(s.get("brand"), CAMERA_BRANDS["Hikvision / HiLook"])
    sub = s.get("stream") == "sub"
    return template.format(auth=auth, ip=s.get("ip", "").strip(), port=int(s.get("port", 554)),
                           channel=int(s.get("channel", 1)), main=2 if sub else 1, sub0=1 if sub else 0,
                           user=q(user), password=q(pw))


def mask(source):
    """Hide the password when showing or logging a camera address."""
    s = str(source)
    s = re.sub(r"(rtsp://[^:/@]+:)[^@]*@", r"\1****@", s)
    return re.sub(r"password=[^&]*", "password=****", s)


# =============================================================================
# FILES WITH NON-ENGLISH NAMES (cv2.imread / imwrite fail on them on Windows)
# =============================================================================
def imread(path):
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


# =============================================================================
# FACE DETECTION (SCRFD) + RECOGNITION (ArcFace) - InsightFace buffalo_l via ONNX Runtime
# =============================================================================
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class FaceModel:
    def __init__(self):
        self.det = self._load("det_")
        self.rec = self._load("w600k_")
        self.anchors = {}

    @staticmethod
    def _load(prefix):
        files = glob.glob(os.path.join(MODEL_DIR, prefix + "*.onnx"))
        if not files:
            raise FileNotFoundError(f"Face model {prefix}*.onnx is missing in {MODEL_DIR}. See README, section Models.")
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        return ort.InferenceSession(files[0], opts, providers=["CPUExecutionProvider"])

    def detect(self, img, size=640, threshold=0.5):
        """Returns a list of (box[x1,y1,x2,y2], landmarks[5,2], score)."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        out = self.det.run(None, {self.det.get_inputs()[0].name: blob})
        boxes, kpss, scores = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            n = size // stride
            if n not in self.anchors:
                grid = np.stack(np.mgrid[:n, :n][::-1], axis=-1).reshape(-1, 2) * stride
                self.anchors[n] = np.repeat(grid, 2, axis=0).astype(np.float32)
            keep = np.where(out[i].ravel() >= threshold)[0]
            c, d = self.anchors[n][keep], out[i + 3][keep] * stride
            boxes.append(np.hstack([c - d[:, :2], c + d[:, 2:]]))
            kpss.append(out[i + 6][keep].reshape(-1, 5, 2) * stride + c[:, None])
            scores.append(out[i].ravel()[keep])
        boxes, kpss, scores = np.vstack(boxes) / scale, np.vstack(kpss) / scale, np.concatenate(scores)
        if len(scores) == 0:
            return []
        idx = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.tolist()],
                               scores.tolist(), threshold, 0.4)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        return [(boxes[i], kpss[i], float(scores[i])) for i in np.array(idx, dtype=int).ravel()]

    @staticmethod
    def align(img, kps):
        M, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
        if M is None:  # landmarks unusable (very blurred / odd angle): fall back to a plain square crop
            (x1, y1), (x2, y2) = kps.min(0), kps.max(0)
            cx, cy, r = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1, 10) * 1.2
            M = np.float32([[112 / (2 * r), 0, 56 - cx * 112 / (2 * r)], [0, 112 / (2 * r), 56 - cy * 112 / (2 * r)]])
        return cv2.warpAffine(img, M, (112, 112))

    def embed(self, crops):
        """112x112 aligned BGR crops -> L2-normalised 512-d embeddings."""
        if len(crops) == 0:
            return np.zeros((0, 512), np.float32)
        blob = cv2.dnn.blobFromImages(crops, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        e = self.rec.run(None, {self.rec.get_inputs()[0].name: blob})[0]
        return (e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)).astype(np.float32)


_model, _model_lock = None, threading.Lock()


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _model = FaceModel()
        return _model


def augment(face):
    """Variations that imitate CCTV conditions (lighting, blur, distance, night mode, mirror).
    They cannot invent a new head angle; add real photos for that."""
    small = lambda px: cv2.resize(cv2.resize(face, (px, px), interpolation=cv2.INTER_AREA), (112, 112))
    jpeg = cv2.imdecode(cv2.imencode(".jpg", face, [cv2.IMWRITE_JPEG_QUALITY, 15])[1], cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    noise = np.clip(face + np.random.default_rng(0).normal(0, 12, face.shape), 0, 255).astype(np.uint8)
    return [cv2.flip(face, 1), cv2.convertScaleAbs(face, alpha=0.55), cv2.convertScaleAbs(face, alpha=1.35, beta=25),
            cv2.GaussianBlur(face, (7, 7), 0), small(32), small(48), jpeg, gray, noise]


def largest_face(img, size=640):
    faces = get_model().detect(img, size, threshold=0.4) if img is not None else []
    return max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1])) if faces else None


# =============================================================================
# PEOPLE AND ENROLLMENT
# =============================================================================
def display_name(folder):
    return os.path.basename(folder.rstrip("\\/")).replace("_", " ").strip().title()


def person_folders():
    return sorted(d for d in glob.glob(os.path.join(IMAGES_DIR, "*")) if os.path.isdir(d))


def person_photos(folder):
    return sorted(f for f in glob.glob(os.path.join(folder, "*")) if f.lower().endswith(PHOTO_EXT))


def list_people():
    """[(display name, folder, number of photos, [aligned face crop paths])]"""
    out = []
    for folder in person_folders():
        crops = sorted(glob.glob(os.path.join(CROPS_DIR, os.path.basename(folder), "*.jpg")))
        out.append((display_name(folder), folder, len(person_photos(folder)), crops))
    return out


def find_person_folder(name):
    for folder in person_folders():
        if display_name(folder).lower() == name.strip().lower():
            return folder
    return None


def clean_name(name):
    name = re.sub(r"[^\w\- ]", "", name, flags=re.UNICODE).strip()
    if not name:
        raise ValueError("Please enter a name (letters, numbers, spaces).")
    if name.lower().startswith("unknown"):
        raise ValueError("'Unknown' is reserved for strangers. Please choose another name.")
    return name


def add_photos(name, paths):
    """Copy photos that contain a face into images/<NAME>/. Returns (added, [(file, reason)])."""
    name = clean_name(name)
    folder = find_person_folder(name) or os.path.join(IMAGES_DIR, name.upper().replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    added, rejected = 0, []
    for i, path in enumerate(paths):
        base = os.path.basename(path)
        if not path.lower().endswith(PHOTO_EXT):
            rejected.append((base, "not a supported image type"))
            continue
        img = imread(path)
        if img is None:
            rejected.append((base, "file could not be read"))
        elif largest_face(img) is None:
            rejected.append((base, "no face found in the photo"))
        else:
            dest = os.path.join(folder, f"{datetime.now():%Y%m%d-%H%M%S}_{i}{os.path.splitext(path)[1].lower()}")
            shutil.copy2(path, dest)
            added += 1
    if not os.listdir(folder):  # nothing usable: don't leave an empty person behind
        os.rmdir(folder)
    return added, rejected


def add_frame_photo(name, frame):
    """Save a camera frame as a photo of this person (helps recognition from the real CCTV angle)."""
    name = clean_name(name)
    if largest_face(frame) is None:
        return False
    folder = find_person_folder(name) or os.path.join(IMAGES_DIR, name.upper().replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    return imwrite(os.path.join(folder, f"camera_{datetime.now():%Y%m%d-%H%M%S}.jpg"), frame)


def delete_person(name):
    folder = find_person_folder(name)
    if folder is None:
        return False
    root = os.path.realpath(IMAGES_DIR)
    if os.path.commonpath([root, os.path.realpath(folder)]) != root or os.path.realpath(folder) == root:
        raise ValueError("Refusing to delete a folder outside images/")
    shutil.rmtree(folder)
    return True


def enroll(progress=print):
    """Learn every person in images/<NAME>/. Photos already processed are reused from a cache.
    Returns ({name: photos used}, [(name, file, reason)])."""
    model = get_model()
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            c = np.load(CACHE_FILE)
            cache = dict(zip(c["keys"].tolist(), c["embs"]))
        except Exception:
            log.warning("Embedding cache unreadable, rebuilding")
    new_cache, embs, names, summary, skipped = {}, [], [], {}, []
    os.makedirs(CROPS_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(CROPS_DIR, "*.jpg")):  # crops from the old single-folder layout
        os.remove(stale)
    folders = person_folders()
    for folder in folders:
        name, person = display_name(folder), []
        crop_dir = os.path.join(CROPS_DIR, os.path.basename(folder))
        os.makedirs(crop_dir, exist_ok=True)
        keep = set()
        for path in person_photos(folder):
            st = os.stat(path)
            key = f"{os.path.relpath(path, IMAGES_DIR)}|{st.st_size}|{int(st.st_mtime)}|{int(AUGMENT)}"
            crop_path = os.path.join(crop_dir, os.path.basename(path) + ".jpg")
            if key in cache and os.path.exists(crop_path):
                e = cache[key]
            else:
                img = imread(path)
                face = largest_face(img)
                if face is None:
                    skipped.append((name, os.path.basename(path), "no face found" if img is not None else "unreadable"))
                    continue
                crop = model.align(img, face[1])
                imwrite(crop_path, crop)
                e = model.embed([crop] + (augment(crop) if AUGMENT else []))
            new_cache[key], person = e, person + [e]
            keep.add(os.path.normcase(crop_path))
        for c in glob.glob(os.path.join(crop_dir, "*.jpg")):
            if os.path.normcase(c) not in keep:
                os.remove(c)
        if person:
            arr = np.vstack(person)
            embs.append(arr)
            names += [name] * len(arr)
            summary[name] = len(person)
        progress(f"{name}: {len(person)} usable photo(s)")
    existing = {os.path.basename(f) for f in folders}
    for d in glob.glob(os.path.join(CROPS_DIR, "*")):
        if os.path.isdir(d) and os.path.basename(d) not in existing:
            shutil.rmtree(d)
    np.savez(FACES_FILE, embeddings=np.vstack(embs) if embs else np.zeros((0, 512), np.float32), names=np.array(names, dtype=str))
    if new_cache:
        np.savez(CACHE_FILE, keys=np.array(list(new_cache), dtype=str), embs=np.stack(list(new_cache.values())))
    elif os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
    for who, f, why in skipped:
        progress(f"  skipped {who}/{f}: {why}")
    return summary, skipped


def load_known():
    if not os.path.exists(FACES_FILE):
        return np.zeros((0, 512), np.float32), []
    d = np.load(FACES_FILE)
    return d["embeddings"].astype(np.float32), [str(n) for n in d["names"]]


# =============================================================================
# ATTENDANCE LOG (one CSV per day; a visit stays in the file of the day it started)
# =============================================================================
_csv_lock = threading.Lock()


def write_event(day, visit, name, event, when):
    path = os.path.join(LOG_DIR, f"attendance_{day}.csv")
    with _csv_lock:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["visit", "name", "event", "time"])
            w.writerow([visit, name, event, f"{when:%Y-%m-%d %H:%M:%S}"])
    log.info("%-7s %s", event, name)


def read_visits(day):
    """[{name, in, out, stayed}] for one day, in order of arrival."""
    path = os.path.join(LOG_DIR, f"attendance_{day}.csv")
    visits = {}
    if os.path.exists(path):
        with _csv_lock, open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = visits.setdefault(row["visit"], {"name": row["name"], "in": "", "out": ""})
                v["name"] = row["name"]
                if row["event"] in ("IN", "OUT"):
                    v[row["event"].lower()] = row["time"]
    out = []
    for v in visits.values():
        stayed = ""
        if v["in"] and v["out"]:
            secs = int((datetime.fromisoformat(v["out"]) - datetime.fromisoformat(v["in"])).total_seconds())
            stayed = f"{secs // 3600}h {secs % 3600 // 60:02d}m {secs % 60:02d}s"
        out.append({"name": v["name"], "in": v["in"][11:], "out": v["out"][11:] if v["out"] else "",
                    "out_date": v["out"][:10], "stayed": stayed})
    return sorted(out, key=lambda v: v["in"])


def available_days():
    files = glob.glob(os.path.join(LOG_DIR, "attendance_*.csv"))
    return sorted((os.path.basename(f)[11:21] for f in files), reverse=True)


def recover_open_visits():
    """Visits left open by a crash or power cut get their OUT at the last time the person was seen."""
    if not os.path.exists(OPEN_VISITS_FILE):
        return
    try:
        with open(OPEN_VISITS_FILE, encoding="utf-8") as f:
            for visit, v in json.load(f).items():
                write_event(v["day"], visit, v["name"], "OUT", datetime.fromisoformat(v["last"]))
                log.info("Recovered OUT for %s after an unexpected stop", v["name"])
    except Exception:
        log.exception("Could not recover open visits")
    os.remove(OPEN_VISITS_FILE)


# =============================================================================
# TRACKER: turns face detections into visits
# =============================================================================
class Tracker:
    def __init__(self, settings, known, known_names):
        self.s = settings
        self.set_known(known, known_names)
        self.people = []
        self.events = deque(maxlen=200)
        self.event_count = 0  # total events so far (the window uses it to spot new ones)
        self.day = f"{datetime.now():%Y-%m-%d}"
        self.strangers = self._load_strangers()
        self._last_save = 0.0

    def set_known(self, known, names):
        self.known, self.known_names = known, names

    # --- strangers keep their Unknown number for the whole day, even across restarts
    def _strangers_file(self):
        return os.path.join(STRANGERS_DIR, f"{self.day}.npz")

    def _load_strangers(self):
        try:
            d = np.load(self._strangers_file())
            return list(zip([str(n) for n in d["names"]], d["embs"]))
        except Exception:
            return []

    def _save_strangers(self):
        os.makedirs(STRANGERS_DIR, exist_ok=True)
        names = [n for n, _ in self.strangers]
        embs = np.array([e for _, e in self.strangers]) if self.strangers else np.zeros((0, 512), np.float32)
        np.savez(self._strangers_file(), names=np.array(names, dtype=str), embs=embs)

    def _stranger_name(self, emb):
        for name, e in self.strangers:
            if float(e @ emb) >= SAME_FACE_SIM:
                return name, False
        numbers = [int(n.split("-")[1]) for n, _ in self.strangers if n.split("-")[-1].isdigit()]
        name = f"Unknown-{max(numbers, default=0) + 1}"
        self.strangers.append((name, emb.copy()))
        self._save_strangers()
        return name, True

    # --- main update, called once per processed frame
    def update(self, faces, embs, frame, now):
        if f"{now:%Y-%m-%d}" != self.day:
            self.day, self.strangers = f"{now:%Y-%m-%d}", []
        thr = float(self.s["match_threshold"])
        taken = set()
        for (box, _, _), emb in zip(faces, embs):
            sim, name = 0.0, None
            if len(self.known_names):
                sims = self.known @ emb
                sim, name = float(sims.max()), self.known_names[int(sims.argmax())]
            free = [p for p in self.people if id(p) not in taken]
            score = lambda p: float(p["emb"] @ emb) + overlap(p["box"], box)
            p = max(free, key=score, default=None)
            if p is None or (float(p["emb"] @ emb) < SAME_FACE_SIM and overlap(p["box"], box) < 0.3):
                p = {"first": now, "frames": 0, "votes": {}, "name": None, "emb": emb, "visit": None,
                     "day": f"{now:%Y-%m-%d}", "new_stranger": False}
                self.people.append(p)
            taken.add(id(p))
            p.update(box=box, last=now, sim=sim, frames=p["frames"] + 1)
            p["emb"] = (p["emb"] + emb) / max(np.linalg.norm(p["emb"] + emb), 1e-9)
            if name and sim >= thr:
                p["votes"][name] = p["votes"].get(name, 0) + 1
            self._decide(p, frame, now)
        self.tick(now)
        if time.time() - self._last_save > 5:  # backup of open visits for crash recovery
            self._save_open()

    def _decide(self, p, frame, now):
        leader = max(p["votes"], key=p["votes"].get, default=None)
        lead_votes = p["votes"].get(leader, 0)
        if p["visit"] is None and p["frames"] >= CONFIRM_FRAMES:
            if lead_votes >= ID_VOTES:
                p["name"] = leader
            else:
                p["name"], p["new_stranger"] = self._stranger_name(p["emb"])
            p["visit"] = f"{p['first']:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
            self._event(p, "IN", p["first"])
            self._snapshot(p, frame)
        elif p["visit"] and leader and leader != p["name"] and lead_votes >= ID_VOTES \
                and lead_votes >= 2 * p["votes"].get(p["name"], 0):
            # e.g. first seen looking down (Unknown), then looked up and was recognised
            if p["new_stranger"]:
                self.strangers = [(n, e) for n, e in self.strangers if n != p["name"]]
                self._save_strangers()
                p["new_stranger"] = False
            p["name"] = leader
            self._event(p, "RENAMED", now)

    def tick(self, now):
        """Log OUT for anyone not seen for exit_after_sec (OUT = last time seen)."""
        gone = [p for p in self.people if (now - p["last"]).total_seconds() >= float(self.s["exit_after_sec"])]
        for p in gone:
            self.people.remove(p)
            if p["visit"]:
                self._event(p, "OUT", p["last"])

    def close_all(self):
        for p in self.people:
            if p["visit"]:
                self._event(p, "OUT", p["last"])
        self.people = []
        self._save_open()

    def _event(self, p, event, when):
        write_event(p["day"], p["visit"], p["name"], event, when)
        self.events.append((when, event, p["name"]))
        self.event_count += 1
        self._save_open()

    def _save_open(self):
        self._last_save = time.time()
        data = {p["visit"]: {"name": p["name"], "day": p["day"], "last": p["last"].isoformat(timespec="seconds")}
                for p in self.people if p["visit"]}
        try:
            if data:
                tmp = OPEN_VISITS_FILE + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp, OPEN_VISITS_FILE)
            elif os.path.exists(OPEN_VISITS_FILE):
                os.remove(OPEN_VISITS_FILE)
        except OSError:
            log.exception("Could not save open visits")

    def _snapshot(self, p, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = p["box"].astype(int)
        pad = (x2 - x1) // 3
        crop = frame[max(y1 - pad, 0):min(y2 + pad, h), max(x1 - pad, 0):min(x2 + pad, w)]
        if crop.size:
            folder = os.path.join(LOG_DIR, "snapshots", p["day"])
            os.makedirs(folder, exist_ok=True)
            imwrite(os.path.join(folder, f"{p['visit']}_{p['name']}.jpg"), crop)


def overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


# =============================================================================
# CAMERA
# =============================================================================
def open_capture(source):
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened() and os.name == "nt":
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cap
    if os.path.isfile(source):
        return cv2.VideoCapture(source)
    if source.lower().startswith("rtsp"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG,
                            [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000])


class Camera(threading.Thread):
    """Reads a live stream in the background, keeps only the newest frame, reconnects on failure."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source, self.frame, self.count = source, None, 0
        self.running, self.connected = True, False

    def run(self):
        warned = False
        while self.running:
            cap = open_capture(self.source)
            while self.running and cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if not self.connected:
                    log.info("Camera connected: %s (%dx%d)", mask(self.source), frame.shape[1], frame.shape[0])
                self.frame, self.count, self.connected, warned = frame, self.count + 1, True, False
            cap.release()
            if self.connected:
                log.warning("Camera signal lost, reconnecting ...")
            elif not warned:
                log.warning("Cannot open camera %s, retrying every 3s ...", mask(self.source))
                warned = True
            self.connected = False
            for _ in range(30):
                if not self.running:
                    break
                time.sleep(0.1)

    def stop(self):
        self.running = False


def test_connection(settings, timeout=10):
    """Try the camera settings. Returns (ok, message, frame or None)."""
    source = build_source(settings)
    if settings.get("camera_type") == "cctv":
        if not settings.get("ip", "").strip():
            return False, "Please enter the camera's IP address.", None
        try:
            socket.create_connection((settings["ip"].strip(), int(settings["port"])), 3).close()
        except OSError:
            return False, (f"Cannot reach {settings['ip']}:{settings['port']}. Check the IP address and that this "
                           "PC is on the same network as the camera / recorder."), None
    if source in ("", None):
        return False, "No camera address entered.", None
    if isinstance(source, str) and settings.get("camera_type") == "file" and not os.path.isfile(source):
        return False, "Video file not found.", None
    result = {}

    def attempt():
        cap = open_capture(source)
        t0 = time.time()
        while cap.isOpened() and time.time() - t0 < timeout:
            ok, frame = cap.read()
            if ok and frame is not None:
                result["frame"] = frame
                break
        result["opened"] = cap.isOpened()
        cap.release()

    t = threading.Thread(target=attempt, daemon=True)
    t.start()
    t.join(timeout + 10)
    frame = result.get("frame")
    if frame is not None:
        return True, f"Connected: {frame.shape[1]}x{frame.shape[0]} video.", frame
    if settings.get("camera_type") == "cctv":
        return False, ("Reached the camera but could not open the video. Check the username, password, brand "
                       "and channel, and that RTSP is enabled in the camera settings."), None
    if settings.get("camera_type") == "webcam":
        return False, "Webcam not available (wrong number, or another program is using it).", None
    return False, "Could not open the video. Check the address.", None


SCAN_PORTS = {554: "RTSP video", 8554: "RTSP video", 8000: "Hikvision", 37777: "Dahua / CP Plus",
              34567: "XMeye DVR", 2020: "Tapo ONVIF"}


def scan_network():
    """Look for cameras / recorders on this PC's network (read-only: discovery + port check, no logins).
    Returns [{ip, ports, hint, brand}]."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect(("10.255.255.255", 1))  # picks the local network interface, sends nothing
        my_ip = s.getsockname()[0]
    found = {}
    probe = (f'<?xml version="1.0" encoding="UTF-8"?><e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
             f'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
             f'xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><e:Header><w:MessageID>uuid:{uuid.uuid4()}</w:MessageID>'
             f'<w:To e:mustUnderstand="true">urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To><w:Action e:mustUnderstand="true">'
             f'http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header><e:Body><d:Probe>'
             f'<d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>')
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2.5)
            s.sendto(probe.encode(), ("239.255.255.250", 3702))
            while True:
                data, addr = s.recvfrom(65535)
                names = re.findall(r"onvif://www\.onvif\.org/(?:name|hardware)/([^\s<]+)", data.decode(errors="ignore"))
                found.setdefault(addr[0], {"ports": set(), "onvif": ""})["onvif"] = urllib.parse.unquote(" ".join(names))
    except OSError:
        pass

    def check(ip):
        open_ports = set()
        for port in SCAN_PORTS:
            with socket.socket() as s:
                s.settimeout(0.35)
                if s.connect_ex((ip, port)) == 0:
                    open_ports.add(port)
        return ip, open_ports

    prefix = my_ip.rsplit(".", 1)[0]
    with ThreadPoolExecutor(64) as ex:
        for ip, ports in ex.map(check, [f"{prefix}.{i}" for i in range(1, 255) if f"{prefix}.{i}" != my_ip]):
            if ports:
                found.setdefault(ip, {"ports": set(), "onvif": ""})["ports"] |= ports
    result = []
    for ip, v in sorted(found.items(), key=lambda kv: tuple(int(x) for x in kv[0].split("."))):
        brand = ("Hikvision / HiLook" if 8000 in v["ports"] else "Dahua / CP Plus / Imou" if 37777 in v["ports"]
                 else "XMeye / generic DVR" if 34567 in v["ports"] else "TP-Link Tapo / VIGI" if 2020 in v["ports"] else "")
        hint = v["onvif"] or brand or ("camera / recorder" if v["ports"] & {554, 8554} else "")
        result.append({"ip": ip, "ports": sorted(v["ports"]), "hint": hint or "unknown device", "brand": brand,
                       "rtsp_port": 554 if 554 in v["ports"] else 8554 if 8554 in v["ports"] else 554})
    return result, my_ip


# =============================================================================
# ENGINE: camera + recognition + logging in a background thread
# =============================================================================
COLORS = {"known": (60, 200, 60), "unknown": (0, 140, 255), "pending": (180, 180, 180)}


class Engine(threading.Thread):
    def __init__(self, settings, source=None):
        super().__init__(daemon=True)
        self.s = dict(settings)
        self.source = build_source(self.s) if source is None else source
        self.running, self.connected, self.finished = True, False, False
        self.status, self.error, self.fps = "Starting ...", None, 0.0
        self.preview, self.raw, self.frame_id = None, None, 0
        self.tracker, self._reload, self._errors = None, False, 0

    def stop(self):
        self.running = False

    def reload_faces(self):
        self._reload = True

    def people_in_view(self):
        return [p for p in (self.tracker.people if self.tracker else []) if p["visit"]]

    def run(self):
        try:
            if self.source in ("", None):
                raise ValueError("No camera configured. Open the Camera tab.")
            recover_open_visits()
            self.status = "Loading face models ..."
            model = get_model()
            known, names = load_known()
            if not names:
                log.warning("Nobody enrolled yet - everyone will be logged as Unknown")
            self.tracker = Tracker(self.s, known, names)
        except Exception as e:
            log.exception("Engine could not start")
            self.status, self.error, self.running, self.finished = f"Error: {e}", str(e), False, True
            return
        log.info("Monitoring started: %s", mask(self.source))
        while self.running:
            try:
                self._loop(model)
            except Exception as e:
                log.exception("Engine error, restarting in 5 seconds")
                self.status = f"Error: {e} - restarting ..."
                time.sleep(5)
        self.tracker.close_all()
        self.connected, self.status, self.finished = False, "Stopped", True
        log.info("Monitoring stopped")

    def _loop(self, model):
        src = self.source
        is_file = isinstance(src, str) and os.path.isfile(src)
        if is_file:  # recorded video: every frame, video time instead of clock time
            cap, start, n = cv2.VideoCapture(src), datetime.now(), 0
            vfps = cap.get(cv2.CAP_PROP_FPS) or 25
        else:
            cam = Camera(src)
            cam.start()
            last_count, last_new = -1, time.time()
        t_fps, n_fps = time.time(), 0
        try:
            while self.running:
                if self._reload:
                    self._reload = False
                    self.tracker.set_known(*load_known())
                    log.info("Known faces reloaded")
                if is_file:
                    ok, frame = cap.read()
                    if not ok:
                        self.running = False
                        break
                    n += 1
                    now, self.connected, self.status = start + timedelta(seconds=n / vfps), True, "Running (video file)"
                else:
                    frame, count = cam.frame, cam.count
                    if frame is None or count == last_count:
                        if cam.connected and time.time() - last_new > STALL_SEC:
                            log.warning("Camera stopped sending video, reconnecting ...")
                            cam.stop()
                            cam = Camera(src)
                            cam.start()
                            last_new = time.time()
                        self.connected = cam.connected and time.time() - last_new < STALL_SEC
                        self.status = "Running" if self.connected else "No camera signal - reconnecting ..."
                        self.tracker.tick(datetime.now())
                        time.sleep(0.02)
                        continue
                    last_count, last_new, now = count, time.time(), datetime.now()
                    self.connected, self.status = True, "Running"
                self._process(model, frame, now)
                n_fps += 1
                if time.time() - t_fps >= 1:
                    self.fps, t_fps, n_fps = n_fps / (time.time() - t_fps), time.time(), 0
        finally:
            if is_file:
                cap.release()
            else:
                cam.stop()

    def _process(self, model, frame, now):
        try:
            size, min_px = int(self.s["det_size"]), int(self.s["min_face_px"])
            faces = [f for f in model.detect(frame, size) if min(f[0][2] - f[0][0], f[0][3] - f[0][1]) >= min_px]
            embs = model.embed([model.align(frame, f[1]) for f in faces])
            self.tracker.update(faces, embs, frame, now)
        except Exception:
            self._errors += 1
            if self._errors <= 5 or self._errors % 500 == 0:
                log.exception("Frame skipped because of an error (%d so far)", self._errors)
        self.raw = frame
        self.preview = draw(frame.copy(), self.tracker.people, now)
        self.frame_id += 1


def draw(frame, people, now):
    scale = max(frame.shape[1] / 1280, 0.6)
    for p in people:
        x1, y1, x2, y2 = p["box"].astype(int)
        if p["name"] is None:
            kind, label = "pending", "..."
        elif p["name"].startswith("Unknown"):
            kind, label = "unknown", p["name"]
        else:
            kind, label = "known", f"{p['name']} {p['sim']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[kind], max(2, int(2 * scale)))
        cv2.putText(frame, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, COLORS[kind], max(2, int(2 * scale)))
    cv2.putText(frame, f"{now:%Y-%m-%d %H:%M:%S}", (10, int(30 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale,
                (255, 255, 255), max(2, int(2 * scale)))
    return frame


# =============================================================================
# COMMAND LINE
# =============================================================================
def _cli():
    ap = argparse.ArgumentParser(description="Smart Attendance System (the window is: python app.py)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("enroll")
    r = sub.add_parser("run")
    r.add_argument("--source", default=None, help="webcam number, RTSP URL or video file")
    r.add_argument("--no-window", action="store_true")
    sub.add_parser("report").add_argument("--date", default=None)
    args = ap.parse_args()

    if args.cmd == "enroll":
        summary, _ = enroll()
        print(f"Enrolled: {summary or 'nobody'}")
    elif args.cmd == "report":
        day = args.date or f"{datetime.now():%Y-%m-%d}"
        print(f"\nAttendance {day}\n{'Name':<16} {'IN':<10} {'OUT':<10} Stayed")
        for v in read_visits(day):
            print(f"{v['name']:<16} {v['in']:<10} {v['out'] or '(inside)':<10} {v['stayed']}")
    else:
        source = args.source
        if source is not None and str(source).isdigit():
            source = int(source)
        engine = Engine(load_settings(), source)
        engine.start()
        last = -1
        try:
            while engine.is_alive():
                if not args.no_window and engine.frame_id != last and engine.preview is not None:
                    last = engine.frame_id
                    cv2.imshow("Smart Attendance (Q to quit)", engine.preview)
                if not args.no_window:
                    if cv2.waitKey(20) & 0xFF in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        engine.stop()
        engine.join(15)
        cv2.destroyAllWindows()
        if engine.error:
            sys.exit(engine.error)


if __name__ == "__main__":
    _cli()
