# Smart Attendance System

Watches a CCTV camera, recognises the people you enroll, labels everyone else
Unknown-1, Unknown-2, ..., and records when each person comes IN and goes OUT.
Runs on a normal Windows PC (no GPU, no internet needed).

**Built with:** Python, OpenCV, ONNX Runtime, InsightFace models (SCRFD face detector and ArcFace
recogniser), Tkinter, RTSP and ONVIF for cameras, Windows DPAPI for the stored camera password.

Faces are found with SCRFD, aligned to five landmarks and turned into 512-number ArcFace embeddings.
A face is named when its embedding is close enough (cosine similarity) to a person enrolled from
photos. A tracker follows each face between frames, so one visit is logged once, with an IN time and
an OUT time.

## Start

Double-click **`Smart Attendance.bat`**, or run:

```
.venv\Scripts\python app.py
```

The window has four tabs:

| Tab | What you do there |
|---|---|
| **Live** | Start / stop monitoring, see the camera with names on faces, who is in view, recent IN/OUT events. "Add this face to a person" adds the current camera picture to a person, which improves recognition from the real CCTV angle. |
| **People** | Add a person (name + photos), add more photos, delete a person, see which faces were learned. Photos without a usable face are rejected with a reason. |
| **Camera** | Choose CCTV / webcam / stream address / video file, enter the camera details, **Search network** for cameras, **Test connection** (shows a picture), recognition settings, and "start when Windows starts". |
| **Attendance** | Visits per day (came in, went out, how long), export to Excel CSV, open the snapshot and log folders. |

## Connecting the CCTV

1. The PC must be on the **same network** as the camera or recorder (DVR/NVR).
2. Camera tab: choose "CCTV camera or recorder", press **Search network** and click a result,
   or type the IP address. Pick the brand, enter the camera's username and password,
   the channel (camera number on a recorder) and quality (main = full, sub = faster).
3. Press **Test connection**. If a picture appears, press **Save settings**.

The password is stored encrypted for your Windows user. If the brand is not listed,
choose "Other stream address" and paste the RTSP address from the camera's manual.

## Files

```
app.py                 the window
attendance.py          the engine (recognition, tracking, logging); also usable from the command line
Smart Attendance.bat   double-click to start
images/<NAME>/         photos per person (managed from the People tab)
models/buffalo_l/      face models: det_10g.onnx (finds faces), w600k_r50.onnx (recognises them)
data/                  settings.json, learned faces, today's Unknown list, open-visit backup
logs/                  attendance_YYYY-MM-DD.csv, snapshots/ of every visit, app.log
```

## How it decides

- A new face must be seen in 3 frames before it is logged (filters false detections).
- It is named when 2 frames match the same enrolled person; otherwise it gets an Unknown number.
  If someone first logged as Unknown is recognised later (for example they looked up), the visit is renamed.
- OUT = the last moment the person was seen, logged once they have been gone for the "mark as left" time.
- A stranger who returns the same day keeps the same Unknown number, also after a restart.
- A visit past midnight stays in the file of the day it started.
- After a crash or power cut, open visits are closed at the last time the person was seen.
- If the camera drops or freezes, it reconnects automatically.

Enrollment adds variations of each photo (mirror, dark, bright, blurred, far away, compressed,
night-mode grey, noisy). That helps with CCTV image quality. It cannot create new head angles,
so add real photos from different angles, ideally with "Add this face to a person" on the Live tab.

## Setup on a new PC

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Models: download <https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip>
and copy `det_10g.onnx` and `w600k_r50.onnx` into `models/buffalo_l/`.

## Command line (optional)

```
.venv\Scripts\python attendance.py enroll
.venv\Scripts\python attendance.py run [--source 0 | rtsp://... | video.mp4] [--no-window]
.venv\Scripts\python attendance.py report [--date 2026-10-16]
```

## Tuning

- Known person shown as Unknown: lower "match strictness" (e.g. 0.38) or add photos from the camera.
- Stranger shown as a known person: raise it (e.g. 0.50).
- Logged OUT while still there: raise "mark as left after".
- Slow: detector size 480 or 320, or use the camera's sub stream.
- Problems: see `logs/app.log`.

## Privacy

`images/`, `data/` and `logs/` hold photos, face embeddings and attendance records of real people.
They stay on the PC and are excluded from git. Ask people before enrolling them, and keep the PC
secured.

## License

Copyright (c) 2026 Ramim Ali Siddiqui. All rights reserved. See [LICENSE](LICENSE).

The InsightFace `buffalo_l` models are downloaded separately and are licensed by InsightFace for
non-commercial research use only; commercial use needs a license from them.
