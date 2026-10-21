"""
Smart Attendance System - window.

    python app.py              (or double-click "Smart Attendance.bat")
    python app.py --autostart  start monitoring immediately (used by "start with Windows")
"""
import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2

import attendance as A

STARTUP_FILE = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs",
                            "Startup", "Smart Attendance.bat")
BRANDS = [b for b in A.CAMERA_BRANDS if b != "Hikvision"]


def to_photo(img, max_w, max_h):
    """OpenCV image -> Tk image scaled to fit (no extra libraries needed)."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 2.0)
    if s <= 0:
        return None
    img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tk.PhotoImage(data=b"P6 %d %d 255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes())


class App(tk.Tk):
    def __init__(self, autostart=False):
        super().__init__()
        self.title("Smart Attendance System")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w, h = min(int(sw * 0.92), 1700), min(int(sh * 0.86), 1150)
        self.geometry(f"{w}x{h}+{(sw - w) // 2}+{max((sh - h) // 3, 0)}")
        self.minsize(min(w, 900), min(h, 600))
        self.settings = A.load_settings()
        self.engine = None
        self.tasks = queue.Queue()
        self.shown_events = 0
        self.last_frame = -1
        self.last_info = 0.0
        self.images = {}  # keep Tk images alive

        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=8)
        style.configure("Treeview", rowheight=24)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken", padding=(8, 3)).pack(fill="x", side="bottom")

        self._build_live()
        self._build_people()
        self._build_camera()
        self._build_attendance()
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self.refresh_attendance() if self.nb.index("current") == 3 else None)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self._poll)

        missing = [p for p in ("det_", "w600k_") if not any(f.startswith(p) for f in os.listdir(A.MODEL_DIR))] \
            if os.path.isdir(A.MODEL_DIR) else ["models"]
        if missing:
            messagebox.showerror("Face models missing", f"The face model files are missing in:\n{A.MODEL_DIR}\n\n"
                                 "See README.md, section Models.")
        elif not A.list_people():
            self.status.set("Nobody is enrolled yet. Open the People tab to add someone.")
        if autostart and not missing:
            self.after(800, self.start_engine)

    # ------------------------------------------------------------------ helpers
    def run_bg(self, work, done, busy_text):
        """Run slow work (face learning, camera test, network scan) without freezing the window."""
        self.status.set(busy_text)
        self.config(cursor="watch")

        def worker():
            try:
                self.tasks.put((done, work(), None))
            except Exception as e:  # shown to the user, also written to logs/app.log
                A.log.exception("Background task failed")
                self.tasks.put((done, None, e))

        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                done, result, error = self.tasks.get_nowait()
                self.config(cursor="")
                if error:
                    self.status.set("Ready")
                    messagebox.showerror("Error", str(error))
                else:
                    done(result)
        except queue.Empty:
            pass
        self._update_live()
        self.after(40, self._poll)

    # ------------------------------------------------------------------ LIVE TAB
    def _build_live(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="  Live  ")
        self.video_box = tk.Frame(tab, bg="#111")
        self.video_box.pack(side="left", fill="both", expand=True)
        self.video_box.pack_propagate(False)  # the picture must not resize its own box
        self.video = tk.Label(self.video_box, bg="#111", fg="#bbb", font=("Segoe UI", 13),
                              text="Press  Start monitoring  to begin")
        self.video.pack(fill="both", expand=True)

        side = ttk.Frame(tab, width=340, padding=(10, 0, 0, 0))
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.start_btn = ttk.Button(side, text="Start monitoring", style="Big.TButton", command=self.toggle_engine)
        self.start_btn.pack(fill="x")
        self.live_info = tk.StringVar(value="Camera: " + A.mask(A.build_source(self.settings)))
        ttk.Label(side, textvariable=self.live_info, wraplength=320, justify="left").pack(fill="x", pady=(8, 8))
        ttk.Label(side, text="In view now", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.in_view = tk.Listbox(side, height=6, font=("Segoe UI", 10))
        self.in_view.pack(fill="x", pady=(2, 8))
        ttk.Label(side, text="Recent events", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.recent = tk.Listbox(side, font=("Consolas", 10))
        self.recent.pack(fill="both", expand=True, pady=(2, 8))
        ttk.Button(side, text="Add this face to a person ...", command=self.capture_face).pack(fill="x")

    def toggle_engine(self):
        if self.engine and self.engine.running:
            self.engine.stop()
            self.start_btn.config(text="Stopping ...", state="disabled")
        else:
            self.start_engine()

    def start_engine(self):
        if self.engine and not self.engine.finished:
            return
        if not A.list_people() and not messagebox.askyesno(
                "Nobody enrolled", "Nobody is enrolled yet, so everyone will be logged as Unknown.\n\nStart anyway?"):
            return
        self.engine = A.Engine(self.settings)
        self.engine.start()
        self.shown_events, self.last_frame = 0, -1
        self.recent.delete(0, "end")
        self.start_btn.config(text="Stop monitoring", state="normal")

    def _update_live(self):
        e = self.engine
        if e is None:
            return
        if e.frame_id != self.last_frame and e.preview is not None:
            self.last_frame = e.frame_id
            w, h = self.video_box.winfo_width(), self.video_box.winfo_height()
            img = to_photo(e.preview, w, h) if w > 10 and h > 10 else None
            if img:
                self.images["video"] = img
                self.video.config(image=img, text="")
        if e.tracker and e.tracker.event_count > self.shown_events:
            new = list(e.tracker.events)[-(e.tracker.event_count - self.shown_events):]
            self.shown_events = e.tracker.event_count
            for when, event, name in new:
                self.recent.insert(0, f"{when:%H:%M:%S}  {event:<7} {name}")
            if self.nb.index("current") == 3:
                self.refresh_attendance()
        if time.time() - self.last_info > 0.5 or e.finished:
            self.last_info = time.time()
            people = e.people_in_view()
            self.in_view.delete(0, "end")
            for p in people:
                self.in_view.insert("end", f"{p['name']}   (since {p['first']:%H:%M:%S})")
            self.live_info.set(f"Camera: {A.mask(e.source)}\nStatus: {e.status}\nSpeed: {e.fps:.1f} frames/s")
            self.status.set(e.status if e.running else self.status.get())
        if e.finished:
            self.engine = None
            self.start_btn.config(text="Start monitoring", state="normal")
            self.video.config(image="", text="Monitoring stopped.  Press  Start monitoring  to begin again")
            self.in_view.delete(0, "end")
            self.status.set("Monitoring stopped")
            if e.error:
                messagebox.showerror("Could not start", e.error)

    def capture_face(self):
        frame = self.engine.raw if self.engine else None
        if frame is None:
            messagebox.showinfo("Camera not running", "Start monitoring first, then stand in front of the camera.")
            return
        frame = frame.copy()
        name = simpledialog.askstring("Add this face", "Whose face is this? (existing or new name)",
                                      initialvalue=self.selected_person() or "", parent=self)
        if not name:
            return

        def work():
            if not A.add_frame_photo(name, frame):
                raise ValueError("No face was found in the current camera picture.")
            return A.enroll(lambda m: None)

        self.run_bg(work, lambda r: self._after_enroll(r, f"Saved a camera photo for {name}."), "Learning faces ...")

    # ------------------------------------------------------------------ PEOPLE TAB
    def _build_people(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="  People  ")
        left = ttk.Frame(tab, width=330)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.people_list = ttk.Treeview(left, columns=("name", "photos"), show="headings", selectmode="browse")
        self.people_list.heading("name", text="Name")
        self.people_list.heading("photos", text="Photos")
        self.people_list.column("name", width=200)
        self.people_list.column("photos", width=80, anchor="center")
        self.people_list.pack(fill="both", expand=True)
        self.people_list.bind("<<TreeviewSelect>>", lambda e: self.show_person())
        for text, cmd in (("Add new person ...", self.add_person), ("Add photos to selected ...", self.add_more_photos),
                          ("Delete selected person", self.remove_person), ("Re-learn all faces", self.relearn)):
            ttk.Button(left, text=text, command=cmd).pack(fill="x", pady=(6, 0))

        right = ttk.Frame(tab, padding=(12, 0, 0, 0))
        right.pack(side="left", fill="both", expand=True)
        self.person_title = tk.StringVar(value="Select a person to see their photos")
        ttk.Label(right, textvariable=self.person_title, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(right, foreground="#555", wraplength=760, justify="left", text=(
            "These are the faces the system learned (straightened and cropped). Tip: 5-10 photos per person with "
            "different angles and lighting work best. Use 'Add this face to a person' on the Live tab to add photos taken "
            "by the CCTV camera itself.")).pack(anchor="w", pady=(2, 8))
        self.thumbs = ttk.Frame(right)
        self.thumbs.pack(fill="both", expand=True, anchor="nw")
        self.refresh_people()

    def refresh_people(self, select=None):
        self.people_list.delete(*self.people_list.get_children())
        for name, _, n, _ in A.list_people():
            if not self.people_list.exists(name):  # two folders with the same display name: show once
                self.people_list.insert("", "end", iid=name, values=(name, n))
        if select and self.people_list.exists(select):
            self.people_list.selection_set(select)
        self.show_person()

    def selected_person(self):
        sel = self.people_list.selection()
        return sel[0] if sel else None

    def show_person(self):
        for w in self.thumbs.winfo_children():
            w.destroy()
        self.images = {k: v for k, v in self.images.items() if not k.startswith("thumb")}
        name = self.selected_person()
        if not name:
            self.person_title.set("Select a person to see their photos")
            return
        info = next((p for p in A.list_people() if p[0] == name), None)
        if not info:
            return
        _, folder, n_photos, crops = info
        self.person_title.set(f"{name}  -  {len(crops)} of {n_photos} photo(s) usable")
        for i, path in enumerate(crops):
            img = A.imread(path)
            if img is None:
                continue
            photo = to_photo(img, 112, 112)
            self.images[f"thumb{i}"] = photo
            tk.Label(self.thumbs, image=photo, relief="groove", bd=2).grid(row=i // 6, column=i % 6, padx=4, pady=4)

    def _after_enroll(self, result, message, rejected=(), select=None):
        summary, skipped = result
        if self.engine:
            self.engine.reload_faces()
        self.refresh_people(select)
        lines = [message]
        if rejected:
            lines.append("\nNot added (no usable face):")
            lines += [f"  - {f}: {why}" for f, why in rejected]
        if skipped:
            lines.append("\nSkipped while learning:")
            lines += [f"  - {who}/{f}: {why}" for who, f, why in skipped]
        self.status.set("Faces learned: " + (", ".join(f"{k} ({v})" for k, v in summary.items()) or "nobody"))
        (messagebox.showwarning if rejected or skipped else messagebox.showinfo)("People", "\n".join(lines))

    def _choose_photos(self):
        return list(filedialog.askopenfilenames(parent=self, title="Choose photos",
                                                filetypes=[("Photos", "*.jpg *.jpeg *.png *.bmp *.webp")]))

    def _add(self, name, paths):
        def work():
            added, rejected = A.add_photos(name, paths)
            return added, rejected, A.enroll(lambda m: None)

        def done(r):
            added, rejected, result = r
            self._after_enroll(result, f"Added {added} photo(s) for {A.clean_name(name).title()}.", rejected,
                               select=A.clean_name(name).title())

        self.run_bg(work, done, "Checking photos and learning faces ...")

    def add_person(self):
        name = simpledialog.askstring("Add person", "Person's name:", parent=self)
        if not name:
            return
        try:
            A.clean_name(name)
        except ValueError as e:
            messagebox.showerror("Name", str(e))
            return
        if A.find_person_folder(name) and not messagebox.askyesno(
                "Already exists", f"{name.title()} already exists. Add the photos to this person?"):
            return
        paths = self._choose_photos()
        if paths:
            self._add(name, paths)

    def add_more_photos(self):
        name = self.selected_person()
        if not name:
            messagebox.showinfo("People", "Select a person first.")
            return
        paths = self._choose_photos()
        if paths:
            self._add(name, paths)

    def remove_person(self):
        name = self.selected_person()
        if not name:
            messagebox.showinfo("People", "Select a person first.")
            return
        if not messagebox.askyesno("Delete person", f"Delete {name} and all of their photos?\n\n"
                                   "Past attendance records are kept.", icon="warning"):
            return

        def work():
            A.delete_person(name)
            return A.enroll(lambda m: None)

        self.run_bg(work, lambda r: self._after_enroll(r, f"{name} was deleted."), "Deleting ...")

    def relearn(self):
        self.run_bg(lambda: A.enroll(lambda m: None), lambda r: self._after_enroll(r, "All faces re-learned."),
                    "Learning faces ...")

    # ------------------------------------------------------------------ CAMERA TAB
    def _build_camera(self):
        tab = ttk.Frame(self.nb, padding=10)
        self.nb.add(tab, text="  Camera  ")
        s = self.settings
        self.v = {k: tk.StringVar(value=str(s[k])) for k in A.DEFAULT_SETTINGS}
        if self.v["brand"].get() not in BRANDS:
            self.v["brand"].set(BRANDS[0])

        left = ttk.Frame(tab)
        left.pack(side="left", fill="both", expand=True)
        kind = ttk.LabelFrame(left, text=" 1. Where does the video come from? ", padding=8)
        kind.pack(fill="x")
        for i, (value, text) in enumerate((("cctv", "CCTV camera or recorder on the network"), ("webcam", "Webcam on this PC"),
                                           ("url", "Other stream address (RTSP / HTTP)"), ("file", "Recorded video file (for testing)"))):
            ttk.Radiobutton(kind, text=text, value=value, variable=self.v["camera_type"],
                            command=self._show_camera_fields).grid(row=i % 2, column=i // 2, sticky="w", padx=(0, 24))

        self.fields = ttk.LabelFrame(left, text=" 2. Details ", padding=8)
        self.fields.pack(fill="x", pady=6)
        self.frames = {k: ttk.Frame(self.fields) for k in ("cctv", "webcam", "url", "file")}
        f = self.frames["cctv"]
        rows = [("Brand", ttk.Combobox(f, textvariable=self.v["brand"], values=BRANDS, state="readonly", width=28)),
                ("IP address", ttk.Entry(f, textvariable=self.v["ip"], width=30)),
                ("RTSP port", ttk.Entry(f, textvariable=self.v["port"], width=8)),
                ("Username", ttk.Entry(f, textvariable=self.v["username"], width=30)),
                ("Password", ttk.Entry(f, textvariable=self.v["password"], width=30, show="*")),
                ("Channel (camera no.)", ttk.Spinbox(f, from_=1, to=64, textvariable=self.v["channel"], width=6)),
                ("Quality", ttk.Combobox(f, textvariable=self.v["stream"], values=["main", "sub"], state="readonly", width=8))]
        for r, (label, widget) in enumerate(rows):
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=2, padx=(0, 10))
            widget.grid(row=r, column=1, sticky="w", pady=2)
        ttk.Label(f, foreground="#555", text="main = full quality,  sub = lighter and faster").grid(row=6, column=2, sticky="w", padx=8)
        f = self.frames["webcam"]
        ttk.Label(f, text="Webcam number (0 = built-in)").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Spinbox(f, from_=0, to=9, textvariable=self.v["webcam_index"], width=6).grid(row=0, column=1, sticky="w")
        f = self.frames["url"]
        ttk.Label(f, text="Full address").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(f, textvariable=self.v["custom_url"], width=70).grid(row=0, column=1, sticky="w")
        f = self.frames["file"]
        ttk.Label(f, text="Video file").grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Entry(f, textvariable=self.v["video_file"], width=60).grid(row=0, column=1, sticky="w")
        ttk.Button(f, text="Browse ...", command=self._browse_video).grid(row=0, column=2, padx=6)
        self.address = tk.StringVar()
        ttk.Label(self.fields, textvariable=self.address, foreground="#225", wraplength=620).pack(side="bottom", anchor="w", pady=(8, 0))
        for var in self.v.values():
            var.trace_add("write", lambda *a: self._update_address())

        adv = ttk.LabelFrame(left, text=" 3. Recognition settings ", padding=8)
        adv.pack(fill="x")
        rows = [("Match strictness (0.30 lenient, 0.55 strict)", ttk.Spinbox(adv, from_=0.25, to=0.7, increment=0.01, textvariable=self.v["match_threshold"], width=6)),
                ("Mark as left after (seconds unseen)", ttk.Spinbox(adv, from_=2, to=600, textvariable=self.v["exit_after_sec"], width=6)),
                ("Detector size (640 accurate, 320 fast)", ttk.Combobox(adv, values=["640", "480", "320"], textvariable=self.v["det_size"], state="readonly", width=6)),
                ("Ignore faces smaller than (pixels)", ttk.Spinbox(adv, from_=20, to=300, textvariable=self.v["min_face_px"], width=6))]
        for i, (label, widget) in enumerate(rows):
            ttk.Label(adv, text=label).grid(row=i // 2, column=(i % 2) * 2, sticky="w", pady=2, padx=(0 if i % 2 == 0 else 24, 8))
            widget.grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w")
        self.autostart = tk.BooleanVar(value=os.path.exists(STARTUP_FILE))
        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Test connection", command=self.test_camera).pack(side="left")
        ttk.Button(btns, text="Save settings", style="Big.TButton", command=self.save_camera).pack(side="left", padx=8)
        ttk.Checkbutton(btns, text="Start monitoring when Windows starts", variable=self.autostart).pack(side="left", padx=12)

        right = ttk.Frame(tab, width=420, padding=(12, 0, 0, 0))
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        scan = ttk.LabelFrame(right, text=" Find cameras on this network ", padding=8)
        scan.pack(fill="both", expand=True)
        ttk.Button(scan, text="Search network", command=self.scan).pack(anchor="w")
        self.scan_info = tk.StringVar(value="Looks for cameras and recorders. Click a result to use it.")
        ttk.Label(scan, textvariable=self.scan_info, wraplength=380, foreground="#555").pack(anchor="w", pady=4)
        self.scan_list = ttk.Treeview(scan, columns=("ip", "what"), show="headings", height=5)
        self.scan_list.heading("ip", text="IP address")
        self.scan_list.heading("what", text="Looks like")
        self.scan_list.column("ip", width=120)
        self.scan_list.column("what", width=240)
        self.scan_list.pack(fill="both", expand=True)
        self.scan_list.bind("<<TreeviewSelect>>", self._use_scan_result)
        self.scan_results = {}
        test = ttk.LabelFrame(right, text=" Test picture ", padding=8)
        test.pack(fill="x", pady=(8, 0))
        self.test_msg = tk.StringVar(value="Press Test connection.")
        ttk.Label(test, textvariable=self.test_msg, wraplength=380).pack(anchor="w")
        self.test_img = tk.Label(test)
        self.test_img.pack(pady=4)
        self._show_camera_fields()

    def _show_camera_fields(self):
        for k, fr in self.frames.items():
            fr.pack_forget()
        self.frames[self.v["camera_type"].get()].pack(fill="x")
        self._update_address()

    def _update_address(self):
        try:
            self.address.set("Address used: " + A.mask(A.build_source(self._collect(validate=False))))
        except Exception:
            self.address.set("Address used: (incomplete)")

    def _browse_video(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")])
        if path:
            self.v["video_file"].set(path)

    def _collect(self, validate=True):
        s = dict(self.settings)
        try:
            for k, default in A.DEFAULT_SETTINGS.items():
                raw = self.v[k].get().strip()
                s[k] = type(default)(float(raw)) if isinstance(default, int) and not isinstance(default, bool) \
                    else type(default)(raw)
        except ValueError:
            raise ValueError(f"The value {self.v[k].get()!r} is not valid for {k.replace('_', ' ')}.")
        if validate:
            if s["camera_type"] == "cctv" and not s["ip"]:
                raise ValueError("Please enter the camera's IP address.")
            if s["camera_type"] == "url" and not s["custom_url"]:
                raise ValueError("Please enter the stream address.")
            if s["camera_type"] == "file" and not os.path.isfile(s["video_file"]):
                raise ValueError("Please choose an existing video file.")
            if not 0.2 <= s["match_threshold"] <= 0.8:
                raise ValueError("Match strictness must be between 0.20 and 0.80.")
            if s["exit_after_sec"] < 1 or s["min_face_px"] < 10:
                raise ValueError("Please check the recognition settings.")
        return s

    def test_camera(self):
        try:
            s = self._collect(validate=False)
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        self.test_msg.set("Connecting ...")
        self.test_img.config(image="")

        def done(r):
            ok, msg, frame = r
            self.test_msg.set(("OK - " if ok else "Failed - ") + msg)
            self.status.set(msg)
            if frame is not None:
                self.images["test"] = to_photo(frame, 380, 240)
                self.test_img.config(image=self.images["test"])

        self.run_bg(lambda: A.test_connection(s), done, "Testing the camera ...")

    def save_camera(self):
        try:
            s = self._collect()
        except ValueError as e:
            messagebox.showerror("Settings", str(e))
            return
        A.save_settings(s)
        self.settings = s
        self._set_autostart(self.autostart.get())
        self.live_info.set("Camera: " + A.mask(A.build_source(s)))
        msg = "Settings saved."
        if self.engine and self.engine.running:
            if messagebox.askyesno("Restart monitoring", "Settings saved. Restart monitoring to use them now?"):
                self.engine.stop()
                self.after(300, self._restart_when_stopped)
            return
        self.status.set(msg)
        messagebox.showinfo("Camera", msg)

    def _restart_when_stopped(self):
        if self.engine and not self.engine.finished:
            self.after(300, self._restart_when_stopped)
        else:
            self._update_live()
            self.start_engine()

    def _set_autostart(self, enabled):
        try:
            if enabled:
                pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
                with open(STARTUP_FILE, "w", encoding="utf-8") as f:
                    f.write(f'@echo off\r\nstart "" "{pythonw}" "{os.path.abspath(__file__)}" --autostart\r\n')
            elif os.path.exists(STARTUP_FILE):
                os.remove(STARTUP_FILE)
        except OSError as e:
            messagebox.showerror("Start with Windows", f"Could not change the Windows startup setting:\n{e}")

    def scan(self):
        self.scan_list.delete(*self.scan_list.get_children())
        self.scan_info.set("Searching (about 15 seconds) ...")

        def done(r):
            results, my_ip = r
            self.scan_results = {d["ip"]: d for d in results}
            for d in results:
                self.scan_list.insert("", "end", iid=d["ip"], values=(d["ip"], d["hint"]))
            net = my_ip.rsplit(".", 1)[0] + ".x"
            self.scan_info.set(f"Found {len(results)} possible camera(s) on network {net}." if results else
                               f"No camera or recorder found on network {net}. The CCTV may be on a different "
                               "network, or cloud-only. This PC must be on the same network as the recorder.")
            self.status.set("Network search finished")

        self.run_bg(A.scan_network, done, "Searching the network for cameras ...")

    def _use_scan_result(self, _):
        sel = self.scan_list.selection()
        if not sel:
            return
        d = self.scan_results[sel[0]]
        self.v["camera_type"].set("cctv")
        self.v["ip"].set(d["ip"])
        self.v["port"].set(str(d["rtsp_port"]))
        if d["brand"]:
            self.v["brand"].set(d["brand"])
        self._show_camera_fields()
        self.status.set(f"Filled in {d['ip']}. Enter the username and password, then Test connection.")

    # ------------------------------------------------------------------ ATTENDANCE TAB
    def _build_attendance(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="  Attendance  ")
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="Day").pack(side="left")
        self.day = tk.StringVar(value=f"{datetime.now():%Y-%m-%d}")
        self.day_box = ttk.Combobox(top, textvariable=self.day, width=12)
        self.day_box.pack(side="left", padx=6)
        self.day_box.bind("<<ComboboxSelected>>", lambda e: self.refresh_attendance())
        self.day_box.bind("<Return>", lambda e: self.refresh_attendance())
        ttk.Button(top, text="Refresh", command=self.refresh_attendance).pack(side="left")
        ttk.Button(top, text="Export ...", command=self.export_attendance).pack(side="right")
        ttk.Button(top, text="Open snapshots", command=lambda: self._open(os.path.join(A.LOG_DIR, "snapshots"))).pack(side="right", padx=6)
        ttk.Button(top, text="Open logs folder", command=lambda: self._open(A.LOG_DIR)).pack(side="right")
        cols = ("name", "in", "out", "stayed")
        self.table = ttk.Treeview(tab, columns=cols, show="headings")
        for c, text, w in zip(cols, ("Name", "Came in", "Went out", "Stayed"), (240, 140, 160, 140)):
            self.table.heading(c, text=text)
            self.table.column(c, width=w, anchor="w" if c == "name" else "center")
        self.table.tag_configure("unknown", foreground="#b35c00")
        self.table.pack(fill="both", expand=True, pady=8)
        self.summary = tk.StringVar()
        ttk.Label(tab, textvariable=self.summary).pack(anchor="w")

    def refresh_attendance(self):
        days = A.available_days()
        today = f"{datetime.now():%Y-%m-%d}"
        self.day_box.config(values=sorted(set(days + [today]), reverse=True))
        day = self.day.get().strip()
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Day", "Please use the format YYYY-MM-DD.")
            return
        self.table.delete(*self.table.get_children())
        visits = A.read_visits(day)
        for v in visits:
            out = v["out"] if v["out_date"] in ("", day) else f"{v['out']} (next day)"
            self.table.insert("", "end", values=(v["name"], v["in"], out or "still inside", v["stayed"]),
                              tags=("unknown",) if v["name"].startswith("Unknown") else ())
        known = sorted({v["name"] for v in visits if not v["name"].startswith("Unknown")})
        strangers = {v["name"] for v in visits if v["name"].startswith("Unknown")}
        self.summary.set(f"{len(visits)} visit(s).  Known: {', '.join(known) or 'none'}.  Unknown people: {len(strangers)}.")

    def export_attendance(self):
        day = self.day.get().strip()
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv", initialfile=f"attendance_{day}.csv",
                                            filetypes=[("CSV (Excel)", "*.csv")])
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["Name", "Came in", "Went out", "Stayed"])
            for v in A.read_visits(day):
                w.writerow([v["name"], v["in"], v["out"], v["stayed"]])
        self.status.set(f"Exported to {path}")

    @staticmethod
    def _open(path):
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    # ------------------------------------------------------------------ closing
    def on_close(self):
        if self.engine and self.engine.running:
            if not messagebox.askyesno("Quit", "Monitoring is running. Stop it and quit?\n"
                                       "Everyone currently in view will be logged OUT."):
                return
            self.engine.stop()
            self.engine.join(15)
        self.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # sharp text on high-resolution screens
    except Exception:
        pass
    App(autostart="--autostart" in sys.argv).mainloop()


if __name__ == "__main__":
    main()
