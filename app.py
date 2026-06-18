import os
import sys
import time
import cv2
import numpy as np
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import threading

# Import local modules
from download_assets import download_file, MODEL_URLS, MODEL_PATH
from emotion_detector import EmotionDetector

# Set CustomTkinter theme and appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Real-Time Face Emotion Detection Dashboard")
        self.geometry("1100x650")
        self.minsize(1000, 600)
        
        # Initialize variables
        self.detector = None
        self.cap = None
        self.camera_running = False
        self.camera_index = 0
        self.start_time = None
        self.elapsed_seconds = 0
        self.dominant_history = []
        self.last_logged_emotion = None
        self.last_log_time = 0
        self.show_bounding_boxes = True
        self.current_frame = None
        
        # Tuning parameters
        self.pad_ratio = 0.10
        self.equalize = True
        self.smoothing_factor = 0.25
        self.smoothed_probs = None
        
        # Colors for each emotion
        self.colors = {
            "Neutral": "#85929E",   # Slate Gray
            "Happy": "#2ECC71",     # Emerald Green
            "Surprise": "#F1C40F",  # Sunflower Yellow
            "Sad": "#3498DB",       # Sky Blue
            "Angry": "#E74C3C",     # Alizarin Red
            "Disgust": "#E67E22",   # Orange
            "Fear": "#9B59B6",      # Purple
            "Contempt": "#A6ACAF"   # Light Gray
        }
        
        # Map emotion names for layout ordering
        self.emotions = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
        
        # Check if assets exist
        if not os.path.exists(MODEL_PATH):
            self.build_loading_ui()
        else:
            self.initialize_detector_and_build_ui()

    # --- Loading UI ---
    def build_loading_ui(self):
        """Displays a beautiful loading screen to download the pre-trained model."""
        self.loading_frame = ctk.CTkFrame(self, corner_radius=15)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.6, relheight=0.5)
        
        title = ctk.CTkLabel(
            self.loading_frame, 
            text="Setting Up Emotion Detection System", 
            font=("Segoe UI", 20, "bold")
        )
        title.pack(pady=(40, 10))
        
        subtitle = ctk.CTkLabel(
            self.loading_frame, 
            text="Downloading pre-trained FERPlus ONNX model (approx. 35 MB)...", 
            font=("Segoe UI", 12),
            text_color="#BDC3C7"
        )
        subtitle.pack(pady=5)
        
        self.progress_bar = ctk.CTkProgressBar(self.loading_frame, width=350)
        self.progress_bar.pack(pady=30)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(
            self.loading_frame, 
            text="Status: Starting download...", 
            font=("Segoe UI", 11, "italic"),
            text_color="#7F8C8D"
        )
        self.progress_label.pack(pady=5)
        
        # Start download thread
        threading.Thread(target=self.download_model_thread, daemon=True).start()

    def download_model_thread(self):
        """Runs the model downloader in a background thread."""
        def progress_callback(downloaded, total):
            if total > 0:
                fraction = downloaded / total
                percent = fraction * 100
                self.progress_bar.set(fraction)
                self.progress_label.configure(
                    text=f"Status: Downloaded {downloaded/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB ({percent:.1f}%)"
                )
            else:
                self.progress_label.configure(
                    text=f"Status: Downloaded {downloaded/(1024*1024):.1f}MB"
                )
            self.update_idletasks()
            
        success = download_file(MODEL_URLS, MODEL_PATH, progress_callback)
        
        if success:
            self.progress_label.configure(text="Status: Download completed! Initializing model...")
            self.progress_bar.set(1.0)
            self.after(1000, self.transition_to_dashboard)
        else:
            self.progress_label.configure(text="Status: Download failed! Please check your connection.")

    def transition_to_dashboard(self):
        """Removes the loading UI and initializes the main dashboard."""
        self.loading_frame.destroy()
        self.initialize_detector_and_build_ui()

    def initialize_detector_and_build_ui(self):
        """Initializes the EmotionDetector backend and builds the dashboard GUI."""
        try:
            self.detector = EmotionDetector(MODEL_PATH)
            self.build_dashboard_ui()
            self.start_camera()  # Auto-start camera on launch
        except Exception as e:
            # Show error inside window
            err_label = ctk.CTkLabel(
                self, 
                text=f"Critical Initialization Error:\n{e}", 
                font=("Segoe UI", 16, "bold"),
                text_color="#E74C3C"
            )
            err_label.pack(expand=True)

    # --- Dashboard UI Layout ---
    def build_dashboard_ui(self):
        """Builds the main multi-panel dashboard UI."""
        # Top Header Bar
        self.header_frame = ctk.CTkFrame(self, height=70, corner_radius=0, fg_color="#1A1D20")
        self.header_frame.pack(fill="x", side="top")
        
        title_label = ctk.CTkLabel(
            self.header_frame, 
            text="REAL-TIME FACE EMOTION DETECTOR", 
            font=("Segoe UI", 22, "bold"),
            text_color="#3498DB"
        )
        title_label.pack(side="left", padx=20, pady=15)
        
        subtitle_label = ctk.CTkLabel(
            self.header_frame, 
            text="Active Session Tracker", 
            font=("Segoe UI", 12, "italic"),
            text_color="#7F8C8D"
        )
        subtitle_label.pack(side="left", padx=(10, 0), pady=(22, 15))
        
        # Outer body container
        self.body_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.body_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Left Panel (Webcam View & Controls)
        self.left_panel = ctk.CTkFrame(self.body_frame, corner_radius=12)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Video Frame container
        self.video_container = ctk.CTkFrame(self.left_panel, corner_radius=8, fg_color="#0D0F11")
        self.video_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Label to display video frames
        self.video_label = ctk.CTkLabel(self.video_container, text="Camera Stream Off", font=("Segoe UI", 14))
        self.video_label.pack(expand=True, fill="both")
        
        # Tuning Settings Frame
        self.tuning_frame = ctk.CTkFrame(self.left_panel, corner_radius=8)
        self.tuning_frame.pack(fill="x", padx=15, pady=(0, 10))
        
        # Grid layout for tuning controls
        self.tuning_frame.grid_columnconfigure(0, weight=1)
        self.tuning_frame.grid_columnconfigure(1, weight=1)
        self.tuning_frame.grid_columnconfigure(2, weight=1)
        
        # 1. Padding control
        pad_container = ctk.CTkFrame(self.tuning_frame, fg_color="transparent")
        pad_container.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.lbl_padding = ctk.CTkLabel(pad_container, text="Face Crop Padding: 10%", font=("Segoe UI", 11))
        self.lbl_padding.pack(anchor="w")
        
        self.sld_padding = ctk.CTkSlider(
            pad_container, 
            from_=0.0, 
            to=0.3, 
            number_of_steps=30,
            command=self.update_padding_val
        )
        self.sld_padding.pack(fill="x", pady=(5, 0))
        self.sld_padding.set(0.10)
        
        # 2. Smoothing control
        smooth_container = ctk.CTkFrame(self.tuning_frame, fg_color="transparent")
        smooth_container.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        self.lbl_smoothing = ctk.CTkLabel(smooth_container, text="Temporal Smoothing: 0.25 (Balanced)", font=("Segoe UI", 11))
        self.lbl_smoothing.pack(anchor="w")
        
        self.sld_smoothing = ctk.CTkSlider(
            smooth_container, 
            from_=0.05, 
            to=1.0, 
            number_of_steps=19,
            command=self.update_smoothing_val
        )
        self.sld_smoothing.pack(fill="x", pady=(5, 0))
        self.sld_smoothing.set(0.25)
        
        # 3. Equalize checkbox container
        chk_container = ctk.CTkFrame(self.tuning_frame, fg_color="transparent")
        chk_container.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        self.chk_equalize = ctk.CTkCheckBox(
            chk_container, 
            text="Equalize Contrast (Lighting)", 
            command=self.toggle_equalize,
            font=("Segoe UI", 11)
        )
        self.chk_equalize.select()
        self.chk_equalize.pack(anchor="w", pady=(15, 0))
        
        # Left Panel Controls
        self.controls_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.controls_frame.pack(fill="x", side="bottom", padx=15, pady=(0, 15))
        
        self.btn_toggle_cam = ctk.CTkButton(
            self.controls_frame, 
            text="Stop Camera", 
            command=self.toggle_camera,
            fg_color="#E74C3C", 
            hover_color="#C0392B", 
            font=("Segoe UI", 13, "bold"),
            width=120
        )
        self.btn_toggle_cam.pack(side="left", padx=(0, 10))
        
        self.btn_snapshot = ctk.CTkButton(
            self.controls_frame, 
            text="Capture Snapshot", 
            command=self.take_snapshot,
            fg_color="#3498DB", 
            hover_color="#2980B9", 
            font=("Segoe UI", 13),
            width=140
        )
        self.btn_snapshot.pack(side="left", padx=5)
        
        # Camera Index Dropdown & Labels
        ctk.CTkLabel(self.controls_frame, text="Camera Index:", font=("Segoe UI", 12)).pack(side="left", padx=(20, 5))
        self.cam_select = ctk.CTkOptionMenu(
            self.controls_frame, 
            values=["0", "1", "2"], 
            command=self.change_camera,
            width=70
        )
        self.cam_select.pack(side="left")
        self.cam_select.set(str(self.camera_index))
        
        # Toggle Bounding Box Checkbox
        self.chk_bbox = ctk.CTkCheckBox(
            self.controls_frame, 
            text="Overlay Box", 
            command=self.toggle_bbox,
            font=("Segoe UI", 12)
        )
        self.chk_bbox.select()
        self.chk_bbox.pack(side="right", padx=(10, 0))

        # Right Panel (Expression Confidence & Statistics)
        self.right_panel = ctk.CTkFrame(self.body_frame, width=380, corner_radius=12)
        self.right_panel.pack(side="right", fill="both", padx=(10, 0))
        self.right_panel.pack_propagate(False) # Prevent size adjustments based on children
        
        # Real-time Expression Analytics Frame
        self.analytics_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.analytics_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(
            self.analytics_frame, 
            text="Expression Confidence", 
            font=("Segoe UI", 15, "bold"),
            text_color="#3498DB"
        ) .pack(anchor="w", pady=(0, 10))
        
        # Real-time bar chart canvas
        self.chart_canvas = tk.Canvas(
            self.analytics_frame, 
            width=350, 
            height=280, 
            bg="#1A1D20", 
            highlightthickness=0
        )
        self.chart_canvas.pack(fill="x")
        
        # Build cached items for bar chart to prevent screen tearing/lag
        self.bar_ids = {}
        self.y_coords = {}
        for i, emotion in enumerate(self.emotions):
            y = 15 + i * 32
            self.y_coords[emotion] = y
            # Draw label name
            self.chart_canvas.create_text(
                10, y + 8, 
                text=emotion, 
                anchor="w", 
                fill="#ECF0F1", 
                font=("Segoe UI", 11, "bold")
            )
            # Draw background slot
            self.chart_canvas.create_rectangle(90, y, 310, y + 16, fill="#2C3E50", outline="")
            # Draw empty foreground bar (will update dynamic length later)
            fg_id = self.chart_canvas.create_rectangle(90, y, 90, y + 16, fill=self.colors[emotion], outline="")
            # Draw percentage label
            val_id = self.chart_canvas.create_text(
                315, y + 8, 
                text="0.0%", 
                anchor="w", 
                fill="#BDC3C7", 
                font=("Segoe UI", 10)
            )
            self.bar_ids[emotion] = (fg_id, val_id)
            
        # Session Analytics Frame
        self.stats_frame = ctk.CTkFrame(self.right_panel, fg_color="#1A1D20", corner_radius=8)
        self.stats_frame.pack(fill="x", padx=15, pady=5)
        
        # Stat grid
        self.lbl_faces = ctk.CTkLabel(
            self.stats_frame, 
            text="Faces Tracked: 0", 
            font=("Segoe UI", 12, "bold")
        )
        self.lbl_faces.grid(row=0, column=0, padx=15, pady=10, sticky="w")
        
        self.lbl_timer = ctk.CTkLabel(
            self.stats_frame, 
            text="Duration: 00:00", 
            font=("Segoe UI", 12, "bold")
        )
        self.lbl_timer.grid(row=0, column=1, padx=15, pady=10, sticky="e")
        
        self.lbl_dominant = ctk.CTkLabel(
            self.stats_frame, 
            text="Dominant Expression: None", 
            font=("Segoe UI", 13, "bold"),
            text_color="#BDC3C7"
        )
        self.lbl_dominant.grid(row=1, column=0, columnspan=2, padx=15, pady=(0, 10), sticky="w")
        
        # Activity History Log Frame
        self.log_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.log_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        
        ctk.CTkLabel(
            self.log_frame, 
            text="Emotion History Log", 
            font=("Segoe UI", 13, "bold"),
            text_color="#3498DB"
        ).pack(anchor="w", pady=(5, 5))
        
        self.txt_log = ctk.CTkTextbox(
            self.log_frame, 
            wrap="none", 
            font=("Consolas", 10), 
            fg_color="#0D0F11"
        )
        self.txt_log.pack(fill="both", expand=True)
        self.txt_log.configure(state="disabled")
        
        # Add window closing handler
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --- Webcam Stream Control & Detection Loop ---
    def start_camera(self):
        """Initializes and starts the webcam frame capture loop."""
        if not self.camera_running:
            # Try DirectShow backend first (highly recommended for Windows to prevent MSMF grab issues)
            self.log_to_dashboard(f"Initializing camera {self.camera_index} (DirectShow)...")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            
            # Verify if it actually streams
            is_ok = False
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    is_ok = True
                else:
                    self.cap.release()
            
            if not is_ok:
                # Fallback to default MSMF/Auto backend
                self.log_to_dashboard(f"DirectShow inactive. Falling back to default backend...")
                self.cap = cv2.VideoCapture(self.camera_index)
                
            if not self.cap.isOpened():
                self.log_to_dashboard("ERROR: Could not open camera. Try a different camera index.")
                self.video_label.configure(text="Camera Initialization Failed.\n\nPlease check permissions or change Camera Index.")
                self.cap = None
                return
            
            # Successful capture
            self.camera_running = True
            self.start_time = time.time()
            self.btn_toggle_cam.configure(text="Stop Camera", fg_color="#E74C3C", hover_color="#C0392B")
            self.log_to_dashboard(f"Camera session started (Index: {self.camera_index})")
            
            # Start UI timer updating
            self.update_timer()
            
            # Trigger first frame update
            self.after(10, self.update_frame)

    def stop_camera(self):
        """Stops the webcam and updates states."""
        if self.camera_running:
            self.camera_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_label.configure(text="Camera Stream Off")
            self.btn_toggle_cam.configure(text="Start Camera", fg_color="#2ECC71", hover_color="#27AE60")
            self.log_to_dashboard("Camera session stopped.")
            self.lbl_faces.configure(text="Faces Tracked: 0")

    def toggle_camera(self):
        """Toggles camera stream on and off."""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def change_camera(self, choice):
        """Changes the active camera index."""
        new_index = int(choice)
        if new_index != self.camera_index:
            self.camera_index = new_index
            if self.camera_running:
                self.stop_camera()
                self.start_camera()

    def toggle_bbox(self):
        """Toggles face overlay boxes on/off."""
        self.show_bounding_boxes = bool(self.chk_bbox.get())

    def update_frame(self):
        """Captures a frame, performs face/emotion inference, draws overlays, and updates UI."""
        if not self.camera_running or self.cap is None:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.log_to_dashboard("WARNING: Failed to grab frame from camera.")
            self.after(30, self.update_frame)
            return

        # Flip horizontally for natural mirror feel
        frame = cv2.flip(frame, 1)
        
        # Convert to Grayscale for face detection and model input
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detector.detect_faces(gray)
        self.lbl_faces.configure(text=f"Faces Tracked: {len(faces)}")
        
        # Reset smoothed probabilities if no faces are in view
        if len(faces) == 0:
            self.smoothed_probs = None
            
        # Variable to accumulate probabilities if faces are detected
        dominant_label = "None"
        dominant_conf = 0.0
        active_prediction = None
        
        for face_box in faces:
            x, y, w, h = face_box
            
            # Predict emotion with pad_ratio and equalization parameters
            prediction = self.detector.predict_emotion(
                gray, 
                face_box, 
                pad_ratio=self.pad_ratio, 
                equalize=self.equalize
            )
            
            if prediction:
                # Cache prediction for the dominant face (usually the first one detected)
                if active_prediction is None:
                    # Apply Temporal Smoothing (EMA) to prevent flickering bars
                    probs = prediction["probabilities"]
                    if self.smoothed_probs is None:
                        self.smoothed_probs = probs.copy()
                    else:
                        for emotion in self.emotions:
                            self.smoothed_probs[emotion] = (
                                self.smoothing_factor * probs[emotion] + 
                                (1.0 - self.smoothing_factor) * self.smoothed_probs[emotion]
                            )
                    
                    # Update values with smoothed results
                    prediction["probabilities"] = self.smoothed_probs.copy()
                    dominant_label = max(self.smoothed_probs, key=self.smoothed_probs.get)
                    dominant_conf = self.smoothed_probs[dominant_label]
                    prediction["dominant"] = dominant_label
                    prediction["confidence"] = dominant_conf
                    
                    active_prediction = prediction
                
                # Draw facial bounding box and text overlays if checked
                if self.show_bounding_boxes:
                    color = self.hex_to_bgr(self.colors[prediction["dominant"]])
                    
                    # Rounded rectangle border
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw a solid bar for the text background
                    cv2.rectangle(frame, (x, y - 25), (x + w, y), color, -1)
                    text = f"{prediction['dominant']} ({prediction['confidence']:.0f}%)"
                    cv2.putText(
                        frame, 
                        text, 
                        (x + 5, y - 7), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        1, 
                        cv2.LINE_AA
                    )

        # Store current frame for snapshots (convert BGR to RGB first)
        self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update graphical bar charts and labels
        if active_prediction:
            self.update_charts(active_prediction["probabilities"])
            self.lbl_dominant.configure(
                text=f"Dominant Expression: {dominant_label} ({dominant_conf:.1f}%)",
                text_color=self.colors[dominant_label]
            )
            
            # Log transition to activity history log if emotion changes (debounce log intervals)
            current_time = time.time()
            if (dominant_label != self.last_logged_emotion and current_time - self.last_log_time > 2.0) or (current_time - self.last_log_time > 10.0):
                self.log_to_dashboard(f"Face detected: {dominant_label} ({dominant_conf:.0f}%)")
                self.last_logged_emotion = dominant_label
                self.last_log_time = current_time
        else:
            # If no faces are detected, gradually fade out charts to 0
            self.decay_charts()
            self.lbl_dominant.configure(text="Dominant Expression: None", text_color="#BDC3C7")

        # Convert frame to PhotoImage and render to video label
        img = Image.fromarray(self.current_frame)
        
        # Calculate dynamic resizing coordinates to maintain aspect ratio
        container_w = self.video_container.winfo_width()
        container_h = self.video_container.winfo_height()
        if container_w > 100 and container_h > 100:
            img_w, img_h = img.size
            aspect = img_w / img_h
            if container_w / container_h > aspect:
                new_h = container_h
                new_w = int(container_h * aspect)
            else:
                new_w = container_w
                new_h = int(container_w / aspect)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # Keep a reference to prevent garbage collection
        
        # Queue next frame capture (~30 FPS)
        self.after(30, self.update_frame)

    # --- Chart Drawing & Decay ---
    def update_charts(self, probs):
        """Updates the visual canvas bars with new probability percentages."""
        for emotion, prob in probs.items():
            fg_id, val_id = self.bar_ids[emotion]
            y = self.y_coords[emotion]
            # Max width is X = 310 (which is 220px maximum bar length starting from X = 90)
            target_x = 90 + int(220 * (prob / 100))
            self.chart_canvas.coords(fg_id, 90, y, target_x, y + 16)
            self.chart_canvas.itemconfig(val_id, text=f"{prob:.1f}%")

    def decay_charts(self):
        """Fades out chart bars gradually if no face is detected."""
        for emotion in self.emotions:
            fg_id, val_id = self.bar_ids[emotion]
            coords = self.chart_canvas.coords(fg_id)
            if coords:
                curr_width = coords[2] - coords[0]
                if curr_width > 0:
                    new_width = max(0, curr_width - 15)
                    y = self.y_coords[emotion]
                    self.chart_canvas.coords(fg_id, 90, y, 90 + new_width, y + 16)
                    # Update label text to 0.0% if bar size collapses
                    if new_width == 0:
                        self.chart_canvas.itemconfig(val_id, text="0.0%")
                    else:
                        prob = (new_width / 220) * 100
                        self.chart_canvas.itemconfig(val_id, text=f"{prob:.1f}%")

    # --- UI Timer Updates ---
    def update_timer(self):
        """Updates the session duration timer label."""
        if self.camera_running and self.start_time is not None:
            self.elapsed_seconds = int(time.time() - self.start_time)
            mins = self.elapsed_seconds // 60
            secs = self.elapsed_seconds % 60
            self.lbl_timer.configure(text=f"Duration: {mins:02d}:{secs:02d}")
            self.after(1000, self.update_timer)

    # --- Dashboard Action Handlers ---
    def take_snapshot(self):
        """Saves current camera frame with bounding boxes overlay to a file."""
        if self.current_frame is None:
            self.log_to_dashboard("WARNING: Capture failed. Camera is not running.")
            return
            
        os.makedirs("snapshots", exist_ok=True)
        filename = f"snapshots/snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        
        try:
            # Image is in RGB, convert back to BGR for OpenCV saving
            bgr_img = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, bgr_img)
            self.log_to_dashboard(f"Snapshot saved: {filename}")
        except Exception as e:
            self.log_to_dashboard(f"ERROR: Failed to save snapshot: {e}")

    def log_to_dashboard(self, message):
        """Appends a timestamped status log to the on-screen terminal box."""
        timestamp = time.strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"
        
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", formatted)
        self.txt_log.see("end")
        self.txt_log.configure(state="disabled")
        print(message)  # Mirror to standard console output

    # --- Tuning Settings Callbacks ---
    def update_padding_val(self, val):
        self.pad_ratio = float(val)
        self.lbl_padding.configure(text=f"Face Crop Padding: {int(self.pad_ratio * 100)}%")

    def update_smoothing_val(self, val):
        self.smoothing_factor = float(val)
        desc = "Balanced"
        if self.smoothing_factor < 0.15:
            desc = "Stable / Slow"
        elif self.smoothing_factor > 0.6:
            desc = "Fast / Dynamic"
        self.lbl_smoothing.configure(text=f"Temporal Smoothing: {self.smoothing_factor:.2f} ({desc})")

    def toggle_equalize(self):
        self.equalize = bool(self.chk_equalize.get())

    # --- Helper Utilities ---
    def hex_to_bgr(self, hex_str):
        """Converts hex color code string to BGR tuple for OpenCV drawing."""
        hex_str = hex_str.lstrip("#")
        r, g, b = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        return (b, g, r)

    def on_closing(self):
        """Triggered when user clicks the window close button."""
        self.stop_camera()
        self.destroy()
        sys.exit(0)

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
