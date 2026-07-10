import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
import threading
import time
import winsound
import webbrowser
import os
import mediapipe as mp
import math 

# =========================================================
# استدعاء النماذج من ملف core
# =========================================================
from core import (
    load_eye_model,
    load_yawn_model,
    extract_region_from_landmarks,
    get_combined_eye_prediction,
    get_mouth_prediction,
    draw_bbox,
    DecisionEngine,
    LEFT_EYE_IDXS,
    RIGHT_EYE_IDXS,
    MOUTH_IDXS,
    calculate_ear
)

# interface settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# =========================================================
# hybrid alert and playlist management system
# =========================================================
class AlertSystem:
    def __init__(self):
        self.alarm_on = False
        self.alarm_thread = None
        
        self.drowsy_count = 0        
        self.playlist_opened = False 
        self.alarm_start_time = 0
        self.last_sleep_duration = 0 
        
        self.current_mode = "أغاني حماسية"
        self.playback_source = "أونلاين"
        
        # Online Playlist URLs
        self.playlists_online = {
            "قرآن كريم": "https://open.spotify.com/intl-ar/artist/4u06Ay3EVwnXcz2V9vB2QK?si=bpia-YKLS-OanpAxWcLJKA",
            "ترددات يقظة": "https://open.spotify.com/playlist/0BXXMlKEm14lRTeEgjTu6V?si=BcbWx9jiS2-pdTZhBfb3yw",
            "أغاني حماسية": "https://open.spotify.com/playlist/1HkSGVAjLxePrUv3j3xrQw?si=VEOqln8IQcCrncs5talK4w"
        }
        
        """علشان كل حاجه تشتغل تمام هتحتاج تغير المسارات بتاعت الاصوات المحليه بالمسار الي
        علي جهازك و تحط المسار الي علي الاب بتاعك 
          و الاصوات دي هتلقيها في ملف اسمه offSound 
          ام الاولاين مش هتحتاج تغير حاجه 
          """
        
        # Local File Paths 
        self.playlists_local = {
            "قرآن كريم": r"D:\prgrames\program.py\programes\rwad\Gradwationpto\DEPI Project\offSound\020.mp3",
            "أغاني حماسية": r"D:\prgrames\program.py\programes\rwad\Gradwationpto\DEPI Project\offSound\جوليا بطرس - مدلي أغاني وطنية _ لايف بلاتيا 2014 _ Julia Boutros - Medly Platea.mp3"
        }

    def set_mode(self, mode_name):
        self.current_mode = mode_name
        self.playlist_opened = False 
        print(f"Content changed to: {self.current_mode}")

    def set_playback_source(self, source_name):
        self.playback_source = source_name
        self.playlist_opened = False
        print(f"Playback source changed to: {self.playback_source}")

    def play_alarm(self):
        while self.alarm_on:
            winsound.Beep(1000, 500)
            time.sleep(0.1)

    def _open_playlist(self):
        if not self.playlist_opened:
            print(f"Starting playback for: {self.current_mode} - Source: {self.playback_source}")
            
            if self.playback_source == "أونلاين":
                link = self.playlists_online.get(self.current_mode, self.playlists_online["أغاني حماسية"])
                webbrowser.open(link)
            else: 
                # Local Playback
                if self.current_mode in self.playlists_local:
                    file_path = os.path.abspath(self.playlists_local[self.current_mode]) 
                    if os.path.exists(file_path):
                        webbrowser.open(f"file:///{file_path}")
                    else:
                        print(f"Error: Local file not found at path: {file_path}")
                else:
                    print(f"Error: Content ({self.current_mode}) is not available in local mode.")
                    
            self.playlist_opened = True

    def trigger_alarm(self):
        if not self.alarm_on:
            self.alarm_on = True
            self.alarm_start_time = time.time()
            self.drowsy_count += 1
            print(f"Drowsiness alert number: {self.drowsy_count}")

            if self.drowsy_count >= 2:
                print("Driver fell asleep for the second time! Activating AI mitigation...")
                self._open_playlist()

            self.alarm_thread = threading.Thread(target=self.play_alarm)
            self.alarm_thread.start()
        else:
            if time.time() - self.alarm_start_time >= 10:
                print("Danger: Driver eyes closed for too long! Activating AI mitigation...")
                self._open_playlist()

    def stop_alarm(self):
        if self.alarm_on:
            if self.alarm_start_time > 0:
                self.last_sleep_duration = int(time.time() - self.alarm_start_time)
                
            self.alarm_on = False
            self.alarm_start_time = 0
            self.playlist_opened = False 
            if self.alarm_thread is not None:
                self.alarm_thread.join()

    def get_sleep_duration(self):
        if self.alarm_on and self.alarm_start_time > 0:
            return int(time.time() - self.alarm_start_time)
        return self.last_sleep_duration

    def reset_system(self):
        self.drowsy_count = 0
        self.playlist_opened = False
        self.alarm_start_time = 0
        self.last_sleep_duration = 0 
        print("System counters reset for a new driving shift.")

# =========================================================
# main user interface 
# =========================================================
class DriverMonitoringApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Driver Monitoring System - Team Role 5")
        self.geometry("950x700") 

        self.alert_system = AlertSystem()
        self.cap = None
        self.is_running = False
        self.current_image = None 
        
        # نعومة الصوره 
        self.smooth_zoom = 1.0

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar 
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(12, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="AI Monitor", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Loading...", font=ctk.CTkFont(size=18, weight="bold"), text_color="orange")
        self.status_label.grid(row=1, column=0, padx=20, pady=10)

        self.eye_label = ctk.CTkLabel(self.sidebar_frame, text="Eyes Closed: 0 frames", font=ctk.CTkFont(size=14))
        self.eye_label.grid(row=2, column=0, padx=20, pady=5, sticky="w")

        self.yawn_label = ctk.CTkLabel(self.sidebar_frame, text="Yawning: No", font=ctk.CTkFont(size=14))
        self.yawn_label.grid(row=3, column=0, padx=20, pady=5, sticky="w")

        self.alarms_count_label = ctk.CTkLabel(self.sidebar_frame, text="Alarms Triggered: 0", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FF4500")
        self.alarms_count_label.grid(row=4, column=0, padx=20, pady=(15, 5), sticky="w")

        self.sleep_timer_label = ctk.CTkLabel(self.sidebar_frame, text="Last Sleep: 0s", font=ctk.CTkFont(size=16, weight="bold"), text_color="#FF4500")
        self.sleep_timer_label.grid(row=5, column=0, padx=20, pady=(5, 10), sticky="w")

        self.distance_label = ctk.CTkLabel(self.sidebar_frame, text="Distance: -- cm", font=ctk.CTkFont(size=16, weight="bold"), text_color="#00FFFF")
        self.distance_label.grid(row=6, column=0, padx=20, pady=(5, 15), sticky="w")

        self.alert_menu_label = ctk.CTkLabel(self.sidebar_frame, text="Alert Mode:", font=ctk.CTkFont(size=14, weight="bold"))
        self.alert_menu_label.grid(row=7, column=0, padx=20, pady=(10, 5), sticky="w")

        self.alert_mode_var = ctk.StringVar(value="أغاني حماسية")
        self.alert_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["قرآن كريم", "ترددات يقظة", "أغاني حماسية"],
            command=self.change_alert_mode,
            variable=self.alert_mode_var,
            fg_color="#2B2B2B", button_color="#1F538D", button_hover_color="#14375E"
        )
        self.alert_menu.grid(row=8, column=0, padx=20, pady=5)

        self.source_menu_label = ctk.CTkLabel(self.sidebar_frame, text="Source:", font=ctk.CTkFont(size=14, weight="bold"))
        self.source_menu_label.grid(row=9, column=0, padx=20, pady=(10, 5), sticky="w")

        self.source_mode_var = ctk.StringVar(value="أونلاين")
        self.source_menu = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=["أونلاين", "محلي"],
            command=self.change_source_mode,
            variable=self.source_mode_var,
            fg_color="#4F4F4F", button_color="#2E8B57", button_hover_color="#1E5C3A"
        )
        self.source_menu.grid(row=10, column=0, padx=20, pady=5)

        self.reset_btn = ctk.CTkButton(self.sidebar_frame, text="Reset Shift", command=self.reset_trip, fg_color="#F8A40A", hover_color="#C78205")
        self.reset_btn.grid(row=11, column=0, padx=20, pady=(30, 20))

    

        self.video_frame = ctk.CTkLabel(self, text="Starting Camera...", font=ctk.CTkFont(size=20))
        self.video_frame.grid(row=0, column=1, padx=20, pady=20)

        #  تحميل الموديلات 
        self.update() 
        self.eye_model = load_eye_model()
        self.yawn_model = load_yawn_model() 
        self.engine = DecisionEngine()
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.status_label.configure(text="Status: READY", text_color="green")
        self.after(500, self.start_camera)

    def change_alert_mode(self, choice):
        self.alert_system.set_mode(choice)

    def change_source_mode(self, choice):
        self.alert_system.set_playback_source(choice)
        
        if choice == "محلي":
            self.alert_menu.configure(values=["قرآن كريم", "أغاني حماسية"])
            if self.alert_mode_var.get() == "ترددات يقظة":
                self.alert_mode_var.set("أغاني حماسية")
                self.alert_system.set_mode("أغاني حماسية")
        else:
            self.alert_menu.configure(values=["قرآن كريم", "ترددات يقظة", "أغاني حماسية"])

    def reset_trip(self):
        self.alert_system.reset_system()
        self.alarms_count_label.configure(text="Alarms Triggered: 0")
        self.sleep_timer_label.configure(text="Last Sleep: 0s")
        self.distance_label.configure(text="Distance: -- cm")
        self.status_label.configure(text="Status: Reset OK", text_color="green")

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Make sure the camera is not being used by another application.")
                self.video_frame.configure(text="Camera Error")
                return

            self.is_running = True
            self.video_frame.configure(text="")
            self.update_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.alert_system.stop_alarm()
        self.video_frame.configure(image=None, text="Camera is OFF")
        self.status_label.configure(text="Status: OFF", text_color="gray")

    def update_frame(self):
        if self.is_running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    fh, fw, _ = frame.shape
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)

                    eye_label = "open"
                    eye_conf = 1.0
                    yawn_label = "no_yawn"
                    yawn_conf = 1.0
                    distance_cm = 0

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        ear = calculate_ear(face_landmarks, frame.shape)

                        # =====================================================
                       # حساب المسافة بين الوجه والكاميرا
                        # =====================================================
                        lx, ly = int(face_landmarks.landmark[468].x * fw), int(face_landmarks.landmark[468].y * fh)
                        rx, ry = int(face_landmarks.landmark[473].x * fw), int(face_landmarks.landmark[473].y * fh)
                        
                        pixel_distance = math.hypot(rx - lx, ry - ly)
                        
                        if pixel_distance > 0:
                            distance_cm = (6.3 * 600) / pixel_distance
                        
                        self.distance_label.configure(text=f"Distance: {int(distance_cm)} cm")

                        left_eye_crop, left_eye_bbox = extract_region_from_landmarks(frame, face_landmarks, LEFT_EYE_IDXS, margin=0.25)
                        right_eye_crop, right_eye_bbox = extract_region_from_landmarks(frame, face_landmarks, RIGHT_EYE_IDXS, margin=0.25)
                        mouth_crop, mouth_bbox = extract_region_from_landmarks(frame, face_landmarks, MOUTH_IDXS, margin=0.30)

                        if left_eye_crop is not None and right_eye_crop is not None:
                            eye_label, eye_conf = get_combined_eye_prediction(self.eye_model, left_eye_crop, right_eye_crop)
                        
                        if ear > 0.28 and eye_label == "close":
                            eye_label = "open"
                            eye_conf = 1.0

                        if mouth_crop is not None:
                            yawn_label, yawn_conf = get_mouth_prediction(self.yawn_model, mouth_crop)
                            
                            if mouth_bbox is not None:
                                x, y, w, h = mouth_bbox
                                mouth_ratio = h / w  
                                
                                if yawn_label == "Yawn":
                                    if mouth_ratio >= 1.0:
                                        pass 
                                    else:
                                        yawn_label = "no_yawn"
                                        
                                draw_bbox(frame, mouth_bbox, color=(255, 255, 0))
                                cv2.putText(frame, f"MAR: {mouth_ratio:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                        if left_eye_crop is not None: draw_bbox(frame, left_eye_bbox, color=(255, 0, 0))
                        if right_eye_crop is not None: draw_bbox(frame, right_eye_bbox, color=(255, 0, 0))

                    status, message = self.engine.update(eye_label, eye_conf, yawn_label, yawn_conf)

                    self.eye_label.configure(text=f"Eyes Closed: {self.engine.eye_closed_frames} frames")
                    yawn_display = "Yes!" if yawn_label == "Yawn" else "No"
                    self.yawn_label.configure(text=f"Yawning: {yawn_display}")
                    
                    self.alarms_count_label.configure(text=f"Alarms Triggered: {self.alert_system.drowsy_count}")
                    current_sleep_secs = self.alert_system.get_sleep_duration()
                    self.sleep_timer_label.configure(text=f"Last Sleep: {current_sleep_secs}s")

                    if status == "DROWSY":
                        self.status_label.configure(text="DROWSY ALERT", text_color="red")
                        self.alert_system.trigger_alarm()
                    elif status == "YAWN":
                        self.status_label.configure(text="YAWNING", text_color="orange")
                        self.alert_system.stop_alarm()
                    else:
                        self.status_label.configure(text="Status: NORMAL", text_color="green")
                        self.alert_system.stop_alarm()

                    # =====================================================
                    #  التتبع التلقائي للوجه 
                    # =====================================================
                    TARGET_DISTANCE = 96.0
                    target_zoom = 1.0 

                    if distance_cm > TARGET_DISTANCE:
                        target_zoom = distance_cm / TARGET_DISTANCE
                        target_zoom = min(target_zoom, 3.0)

                    self.smooth_zoom = (0.1 * target_zoom) + (0.9 * self.smooth_zoom)

                    if self.smooth_zoom > 1.02:
                        cw = int(fw / self.smooth_zoom) 
                        ch = int(fh / self.smooth_zoom) 
                        
                        x1 = int((fw - cw) / 2)
                        y1 = int((fh - ch) / 2)
                        
                        display_frame = frame[y1:y1+ch, x1:x1+cw]
                        display_frame = cv2.resize(display_frame, (fw, fh))
                    else:
                        display_frame = frame

                    final_rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(final_rgb_image)
                    
                    self.current_image = ctk.CTkImage(light_image=pil_image, dark_image=pil_image, size=(640, 480))
                    self.video_frame.configure(image=self.current_image)

                self.after(15, self.update_frame)

            except Exception as e:
                print(f"An error occurred, camera stopped: {e}")
                self.stop_camera()

    def on_closing(self):
        self.stop_camera()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        self.destroy()

if __name__ == "__main__":
    app = DriverMonitoringApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()